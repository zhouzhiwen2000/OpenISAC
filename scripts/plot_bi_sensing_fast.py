
import numpy as np
import sys
import os
import platform
import subprocess
import time
import threading
from queue import Queue, Empty, Full
from collections import deque
import socket
import struct
import datetime
import scipy.io as sio

# PyQt6 + PyQtGraph Imports
from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

import zmq

from sensing_runtime_protocol import (
    CALIBRATE_SYSTEM_RESPONSE_COMMAND,
    CTRL_HEADER,
    PARAMS_COMMAND,
    READY_COMMAND,
    ViewerRuntimeParams,
    build_control_command,
    build_params_request,
    decode_sensing_payload,
    make_control_dealer,
    make_data_sub,
    make_tcp_endpoint,
    parse_params_packet,
    recv_sensing_frame,
)
from sensing_detection import (
    CleanParams,
    build_detection_views,
    estimate_clean_padding,
    run_local_psf_clean,
)
from viewer_panel_utils import (
    CollapsibleSection,
    ViewerSettings,
    VIEWER_COLORS,
    apply_viewer_theme,
    load_viewer_setting,
    sensing_colormap,
    set_button_active,
    style_spectrum_plot,
)

# ====== Backend Detection ======
USE_NVIDIA_GPU = False
USE_APPLE_GPU = False
USE_INTEL_GPU = False
cp = None
xp = np
mx = None
BACKEND_NAME = "CPU"
FORCED_BACKEND = os.environ.get("OPENISAC_BACKEND", "auto").strip().lower()

if FORCED_BACKEND not in {"auto", "cpu", "cuda", "apple", "mlx", "intel"}:
    print(f"Backend: Unknown OPENISAC_BACKEND={FORCED_BACKEND!r}; falling back to auto")
    FORCED_BACKEND = "auto"


def _backend_allowed(name):
    if name == "mlx":
        return FORCED_BACKEND in ("auto", "apple", "mlx")
    return FORCED_BACKEND in ("auto", name)


def _probe_mlx_backend(timeout_s=5.0):
    if sys.platform != "darwin" or platform.machine().lower() != "arm64":
        return False, "MLX backend requires Apple Silicon macOS"

    probe_code = (
        "import mlx.core as mx\n"
        "x = mx.array([1.0], dtype=mx.float32)\n"
        "y = mx.fft.fftshift(mx.array([[1+0j, 2+0j]], dtype=mx.complex64), axes=(1,))\n"
        "z = mx.fft.ifft(y, axis=1)\n"
        "_ = float(mx.abs(z).sum().item())\n"
        "print('mlx-ok')\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
            env=dict(os.environ, PYTHONNOUSERSITE="1"),
        )
    except Exception as exc:
        return False, f"MLX probe failed: {exc}"

    if result.returncode == 0 and "mlx-ok" in result.stdout:
        return True, "Apple GPU via MLX"

    stderr = (result.stderr or "").strip().splitlines()
    if stderr:
        detail = stderr[-1]
    else:
        detail = f"probe exited with code {result.returncode}"
    return False, f"MLX probe failed: {detail}"

if _backend_allowed("cuda"):
    try:
        import cupy as _cp
        try:
            _gpu_count = _cp.cuda.runtime.getDeviceCount()
            if _gpu_count > 0:
                USE_NVIDIA_GPU = True
                cp = _cp
                xp = _cp
                BACKEND_NAME = f"NVIDIA GPU via CuPy (devices: {_gpu_count})"
                print(f"Backend: Using {BACKEND_NAME}")
        except Exception as _e:
            print(f"Backend: CuPy available but CUDA init failed: {_e}")
    except ImportError:
        if FORCED_BACKEND == "cuda":
            print("Backend: Forced CUDA backend requested but CuPy is not installed")

if not USE_NVIDIA_GPU and _backend_allowed("mlx"):
    _mlx_ok, _mlx_msg = _probe_mlx_backend()
    if _mlx_ok:
        import mlx.core as _mx

        if not hasattr(_mx, "fft") or not hasattr(_mx.fft, "fftshift"):
            _mlx_msg = "MLX import succeeded but required FFT APIs are missing"
        else:
            USE_APPLE_GPU = True
            mx = _mx
            BACKEND_NAME = _mlx_msg
            print(f"Backend: Using {BACKEND_NAME}")

    if not USE_APPLE_GPU and FORCED_BACKEND in {"apple", "mlx"}:
        print(f"Backend: Forced MLX backend unavailable: {_mlx_msg}")

if not USE_NVIDIA_GPU and not USE_APPLE_GPU and _backend_allowed("intel"):
    def _try_intel_gpu():
        global USE_INTEL_GPU, cp, xp
        try:
            import dpnp as _dpnp
            import dpctl
            try:
                _intel_devices = [d for d in dpctl.get_devices() if d.is_gpu]
            except Exception:
                _intel_devices = []
            if _intel_devices:
                return _dpnp, _intel_devices[0].name
            return None, None
        except Exception:
            return None, None

    _intel_result = [None, None]

    def _intel_init_thread():
        _intel_result[0], _intel_result[1] = _try_intel_gpu()

    _init_thread = threading.Thread(target=_intel_init_thread, daemon=True)
    _init_thread.start()
    _init_thread.join(timeout=5.0)

    if not _init_thread.is_alive() and _intel_result[0] is not None:
        USE_INTEL_GPU = True
        cp = _intel_result[0]
        xp = _intel_result[0]
        BACKEND_NAME = f"Intel GPU via dpnp ({_intel_result[1]})"
        print(f"Backend: Using {BACKEND_NAME}")

USE_GPU = USE_NVIDIA_GPU or USE_APPLE_GPU or USE_INTEL_GPU
if not USE_GPU:
    print("Backend: No GPU acceleration available; using CPU")

HAVE_NUMBA = False
try:
    from numba import njit, prange
    HAVE_NUMBA = True
    print("Backend: Numba available for CPU acceleration")
except ImportError:
    print("Backend: Numba not available")

def to_numpy(arr):
    if hasattr(arr, 'get'):
        return arr.get()
    elif hasattr(arr, 'asnumpy'):
        return arr.asnumpy()
    elif hasattr(arr, '__array__'):
        return np.asarray(arr)
    return arr


def to_cpu_array(arr, dtype=None):
    host = to_numpy(arr)
    if dtype is None:
        return np.asarray(host)
    return np.asarray(host, dtype=dtype)

try:
    from scipy.signal import stft
    _SCIPY_AVAILABLE = True
except Exception:
    _SCIPY_AVAILABLE = False
    def stft(x, fs=1.0, window='hamming', nperseg=256, noverlap=192, nfft=256, return_onesided=False):
        x = np.asarray(x)
        if x.ndim != 1:
            x = x.ravel()
        step = nperseg - noverlap
        if isinstance(window, str):
            w = np.hamming(nperseg).astype(np.float32) if window.lower() == 'hamming' else np.ones(nperseg, dtype=np.float32)
        else:
            w = np.asarray(window).astype(np.float32)
        n_frames = 1 + max(0, (len(x) - nperseg) // step)
        Zxx = np.empty((nfft, n_frames), dtype=np.complex64)
        for i in range(n_frames):
            start = i * step
            seg = x[start:start + nperseg]
            if len(seg) < nperseg:
                seg = np.pad(seg, (0, nperseg - len(seg)))
            buf = np.zeros(nfft, dtype=np.complex64)
            buf[:nperseg] = (seg * w).astype(np.complex64)
            Zxx[:, i] = np.fft.fft(buf)
        f = np.fft.fftfreq(nfft, d=1.0 / fs)
        t = (np.arange(n_frames) * step) / fs
        return f, t, Zxx

# ====== Global Configuration ======
UDP_IP = "0.0.0.0"
UDP_PORT = 8889  # Bi-sensing uses port 8889
CONTROL_PORT = 10000
LOCAL_CONTROL_PORT = 0
FFT_SIZE = 1024
NUM_SYMBOLS = 100
MAX_CHUNK_SIZE = 60000
HEADER_SIZE = 12
SOCKET_BUFFER_SIZE = 8 * 1024 * 1024
MAX_RANGE_BIN = 1000
MAX_DOPPLER_BINS = 1000
RANGE_FFT_SIZE = 10240
DOPPLER_FFT_SIZE = 1000
PROCESS_RANGE_FFT_SIZE = RANGE_FFT_SIZE
PROCESS_DOPPLER_FFT_SIZE = DOPPLER_FFT_SIZE
DISPLAY_RANGE_BIN_LIMIT = MAX_RANGE_BIN
DISPLAY_DOPPLER_BIN_LIMIT = MAX_DOPPLER_BINS
enable_mti = True
enable_range_window = True
enable_doppler_window = True
cfar_enabled = False
cfar_train_doppler = 20
cfar_train_range = 20
cfar_guard_doppler = 10
cfar_guard_range = 10
cfar_alpha_db = float(10.0 * np.log10(50.0))
cfar_min_range_bin = 0
cfar_dc_exclusion_bins = 0
cfar_min_power_db = 0.0
cfar_max_points = 256
clean_loop_gain = 1.0
clean_max_targets = 64
clean_min_power_db = 10.0
CLEAN_PSF_THRESHOLD_DB = -35.0
CLEAN_MIN_HALF_WIDTH = 3
target_cluster_doppler_gap = 2
target_cluster_range_gap = 2

LEGACY_VIEWER_PARAMS = ViewerRuntimeParams(
    version=0,
    flags=0,
    frame_format=0,
    wire_rows=NUM_SYMBOLS,
    wire_cols=FFT_SIZE,
    active_rows=NUM_SYMBOLS,
    active_cols=FFT_SIZE,
    frame_symbol_period=NUM_SYMBOLS,
    range_fft_size=FFT_SIZE,
    doppler_fft_size=NUM_SYMBOLS,
    compact_mask_hash=0,
)

RAW_QUEUE_SIZE = 20
DISPLAY_QUEUE_SIZE = 5
DISPLAY_DOWNSAMPLE = 2

raw_frame_queue = Queue(maxsize=RAW_QUEUE_SIZE)
display_queue = Queue(maxsize=DISPLAY_QUEUE_SIZE)
startup_sync_queue = Queue(maxsize=8)
startup_sync_lock = threading.Lock()
startup_sync_queued_generation = None
buffer_lock = threading.Lock()
sender_ip = None
sender_lock = threading.Lock()
selected_range_bin = 0
show_micro_doppler = True
range_time_data = None
range_time_lock = threading.Lock()
current_display_data = {}
BUFFER_LENGTH = 5000
micro_doppler_buffer = deque(maxlen=BUFFER_LENGTH)
viewer_params = LEGACY_VIEWER_PARAMS
viewer_params_lock = threading.Lock()
last_param_error = None

# Minimal CLI so the backend host / ports can be overridden (defaults match
# the bi-sensing stream).
import argparse as _argparse
_parser = _argparse.ArgumentParser(description="Bistatic sensing viewer (fast)")
_parser.add_argument("--host", type=str, default=None, help="Backend host (ZeroMQ, overrides saved host)")
_parser.add_argument("--port", type=int, default=UDP_PORT, help="Bi-sensing stream port")
_parser.add_argument("--control-port", type=int, default=CONTROL_PORT, help="Control port")
_args, _ = _parser.parse_known_args()
_saved_host = load_viewer_setting("plot_bi_sensing_fast", "host")
HOST = str(_args.host or _saved_host or "127.0.0.1").strip() or "127.0.0.1"
UDP_PORT = int(_args.port)
CONTROL_PORT = int(_args.control_port)
# The backend host is a fixed connect target under ZeroMQ, so the control
# sender is known up front rather than discovered from incoming packets.
sender_ip = HOST

# ZeroMQ DEALER for the bidirectional control/params channel.
CONTROL_ENDPOINT = make_tcp_endpoint(HOST, CONTROL_PORT)
control_socket = make_control_dealer(CONTROL_ENDPOINT, identity=f"bi-sensing-{os.getpid()}")
LOCAL_CONTROL_PORT = 0
control_socket_lock = threading.Lock()
control_tx_queue = Queue(maxsize=256)
connection_generation = 0


def get_processing_range_fft_size():
    return max(1, int(PROCESS_RANGE_FFT_SIZE))


def get_processing_doppler_fft_size():
    return max(1, int(PROCESS_DOPPLER_FFT_SIZE))


def get_display_range_bin_limit():
    return max(1, int(DISPLAY_RANGE_BIN_LIMIT))


def get_display_doppler_bin_limit():
    return max(1, int(DISPLAY_DOPPLER_BIN_LIMIT))


def get_local_clean_params():
    return CleanParams(
        loop_gain=float(clean_loop_gain),
        max_targets=max(1, int(clean_max_targets)),
        min_power_db=float(clean_min_power_db),
        min_range_bin=max(0, int(cfar_min_range_bin)),
        dc_exclusion_bins=max(0, int(cfar_dc_exclusion_bins)),
        psf_threshold_db=float(CLEAN_PSF_THRESHOLD_DB),
        min_half_width=max(0, int(CLEAN_MIN_HALF_WIDTH)),
    )


def cluster_detected_targets(cfar_points, rd_data, point_strengths_db=None):
    if cfar_points is None:
        return []

    points = np.asarray(cfar_points, dtype=np.int32)
    if points.size == 0 or rd_data is None:
        return []

    rows, cols = rd_data.shape
    in_bounds = (
        (points[:, 0] >= 0)
        & (points[:, 0] < rows)
        & (points[:, 1] >= 0)
        & (points[:, 1] < cols)
    )
    points = points[in_bounds]
    if point_strengths_db is not None:
        point_strengths_db = np.asarray(point_strengths_db, dtype=np.float32)
        if point_strengths_db.shape[0] != in_bounds.shape[0]:
            point_strengths_db = None
        else:
            point_strengths_db = point_strengths_db[in_bounds]
    if points.size == 0:
        return []

    n_points = points.shape[0]
    visited = np.zeros(n_points, dtype=bool)
    clusters = []

    for seed_idx in range(n_points):
        if visited[seed_idx]:
            continue

        queue = [seed_idx]
        visited[seed_idx] = True
        cluster_indices = []

        while queue:
            cur_idx = queue.pop()
            cluster_indices.append(cur_idx)
            cur_d = points[cur_idx, 0]
            cur_r = points[cur_idx, 1]

            for nbr_idx in range(n_points):
                if visited[nbr_idx]:
                    continue
                if (
                    abs(int(points[nbr_idx, 0]) - int(cur_d)) <= target_cluster_doppler_gap
                    and abs(int(points[nbr_idx, 1]) - int(cur_r)) <= target_cluster_range_gap
                ):
                    visited[nbr_idx] = True
                    queue.append(nbr_idx)

        cluster_points = points[np.asarray(cluster_indices, dtype=np.int32)]
        if point_strengths_db is not None:
            strengths_db = point_strengths_db[np.asarray(cluster_indices, dtype=np.int32)]
        else:
            strengths_db = rd_data[cluster_points[:, 0], cluster_points[:, 1]]
        peak_idx = int(np.argmax(strengths_db))
        peak_point = cluster_points[peak_idx]
        peak_strength_db = float(strengths_db[peak_idx])
        linear_weights = np.power(10.0, strengths_db.astype(np.float64) / 20.0)
        weight_sum = float(np.sum(linear_weights))
        if weight_sum > 0.0:
            centroid_d = float(np.sum(cluster_points[:, 0] * linear_weights) / weight_sum)
            centroid_r = float(np.sum(cluster_points[:, 1] * linear_weights) / weight_sum)
        else:
            centroid_d = float(np.mean(cluster_points[:, 0]))
            centroid_r = float(np.mean(cluster_points[:, 1]))

        clusters.append({
            'peak_doppler_idx': int(peak_point[0]),
            'peak_range_idx': int(peak_point[1]),
            'peak_strength_db': peak_strength_db,
            'cluster_size': int(cluster_points.shape[0]),
            'centroid_doppler_idx': centroid_d,
            'centroid_range_idx': centroid_r,
        })

    clusters.sort(key=lambda item: item['peak_strength_db'], reverse=True)
    return clusters


def reset_processing_state():
    global current_display_data, range_time_data
    with range_time_lock:
        range_time_data = None
    with buffer_lock:
        micro_doppler_buffer.clear()
    while True:
        try:
            display_queue.get_nowait()
        except Exception:
            break
    current_display_data = {}


def _clear_control_tx_queue():
    while True:
        try:
            control_tx_queue.get_nowait()
        except Empty:
            break
        except Exception:
            break


def queue_startup_backend_sync():
    global startup_sync_queued_generation
    generation = connection_generation
    with startup_sync_lock:
        if startup_sync_queued_generation == generation:
            return
        startup_sync_queued_generation = generation
    try:
        startup_sync_queue.put_nowait(generation)
    except Exception:
        pass


def get_viewer_params():
    with viewer_params_lock:
        return viewer_params


def set_viewer_params(params):
    global viewer_params, last_param_error
    with viewer_params_lock:
        viewer_params = params
    last_param_error = None


def reconnect_backend(host):
    global HOST, sender_ip, CONTROL_ENDPOINT, control_socket, connection_generation
    global last_param_error, range_time_data
    host = (host or "").strip() or "127.0.0.1"
    new_endpoint = make_tcp_endpoint(host, CONTROL_PORT)
    new_socket = make_control_dealer(
        new_endpoint,
        identity=f"bi-sensing-{os.getpid()}-{int(time.time() * 1000)}",
    )
    with control_socket_lock:
        old_socket = control_socket
        HOST = host
        sender_ip = host
        CONTROL_ENDPOINT = new_endpoint
        control_socket = new_socket
        connection_generation += 1
        _clear_control_tx_queue()
        try:
            old_socket.close()
        except Exception:
            pass

    set_viewer_params(LEGACY_VIEWER_PARAMS)
    last_param_error = None
    while not raw_frame_queue.empty():
        try:
            raw_frame_queue.get_nowait()
        except Exception:
            break
    with range_time_lock:
        range_time_data = None
    with micro_lock:
        micro_doppler_buffer.clear()
    print(f"[CTRL] Reconnected viewer sockets to backend {HOST}")
    request_viewer_params()


def backend_stream_unsupported(params):
    return params.backend_processing() or params.metadata_sidecar()


def warn_unsupported_backend_params(params):
    print(
        "Warning: fast bi-sensing viewer does not support backend sensing output; "
        f"ignoring backend stream until sender returns to local/raw mode. Params: {params.describe()}"
    )


def warn_unsupported_backend_payload(kind, frame_id=None):
    frame_text = "" if frame_id is None else f" frame {int(frame_id)}"
    print(
        "Warning: fast bi-sensing viewer does not support backend sensing output; "
        f"dropping backend {kind}{frame_text}. Use plot_backend_bi_sensing.py instead."
    )

class FrameBuffer:
    def __init__(self):
        self.frame_id = 0
        self.total_chunks = 0
        self.buffer = [None] * 1024
        self.received_chunks = 0
    
    def init(self, frame_id, total_chunks):
        self.frame_id = frame_id
        self.total_chunks = total_chunks
        self.buffer = [None] * total_chunks
        self.received_chunks = 0
    
    def add_chunk(self, chunk_id, data):
        if chunk_id < self.total_chunks and self.buffer[chunk_id] is None:
            self.buffer[chunk_id] = data
            self.received_chunks += 1
            return self.received_chunks == self.total_chunks
        return False
    
    def assemble_frame(self, runtime_params):
        byte_data = b''.join(self.buffer[:self.total_chunks])
        decoded = decode_sensing_payload(self.frame_id, byte_data, runtime_params)
        return decoded.frame_id, decoded.matrix

current_frame = FrameBuffer()
current_metadata_frame = FrameBuffer()


def _control_endpoint():
    with sender_lock:
        if sender_ip is None:
            return None
        return sender_ip, CONTROL_PORT


def _send_control_packet(packet, endpoint, error_prefix):
    if endpoint is None:
        return False
    try:
        control_tx_queue.put_nowait((packet, endpoint, error_prefix))
        return True
    except Full:
        try:
            control_tx_queue.get_nowait()
        except Exception:
            pass
        try:
            control_tx_queue.put_nowait((packet, endpoint, error_prefix))
            return True
        except Exception as e:
            print(f"{error_prefix}: control TX queue full: {e}")
            return False
    except Exception as e:
        print(f"{error_prefix}: {e}")
        return False


def _flush_control_tx_queue(local_socket):
    while True:
        try:
            packet, _endpoint, error_prefix = control_tx_queue.get_nowait()
        except Empty:
            break
        except Exception:
            break
        try:
            local_socket.send(packet, flags=zmq.NOBLOCK)
        except zmq.Again:
            try:
                control_tx_queue.put_nowait((packet, _endpoint, error_prefix))
            except Exception:
                print(f"{error_prefix} via {CONTROL_ENDPOINT}: send would block")
            break
        except Exception as e:
            print(f"{error_prefix} via {CONTROL_ENDPOINT}: {e}")


def send_skip_command():
    global sender_ip
    with sender_lock:
        has_sender = sender_ip is not None
    if not has_sender:
        print("Error: Cannot send command - sender IP not detected.")
        return
    try:
        request_viewer_params()
        command_data = build_control_command(b"SKIP", 1)
        if _send_control_packet(command_data, _control_endpoint(), "Failed to send skip FFT command"):
            print(f"Sent skip FFT command to {sender_ip}: SKIP 1 from control port {LOCAL_CONTROL_PORT}")
        request_viewer_params()
    except Exception as e:
        print(f"Failed to send skip FFT command: {str(e)}")


def request_viewer_params():
    global sender_ip
    if sender_ip is None:
        return
    _send_control_packet(build_params_request(0), _control_endpoint(), "Failed to request viewer params")


def handle_control_packet(data, addr, prefix):
    global sender_ip, CONTROL_PORT, last_param_error
    if len(data) < 8 or data[:4] != CTRL_HEADER:
        return False

    command = data[4:8]
    if addr is not None:
        with sender_lock:
            if sender_ip is None:
                sender_ip = addr[0]
                print(f"Detected control sender IP: {sender_ip}")
            if CONTROL_PORT != addr[1]:
                CONTROL_PORT = addr[1]
    if command == PARAMS_COMMAND:
        params = parse_params_packet(data)
        if params is not None:
            set_viewer_params(params)
            print(f"{prefix} Viewer params: {params.describe()}")
            if backend_stream_unsupported(params):
                warn_unsupported_backend_params(params)
            queue_startup_backend_sync()
    elif command == READY_COMMAND:
        if get_viewer_params().version == 0 or last_param_error is not None:
            request_viewer_params()
    return True


def control_receiver():
    print(f"[CTRL] DEALER connected to {CONTROL_ENDPOINT} for viewer params/control replies")
    poller = None
    local_socket = None
    local_generation = -1
    while True:
        try:
            # The DEALER socket is not thread-safe, so this thread owns actual
            # send/recv. UI callbacks enqueue packets instead of touching it.
            with control_socket_lock:
                if local_generation != connection_generation:
                    local_socket = control_socket
                    local_generation = connection_generation
                    poller = zmq.Poller()
                    poller.register(local_socket, zmq.POLLIN)
                    print(f"[CTRL] DEALER connected to {CONTROL_ENDPOINT} for viewer params/control replies")
                _flush_control_tx_queue(local_socket)
                events = dict(poller.poll(timeout=20))
                if local_socket not in events:
                    continue
                parts = local_socket.recv_multipart(flags=zmq.NOBLOCK)
            data = parts[-1] if parts else b""
            handle_control_packet(data, None, "[CTRL]")
        except zmq.Again:
            continue
        except Exception as e:
            print(f"[CTRL] Receiver error: {e}")

def udp_receiver():
    global sender_ip, CONTROL_PORT, last_param_error
    sock = None
    poller = None
    local_generation = -1

    last_param_retry_t = 0.0
    param_retry_interval = 0.5

    def request_params_if_needed():
        nonlocal last_param_retry_t
        if get_viewer_params().version != 0:
            return
        now = time.monotonic()
        if now - last_param_retry_t < param_retry_interval:
            return
        request_viewer_params()
        last_param_retry_t = now

    # Each ZMQ multipart message is one whole frame: part 0 = data payload,
    # optional part 1 = metadata sidecar (unsupported by this fast viewer).
    frame_counter = 0
    while True:
        if local_generation != connection_generation:
            if sock is not None:
                try:
                    sock.close(0)
                except Exception:
                    pass
            endpoint = make_tcp_endpoint(HOST, UDP_PORT)
            try:
                sock = make_data_sub(endpoint, rcvhwm=4)
            except Exception as e:
                print(f"Sensing SUB connect error: {e}")
                time.sleep(0.5)
                continue
            poller = zmq.Poller()
            poller.register(sock, zmq.POLLIN)
            local_generation = connection_generation
            print(f"Subscribed to bi-sensing stream at {endpoint}")

        try:
            events = dict(poller.poll(timeout=500))
            if sock not in events:
                request_params_if_needed()
                continue

            frame = recv_sensing_frame(sock, flags=zmq.NOBLOCK)
            if frame is None:
                continue
            data_payload, metadata_payload = frame
            request_params_if_needed()

            if get_viewer_params().version == 0:
                request_params_if_needed()
                continue

            runtime_params = get_viewer_params()
            if metadata_payload is not None:
                warn_unsupported_backend_payload("metadata sidecar", None)
                continue
            if backend_stream_unsupported(runtime_params):
                warn_unsupported_backend_payload("sensing frame", None)
                continue
            if not data_payload:
                continue

            frame_counter += 1
            decoded = decode_sensing_payload(frame_counter, data_payload, runtime_params)
            frame_data = (decoded.frame_id, decoded.matrix)
            if raw_frame_queue.full():
                try:
                    raw_frame_queue.get_nowait()
                except Exception:
                    pass
            raw_frame_queue.put(frame_data)
        except Exception as e:
            print(f"Unexpected error in receiver: {str(e)}")
            last_param_error = str(e)
            request_viewer_params()

control_thread = threading.Thread(target=control_receiver, daemon=True)
control_thread.start()

receiver_thread = threading.Thread(target=udp_receiver, daemon=True)
receiver_thread.start()

# FFT Processing
_range_window_cache = {}
_doppler_window_cache = {}


def get_range_window(length):
    length = max(1, int(length))
    cached = _range_window_cache.get(length)
    if cached is not None:
        return cached
    base = np.hamming(length).reshape(1, length).astype(np.float32)
    if USE_NVIDIA_GPU or USE_INTEL_GPU:
        cached = cp.array(base, dtype=cp.float32)
    elif USE_APPLE_GPU:
        cached = mx.array(base)
    else:
        cached = base
    _range_window_cache[length] = cached
    return cached


def get_doppler_window(length):
    length = max(1, int(length))
    cached = _doppler_window_cache.get(length)
    if cached is not None:
        return cached
    base = np.hamming(length).reshape(length, 1).astype(np.float32)
    if USE_NVIDIA_GPU or USE_INTEL_GPU:
        cached = cp.array(base, dtype=cp.float32)
    elif USE_APPLE_GPU:
        cached = mx.array(base)
    else:
        cached = base
    _doppler_window_cache[length] = cached
    return cached

if HAVE_NUMBA:
    @njit(parallel=True, fastmath=True)
    def cpu_prep_range_fft(frame_data, window, fft_size, range_fft_size):
        num_symbols = frame_data.shape[0]
        half_n = fft_size // 2
        out = np.zeros((num_symbols, range_fft_size), dtype=np.complex64)
        for i in prange(num_symbols):
            for j in range(half_n):
                out[i, j] = frame_data[i, j + half_n] * window[0, j]
            for j in range(half_n):
                out[i, j + half_n] = frame_data[i, j] * window[0, j + half_n]
        return out

    @njit(parallel=True, fastmath=True)
    def cpu_prep_doppler_fft(range_time, window, num_symbols, doppler_fft_size, range_fft_size):
        out = np.zeros((doppler_fft_size, range_fft_size), dtype=np.complex64)
        for i in prange(num_symbols):
            for j in range(range_fft_size):
                out[i, j] = range_time[i, j] * window[i, 0]
        return out

    @njit(parallel=True, fastmath=True)
    def cpu_calc_mag_db(doppler_fft_data, num_symbols, fft_size):
        rows = doppler_fft_data.shape[0]
        cols = doppler_fft_data.shape[1]
        half_rows = rows // 2
        out = np.empty((rows, cols), dtype=np.float32)
        norm_factor = 1.0 / np.sqrt(num_symbols * fft_size)
        for j in prange(cols):
            for i in range(half_rows):
                out[i, j] = 20.0 * np.log10(np.abs(doppler_fft_data[i + half_rows, j]) * norm_factor + 1e-12)
            for i in range(half_rows):
                out[i + half_rows, j] = 20.0 * np.log10(np.abs(doppler_fft_data[i, j]) * norm_factor + 1e-12)
        return out

def process_range_doppler(frame_data, runtime_params, max_view_range_bins=None):
    if backend_stream_unsupported(runtime_params):
        raise ValueError(
            "Backend sensing output is not supported in plot_bi_sensing_fast.py"
        )

    if runtime_params.is_dense_range_doppler():
        rd_complex = np.asarray(
            frame_data[:runtime_params.wire_rows, :runtime_params.wire_cols],
            dtype=np.complex64,
        )
        rd_shifted = np.fft.fftshift(rd_complex, axes=0)
        magnitude_db = 20.0 * np.log10(np.abs(rd_shifted) + 1e-12)
        return magnitude_db.astype(np.float32, copy=False), None, rd_shifted.astype(np.complex64, copy=False)

    if not runtime_params.raw_fft_locally_supported():
        raise ValueError(f"Unsupported sensing frame format: {runtime_params.describe()}")

    raw_rows = max(1, int(runtime_params.active_rows))
    raw_cols = max(1, int(runtime_params.active_cols))
    range_fft_size = max(raw_cols, get_processing_range_fft_size())
    doppler_fft_size = max(raw_rows, get_processing_doppler_fft_size())
    raw_frame = np.asarray(frame_data[:raw_rows, :raw_cols], dtype=np.complex64)

    if max_view_range_bins is None:
        max_view_range_bins = get_display_range_bin_limit()
    max_view_range_bins = min(max_view_range_bins, range_fft_size)

    if USE_NVIDIA_GPU or USE_INTEL_GPU:
        range_win = get_range_window(raw_cols) if enable_range_window else cp.ones((1, raw_cols), dtype=cp.float32)
        doppler_win = get_doppler_window(raw_rows) if enable_doppler_window else cp.ones((raw_rows, 1), dtype=cp.float32)
        frame_data_gpu = cp.asarray(raw_frame, dtype=cp.complex64)
        shifted_data = cp.fft.fftshift(frame_data_gpu, axes=1)
        windowed_data = shifted_data * range_win
        padded_data = cp.zeros((raw_rows, range_fft_size), dtype=cp.complex64)
        padded_data[:, :raw_cols] = windowed_data
        range_time = cp.fft.ifft(padded_data, axis=1) * range_fft_size
        range_time_view = range_time[:, :max_view_range_bins] if max_view_range_bins < range_fft_size else range_time
        doppler_windowed = range_time_view * doppler_win
        view_width = range_time_view.shape[1]
        padded_doppler = cp.zeros((doppler_fft_size, view_width), dtype=cp.complex64)
        padded_doppler[:raw_rows, :] = doppler_windowed
        doppler_fft = cp.fft.fft(padded_doppler, axis=0)
        doppler_shifted = cp.fft.fftshift(doppler_fft, axes=0)
        magnitude = cp.abs(doppler_shifted) / np.sqrt(raw_rows * raw_cols)
        magnitude_db = 20 * cp.log10(magnitude + 1e-12)
        return (
            magnitude_db.astype(cp.float32, copy=False),
            range_time_view.astype(cp.complex64, copy=False),
            doppler_shifted.astype(cp.complex64, copy=False),
        )
    elif USE_APPLE_GPU:
        range_win = get_range_window(raw_cols) if enable_range_window else mx.ones((1, raw_cols), dtype=mx.float32)
        doppler_win = get_doppler_window(raw_rows) if enable_doppler_window else mx.ones((raw_rows, 1), dtype=mx.float32)
        frame_data_gpu = mx.array(raw_frame, dtype=mx.complex64)
        shifted_data = mx.fft.fftshift(frame_data_gpu, axes=(1,))
        windowed_data = shifted_data * range_win
        range_time = mx.fft.ifft(windowed_data, n=range_fft_size, axis=1) * range_fft_size
        range_time_view = range_time[:, :max_view_range_bins] if max_view_range_bins < range_fft_size else range_time
        doppler_windowed = range_time_view * doppler_win
        doppler_fft = mx.fft.fft(doppler_windowed, n=doppler_fft_size, axis=0)
        doppler_shifted = mx.fft.fftshift(doppler_fft, axes=(0,))
        magnitude = mx.abs(doppler_shifted) / np.sqrt(raw_rows * raw_cols)
        magnitude_db = 20.0 * mx.log10(magnitude + 1e-12)
        return (
            np.array(magnitude_db, copy=False).astype(np.float32, copy=False),
            np.array(range_time_view, copy=False).astype(np.complex64, copy=False),
            np.array(doppler_shifted, copy=False).astype(np.complex64, copy=False),
        )
    elif HAVE_NUMBA:
        range_win = get_range_window(raw_cols) if enable_range_window else np.ones((1, raw_cols), dtype=np.float32)
        doppler_win = get_doppler_window(raw_rows) if enable_doppler_window else np.ones((raw_rows, 1), dtype=np.float32)
        padded_data = cpu_prep_range_fft(raw_frame, range_win, raw_cols, range_fft_size)
        range_time = np.fft.ifft(padded_data, axis=1) * range_fft_size
        range_time_view = range_time[:, :max_view_range_bins] if max_view_range_bins < range_fft_size else range_time
        view_width = range_time_view.shape[1]
        padded_doppler = cpu_prep_doppler_fft(range_time_view, doppler_win, raw_rows, doppler_fft_size, view_width)
        doppler_fft = np.fft.fft(padded_doppler, axis=0)
        doppler_shifted = np.fft.fftshift(doppler_fft, axes=0)
        magnitude_db = 20.0 * np.log10(np.abs(doppler_shifted) / np.sqrt(raw_rows * raw_cols) + 1e-12)
        return (
            magnitude_db.astype(np.float32, copy=False),
            range_time_view.astype(np.complex64, copy=False),
            doppler_shifted.astype(np.complex64, copy=False),
        )
    else:
        range_win = get_range_window(raw_cols) if enable_range_window else np.ones((1, raw_cols), dtype=np.float32)
        doppler_win = get_doppler_window(raw_rows) if enable_doppler_window else np.ones((raw_rows, 1), dtype=np.float32)
        shifted_data = np.fft.fftshift(raw_frame, axes=1)
        windowed_data = shifted_data * range_win
        padded_data = np.zeros((raw_rows, range_fft_size), dtype=np.complex64)
        padded_data[:, :raw_cols] = windowed_data
        range_time = np.fft.ifft(padded_data, axis=1) * range_fft_size
        range_time_view = range_time[:, :max_view_range_bins] if max_view_range_bins < range_fft_size else range_time
        view_width = range_time_view.shape[1]
        doppler_windowed = range_time_view * doppler_win
        padded_doppler = np.zeros((doppler_fft_size, view_width), dtype=np.complex64)
        padded_doppler[:raw_rows, :] = doppler_windowed
        doppler_fft = np.fft.fft(padded_doppler, axis=0)
        doppler_shifted = np.fft.fftshift(doppler_fft, axes=0)
        magnitude_db = 20 * np.log10(np.abs(doppler_shifted) / np.sqrt(raw_rows * raw_cols) + 1e-12)
        return (
            magnitude_db.astype(np.float32, copy=False),
            range_time_view.astype(np.complex64, copy=False),
            doppler_shifted.astype(np.complex64, copy=False),
        )

def accumulate_range_time_data():
    with range_time_lock:
        if range_time_data is not None and range_time_data.size > 0:
            range_idx = min(selected_range_bin, range_time_data.shape[1] - 1)
            with buffer_lock:
                micro_doppler_buffer.extend(
                    to_cpu_array(range_time_data[:, range_idx], dtype=np.complex64)
                )

def calculate_micro_doppler():
    if len(micro_doppler_buffer) < 256:
        return None, None, None
    with buffer_lock:
        complex_signal = np.array(micro_doppler_buffer)
    f, t, Zxx = stft(complex_signal, fs=1.0, window='hamming', nperseg=256, noverlap=192, nfft=256, return_onesided=False)
    Pxx_db = 20 * np.log10(np.abs(Zxx) + 1e-12)
    Pxx_db_shifted = np.fft.fftshift(Pxx_db, axes=0)
    f_shifted = np.fft.fftshift(f)
    f_idx = (f_shifted > -0.5) & (f_shifted < 0.5)
    return f_shifted[f_idx], t, Pxx_db_shifted[f_idx, :]

def dsp_worker():
    global range_time_data
    print("DSP Worker started")
    # Power calculation for bi-sensing
    accumulated_power = 0.0
    power_frame_count = 0
    MAX_POWER_FRAMES = 100
    
    while True:
        try:
            try:
                raw_item = raw_frame_queue.get(timeout=0.1)
            except:
                continue

            if isinstance(raw_item, tuple) and len(raw_item) == 2:
                frame_id, raw_frame = raw_item
            else:
                frame_id, raw_frame = -1, raw_item
            
            # Power calculation
            current_power = np.mean(np.abs(raw_frame)**2)
            accumulated_power += current_power
            power_frame_count += 1
            if power_frame_count >= MAX_POWER_FRAMES:
                avg_power = accumulated_power / MAX_POWER_FRAMES
                power_dbm = 10 * np.log10(avg_power + 1e-12)
                print(f"Average Power (last {MAX_POWER_FRAMES} frames): {power_dbm:.2f} dBm")
                accumulated_power = 0.0
                power_frame_count = 0
            
            t_start = time.time()
            runtime_params = get_viewer_params()
            if backend_stream_unsupported(runtime_params):
                warn_unsupported_backend_payload("sensing frame", frame_id)
                continue
            rd_spectrum, range_time_view, rd_complex = process_range_doppler(raw_frame, runtime_params)
            display_range_bins = get_display_range_bin_limit()
            display_doppler_bins = get_display_doppler_bin_limit()
            raw_rows = max(1, int(runtime_params.active_rows))
            raw_cols = max(1, int(runtime_params.active_cols))
            range_fft_size = max(raw_cols, get_processing_range_fft_size())
            doppler_fft_size = max(raw_rows, get_processing_doppler_fft_size())
            clean_params = get_local_clean_params()
            clean_pad_rows, clean_pad_cols = estimate_clean_padding(
                raw_rows=raw_rows,
                raw_cols=raw_cols,
                range_fft_size=range_fft_size,
                doppler_fft_size=doppler_fft_size,
                downsample=DISPLAY_DOWNSAMPLE,
                enable_range_window=enable_range_window,
                enable_doppler_window=enable_doppler_window,
                params=clean_params,
            )
            rd_spectrum_plot, _, cfar_row_offset, cfar_col_offset = build_detection_views(
                rd_spectrum,
                display_range_bins,
                display_doppler_bins,
                DISPLAY_DOWNSAMPLE,
                pad_doppler_bins=clean_pad_rows,
                pad_range_bins=clean_pad_cols,
            )
            _, rd_complex_clean, _, _ = build_detection_views(
                rd_complex,
                display_range_bins,
                display_doppler_bins,
                DISPLAY_DOWNSAMPLE,
                pad_doppler_bins=clean_pad_rows,
                pad_range_bins=clean_pad_cols,
            )
            rd_spectrum_plot_cpu = to_cpu_array(rd_spectrum_plot, dtype=np.float32)
            rd_complex_clean_cpu = to_cpu_array(rd_complex_clean, dtype=np.complex64)
            cfar_points = np.empty((0, 2), dtype=np.int32)
            cfar_hits = 0
            cfar_shown_hits = 0
            cfar_backend = "off"
            cfar_stats = None
            clean_component_points = np.empty((0, 2), dtype=np.int32)
            clean_component_strength_db = np.empty((0,), dtype=np.float32)
            if cfar_enabled and rd_spectrum_plot.size > 0:
                (
                    clean_component_points,
                    clean_component_strength_db,
                    cfar_hits,
                    cfar_shown_hits,
                    cfar_backend,
                    cfar_stats,
                ) = run_local_psf_clean(
                    rd_complex_clean_cpu,
                    clean_params,
                    raw_rows=raw_rows,
                    raw_cols=raw_cols,
                    range_fft_size=range_fft_size,
                    doppler_fft_size=doppler_fft_size,
                    downsample=DISPLAY_DOWNSAMPLE,
                    enable_range_window=enable_range_window,
                    enable_doppler_window=enable_doppler_window,
                    active_row_start=cfar_row_offset,
                    active_row_stop=cfar_row_offset + rd_spectrum_plot_cpu.shape[0],
                    active_col_start=cfar_col_offset,
                    active_col_stop=cfar_col_offset + rd_spectrum_plot_cpu.shape[1],
                    dc_center_row=cfar_row_offset + (rd_spectrum_plot_cpu.shape[0] // 2),
                )
                if clean_component_points.size > 0:
                    clean_component_points = clean_component_points - np.array(
                        [cfar_row_offset, cfar_col_offset],
                        dtype=np.int32,
                    )
                    in_bounds = (
                        (clean_component_points[:, 0] >= 0)
                        & (clean_component_points[:, 0] < rd_spectrum_plot_cpu.shape[0])
                        & (clean_component_points[:, 1] >= 0)
                        & (clean_component_points[:, 1] < rd_spectrum_plot_cpu.shape[1])
                    )
                    clean_component_points = clean_component_points[in_bounds]
                    clean_component_strength_db = clean_component_strength_db[in_bounds]
            range_time_view_cpu = None
            if range_time_view is not None:
                range_time_view_cpu = to_cpu_array(range_time_view, dtype=np.complex64)
            with range_time_lock:
                range_time_data = range_time_view_cpu
            clusters = (
                cluster_detected_targets(
                    clean_component_points,
                    rd_spectrum_plot_cpu,
                    point_strengths_db=clean_component_strength_db,
                )
                if clean_component_points.size > 0 else []
            )
            if clusters:
                cfar_points = np.asarray(
                    [
                        (int(cluster['peak_doppler_idx']), int(cluster['peak_range_idx']))
                        for cluster in clusters
                    ],
                    dtype=np.int32,
                )
                cfar_shown_hits = int(cfar_points.shape[0])
            else:
                cfar_points = clean_component_points
            md_spectrum = None
            if show_micro_doppler:
                accumulate_range_time_data()
                f, t, Pxx = calculate_micro_doppler()
                if Pxx is not None:
                    md_spectrum = Pxx.T  # Transpose: Y=Time, X=Frequency
            dsp_time = time.time() - t_start
            result = {
                'frame_id': int(frame_id),
                'raw': raw_frame,
                'rd': rd_spectrum_plot_cpu,
                'cfar_points': cfar_points,
                'cfar_hits': cfar_hits,
                'cfar_shown_hits': cfar_shown_hits,
                'cfar_backend': cfar_backend,
                'cfar_stats': cfar_stats,
                'detector_mode': 'local_clean',
                'target_clusters': clusters,
                'md': md_spectrum,
                'dsp_time': dsp_time,
                'viewer_params': runtime_params,
            }
            if display_queue.full():
                try:
                    display_queue.get_nowait()
                except:
                    pass
            display_queue.put(result)
        except Exception as e:
            print(f"DSP Worker Error: {e}")
            import traceback
            traceback.print_exc()

dsp_thread = threading.Thread(target=dsp_worker, daemon=True)
dsp_thread.start()

# Command Functions
def send_command(header, cmd_id, value):
    global sender_ip
    if sender_ip is None:
        print("Error: Cannot send command - sender IP not detected.")
        return
    packet = struct.pack("!4s4si", header, cmd_id, int(value))
    if _send_control_packet(packet, _control_endpoint(), "Failed to send command"):
        print(f"Sent command {cmd_id} to {sender_ip}: {value} from control port {LOCAL_CONTROL_PORT}")

def send_alignment_command(val):
    send_command(b"CMD ", b"ALGN", val)

def send_strd_command(val):
    try:
        strd_val = max(1, min(get_viewer_params().max_strd_value(), int(val)))
        send_command(b"CMD ", b"STRD", strd_val)
        request_viewer_params()
    except ValueError:
        print(f"Invalid STRD value: {val}")

def send_mti_command(enabled):
    send_command(b"CMD ", b"MTI ", 1 if enabled else 0)


def send_startup_backend_sync(strd_value, mti_enabled):
    print(f"[CTRL] Syncing startup backend controls: SKIP=1 MTI={1 if mti_enabled else 0} STRD={strd_value}")
    send_skip_command()
    time.sleep(0.02)
    send_mti_command(mti_enabled)
    time.sleep(0.02)
    send_strd_command(strd_value)


def send_system_response_calibration_command():
    send_command(b"CMD ", CALIBRATE_SYSTEM_RESPONSE_COMMAND, 0)
    print("Requested bistatic Hsys calibration. Keep RF TX/RX directly connected until backend reports completion.")

# ====== MainWindow with PyQt6 + PyQtGraph ======
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenISAC Bi-Sensing - PyQtGraph")
        self.resize(1600, 900)
        self._control_panel_width = 420
        self._status_label_text_width = self._control_panel_width - 24
        self.targets_window = QtWidgets.QWidget()
        self.targets_window.setWindowTitle("OpenISAC Bi-Sensing Targets")
        self.targets_window.resize(520, 220)
        targets_layout = QtWidgets.QVBoxLayout(self.targets_window)
        self.targets_text = QtWidgets.QPlainTextEdit()
        self.targets_text.setReadOnly(True)
        self.targets_text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        targets_layout.addWidget(self.targets_text)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # Plot Area
        plot_widget = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_widget)
        main_layout.addWidget(plot_widget, stretch=4)

        # Range-Doppler Plot
        self.rd_plot = pg.PlotWidget(title="Range-Doppler Spectrum (Bi-Sensing)")
        style_spectrum_plot(self.rd_plot)
        self.rd_img = pg.ImageItem()
        self.rd_plot.addItem(self.rd_img)
        self.rd_plot.setLabel('left', 'Doppler Bin')
        self.rd_plot.setLabel('bottom', 'Range Bin')
        spectrum_cmap = sensing_colormap()
        self.rd_img.setLookupTable(spectrum_cmap.getLookupTable())
        self.rd_colorbar = pg.ColorBarItem(values=(-40, 20), colorMap=spectrum_cmap, interactive=False)
        self.rd_colorbar.setImageItem(self.rd_img, insert_in=self.rd_plot.plotItem)
        self.rd_cfar_marker = pg.ScatterPlotItem(
            [], [], symbol='o', size=7,
            pen=pg.mkPen(VIEWER_COLORS["accent"], width=1.7),
            brush=pg.mkBrush(56, 189, 248, 55)
        )
        self.rd_plot.addItem(self.rd_cfar_marker)
        plot_layout.addWidget(self.rd_plot)

        # Micro-Doppler Plot
        self.md_plot = pg.PlotWidget(title="Micro-Doppler Spectrum")
        style_spectrum_plot(self.md_plot)
        self.md_img = pg.ImageItem()
        self.md_plot.addItem(self.md_img)
        self.md_plot.setLabel('left', 'Time')
        self.md_plot.setLabel('bottom', 'Frequency')
        self.md_img.setLookupTable(spectrum_cmap.getLookupTable())
        self.md_colorbar = pg.ColorBarItem(values=(-30, 30), colorMap=spectrum_cmap, interactive=False)
        self.md_colorbar.setImageItem(self.md_img, insert_in=self.md_plot.plotItem)
        plot_layout.addWidget(self.md_plot)

        # Control Panel
        control_panel = QtWidgets.QWidget()
        control_panel.setFixedWidth(self._control_panel_width)
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        control_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        main_layout.addWidget(control_panel, stretch=1)

        self._settings = ViewerSettings("plot_bi_sensing_fast")

        connection_row = QtWidgets.QHBoxLayout()
        connection_row.addWidget(QtWidgets.QLabel("Backend IP:"))
        self.txt_backend_host = QtWidgets.QLineEdit(HOST or "127.0.0.1")
        self.txt_backend_host.setPlaceholderText("127.0.0.1")
        self.txt_backend_host.returnPressed.connect(self.connect_backend)
        self.btn_connect_backend = QtWidgets.QPushButton("Connect")
        self.btn_connect_backend.clicked.connect(self.connect_backend)
        connection_row.addWidget(self.txt_backend_host)
        connection_row.addWidget(self.btn_connect_backend)
        control_layout.addLayout(connection_row)
        control_layout.addSpacing(12)

        # Status Labels
        self.lbl_fps = QtWidgets.QLabel("FPS: 0.0")
        self.lbl_queue = QtWidgets.QLabel("Queue: 0")
        self.lbl_sender = QtWidgets.QLabel("Sender: Detecting...")
        self.lbl_params = QtWidgets.QLabel("Params: waiting...")
        self.lbl_buffer = QtWidgets.QLabel("MD Buffer: 0/5000")
        self.lbl_cfar = QtWidgets.QLabel("Detector: off")
        for lbl in [self.lbl_fps, self.lbl_queue, self.lbl_sender, self.lbl_params, self.lbl_buffer, self.lbl_cfar]:
            lbl.setWordWrap(False)
            lbl.setTextFormat(QtCore.Qt.TextFormat.PlainText)
            lbl.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
            control_layout.addWidget(lbl)

        fixed_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        for lbl in [self.lbl_fps, self.lbl_queue, self.lbl_sender, self.lbl_params, self.lbl_buffer, self.lbl_cfar]:
            lbl.setFont(fixed_font)
            lbl.setFixedWidth(self._status_label_text_width)
            self._set_status_label_text(lbl, lbl.text())
        self.targets_text.setFont(fixed_font)
        self.targets_text.setPlainText("Top Targets: waiting detector hits")
        control_layout.addSpacing(20)

        # ── Collapsible Section: Display & View ──
        self.section_display = CollapsibleSection("Display & View", collapsed=False)
        disp = self.section_display.content_layout()

        # Range Bin
        rb_layout = QtWidgets.QHBoxLayout()
        rb_layout.addWidget(QtWidgets.QLabel("Range Bin:"))
        self.txt_range_bin = QtWidgets.QLineEdit(str(selected_range_bin))
        btn_set_rb = QtWidgets.QPushButton("Set")
        btn_set_rb.clicked.connect(self.set_range_bin)
        rb_layout.addWidget(self.txt_range_bin)
        rb_layout.addWidget(btn_set_rb)
        disp.addLayout(rb_layout)

        fft_layout = QtWidgets.QHBoxLayout()
        fft_layout.addWidget(QtWidgets.QLabel("Delay FFT:"))
        self.txt_delay_fft = QtWidgets.QLineEdit(str(get_processing_range_fft_size()))
        btn_delay_fft = QtWidgets.QPushButton("Set")
        btn_delay_fft.clicked.connect(self.set_delay_fft_size)
        fft_layout.addWidget(self.txt_delay_fft)
        fft_layout.addWidget(btn_delay_fft)
        disp.addLayout(fft_layout)

        doppler_fft_layout = QtWidgets.QHBoxLayout()
        doppler_fft_layout.addWidget(QtWidgets.QLabel("Doppler FFT:"))
        self.txt_doppler_fft = QtWidgets.QLineEdit(str(get_processing_doppler_fft_size()))
        btn_doppler_fft = QtWidgets.QPushButton("Set")
        btn_doppler_fft.clicked.connect(self.set_doppler_fft_size)
        doppler_fft_layout.addWidget(self.txt_doppler_fft)
        doppler_fft_layout.addWidget(btn_doppler_fft)
        disp.addLayout(doppler_fft_layout)

        delay_view_layout = QtWidgets.QHBoxLayout()
        delay_view_layout.addWidget(QtWidgets.QLabel("Delay View:"))
        self.txt_delay_view = QtWidgets.QLineEdit(str(get_display_range_bin_limit()))
        btn_delay_view = QtWidgets.QPushButton("Set")
        btn_delay_view.clicked.connect(self.set_delay_view_bins)
        delay_view_layout.addWidget(self.txt_delay_view)
        delay_view_layout.addWidget(btn_delay_view)
        disp.addLayout(delay_view_layout)

        doppler_view_layout = QtWidgets.QHBoxLayout()
        doppler_view_layout.addWidget(QtWidgets.QLabel("Doppler View:"))
        self.txt_doppler_view = QtWidgets.QLineEdit(str(get_display_doppler_bin_limit()))
        btn_doppler_view = QtWidgets.QPushButton("Set")
        btn_doppler_view.clicked.connect(self.set_doppler_view_bins)
        doppler_view_layout.addWidget(self.txt_doppler_view)
        doppler_view_layout.addWidget(btn_doppler_view)
        disp.addLayout(doppler_view_layout)

        control_layout.addWidget(self.section_display)

        # ── Collapsible Section: Target Detection ──
        self.section_detection = CollapsibleSection("Target Detection", collapsed=True)
        det_sec = self.section_detection.content_layout()

        self.btn_cfar = QtWidgets.QPushButton("Local CLEAN: OFF")
        self.btn_cfar.setCheckable(True)
        self.btn_cfar.setChecked(False)
        self.btn_cfar.clicked.connect(self.toggle_cfar)
        det_sec.addWidget(self.btn_cfar)

        clean_main_layout = QtWidgets.QHBoxLayout()
        clean_main_layout.addWidget(QtWidgets.QLabel("Loop Gain:"))
        self.txt_clean_loop_gain = QtWidgets.QLineEdit(f"{clean_loop_gain:.2f}")
        clean_main_layout.addWidget(self.txt_clean_loop_gain)
        clean_main_layout.addWidget(QtWidgets.QLabel("Max Targets:"))
        self.txt_clean_max_targets = QtWidgets.QLineEdit(str(clean_max_targets))
        clean_main_layout.addWidget(self.txt_clean_max_targets)
        det_sec.addLayout(clean_main_layout)

        clean_misc_layout = QtWidgets.QHBoxLayout()
        clean_misc_layout.addWidget(QtWidgets.QLabel("Min R:"))
        self.txt_clean_min_range = QtWidgets.QLineEdit(str(cfar_min_range_bin))
        clean_misc_layout.addWidget(self.txt_clean_min_range)
        clean_misc_layout.addWidget(QtWidgets.QLabel("Min P(dB):"))
        self.txt_clean_min_power = QtWidgets.QLineEdit(f"{clean_min_power_db:.1f}")
        clean_misc_layout.addWidget(self.txt_clean_min_power)
        det_sec.addLayout(clean_misc_layout)

        clean_dc_layout = QtWidgets.QHBoxLayout()
        clean_dc_layout.addWidget(QtWidgets.QLabel("DC Excl:"))
        self.txt_clean_dc_excl = QtWidgets.QLineEdit(str(cfar_dc_exclusion_bins))
        btn_clean_apply = QtWidgets.QPushButton("Apply CLEAN")
        btn_clean_apply.clicked.connect(self.apply_clean_settings)
        clean_dc_layout.addWidget(self.txt_clean_dc_excl)
        clean_dc_layout.addWidget(btn_clean_apply)
        det_sec.addLayout(clean_dc_layout)
        control_layout.addWidget(self.section_detection)

        # ── Collapsible Section: Alignment ──
        self.section_align = CollapsibleSection("Alignment", collapsed=True)
        align_sec = self.section_align.content_layout()

        align_layout = QtWidgets.QHBoxLayout()
        align_layout.addWidget(QtWidgets.QLabel("Delay:"))
        self.txt_delay = QtWidgets.QLineEdit("0")
        btn_apply = QtWidgets.QPushButton("Apply")
        btn_apply.clicked.connect(self.apply_alignment)
        align_layout.addWidget(self.txt_delay)
        align_layout.addWidget(btn_apply)
        align_sec.addLayout(align_layout)

        quick_layout = QtWidgets.QHBoxLayout()
        for label, val in [('+10', 10), ('-10', -10), ('+1', 1), ('-1', -1)]:
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(lambda ch, v=val: send_alignment_command(v))
            quick_layout.addWidget(btn)
        align_sec.addLayout(quick_layout)
        control_layout.addWidget(self.section_align)

        # ── Collapsible Section: Hardware ──
        self.section_hw = CollapsibleSection("Hardware", collapsed=True)
        hw_sec = self.section_hw.content_layout()

        strd_layout = QtWidgets.QHBoxLayout()
        strd_layout.addWidget(QtWidgets.QLabel("STRD:"))
        self.txt_strd = QtWidgets.QLineEdit("20")
        btn_strd = QtWidgets.QPushButton("Set")
        btn_strd.clicked.connect(self.set_strd)
        strd_layout.addWidget(self.txt_strd)
        strd_layout.addWidget(btn_strd)
        hw_sec.addLayout(strd_layout)

        btn_hsys_cal = QtWidgets.QPushButton("Calibrate Hsys")
        btn_hsys_cal.clicked.connect(send_system_response_calibration_command)
        hw_sec.addWidget(btn_hsys_cal)
        control_layout.addWidget(self.section_hw)

        # ── Collapsible Section: Processing Toggles ──
        self.section_toggles = CollapsibleSection("Processing Toggles", collapsed=False)
        toggles_sec = self.section_toggles.content_layout()

        # Micro-Doppler Toggle
        self.btn_md = QtWidgets.QPushButton("Micro-Doppler: ON")
        self.btn_md.setCheckable(True)
        self.btn_md.setChecked(True)
        self.btn_md.clicked.connect(self.toggle_micro_doppler)
        toggles_sec.addWidget(self.btn_md)

        self.btn_mti = QtWidgets.QPushButton("MTI")
        self.update_toggle_style(self.btn_mti, enable_mti)
        self.btn_mti.clicked.connect(self.toggle_mti)
        toggles_sec.addWidget(self.btn_mti)

        self.btn_range_win = QtWidgets.QPushButton("Range Window")
        self.update_toggle_style(self.btn_range_win, enable_range_window)
        self.btn_range_win.clicked.connect(self.toggle_range_win)
        toggles_sec.addWidget(self.btn_range_win)

        self.btn_doppler_win = QtWidgets.QPushButton("Doppler Window")
        self.update_toggle_style(self.btn_doppler_win, enable_doppler_window)
        self.btn_doppler_win.clicked.connect(self.toggle_doppler_win)
        toggles_sec.addWidget(self.btn_doppler_win)
        control_layout.addWidget(self.section_toggles)

        # Save
        save_layout = QtWidgets.QHBoxLayout()
        for name, fn in [("Save Raw", self.save_raw), ("Save RD", self.save_rd), ("Save MD", self.save_md)]:
            btn = QtWidgets.QPushButton(name)
            btn.clicked.connect(fn)
            save_layout.addWidget(btn)
        control_layout.addLayout(save_layout)
        control_layout.addStretch()

        # ── Load saved settings, then connect persistence ──
        self._load_and_apply_settings()
        self._connect_settings_persistence()

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(1)  # Max FPS

        self.last_update_time = time.time()
        self.frame_count = 0
        self.total_dsp_time = 0.0
        self.targets_window.show()

    # ── Settings Persistence ──────────────────────────────────────────

    def _load_and_apply_settings(self):
        """Load saved viewer settings and apply them to UI widgets."""
        global show_micro_doppler, cfar_enabled
        global enable_mti, enable_range_window, enable_doppler_window
        s = self._settings

        def _restore_text(key, widget):
            value = s.get(key)
            if value is not None:
                widget.setText(str(value))

        saved_host = s.get("host")
        if saved_host:
            self.txt_backend_host.setText(str(saved_host))
        for key, widget in [
            ("range_bin", self.txt_range_bin),
            ("delay_fft", self.txt_delay_fft),
            ("doppler_fft", self.txt_doppler_fft),
            ("delay_view", self.txt_delay_view),
            ("doppler_view", self.txt_doppler_view),
            ("clean_loop_gain", self.txt_clean_loop_gain),
            ("clean_max_targets", self.txt_clean_max_targets),
            ("clean_min_range", self.txt_clean_min_range),
            ("clean_min_power", self.txt_clean_min_power),
            ("clean_dc_excl", self.txt_clean_dc_excl),
        ]:
            _restore_text(key, widget)
        saved_delay = s.get("alignment_delay")
        if saved_delay is not None:
            self.txt_delay.setText(str(saved_delay))
        saved_strd = s.get("strd")
        if saved_strd is not None:
            self.txt_strd.setText(str(saved_strd))

        show_micro_doppler = bool(s.get("show_micro_doppler", show_micro_doppler))
        cfar_enabled = bool(s.get("cfar_enabled", cfar_enabled))
        enable_mti = bool(s.get("enable_mti", enable_mti))
        enable_range_window = bool(s.get("enable_range_window", enable_range_window))
        enable_doppler_window = bool(s.get("enable_doppler_window", enable_doppler_window))
        self.btn_md.setChecked(show_micro_doppler)
        self.toggle_micro_doppler()
        self.btn_cfar.setChecked(cfar_enabled)
        self.toggle_cfar()
        self.update_toggle_style(self.btn_mti, enable_mti)
        self.update_toggle_style(self.btn_range_win, enable_range_window)
        self.update_toggle_style(self.btn_doppler_win, enable_doppler_window)

        self.set_range_bin()
        self.set_delay_fft_size()
        self.set_doppler_fft_size()
        self.set_delay_view_bins()
        self.set_doppler_view_bins()
        self.apply_clean_settings()
        # Section collapsed states
        for sec_name, sec_attr in [
            ("display", "section_display"),
            ("detection", "section_detection"),
            ("align", "section_align"),
            ("hw", "section_hw"),
            ("toggles", "section_toggles"),
        ]:
            collapsed = s.get(f"section_{sec_name}_collapsed")
            if collapsed is not None and hasattr(self, sec_attr):
                getattr(self, sec_attr).set_collapsed(bool(collapsed))

    def _connect_settings_persistence(self):
        """Wire widget changes to auto-save settings."""
        s = self._settings

        def _sv(key, widget, coerce=None):
            widget.textChanged.connect(lambda v: s.set(key, coerce(v) if coerce else v))

        _sv("host", self.txt_backend_host)
        _sv("range_bin", self.txt_range_bin)
        _sv("delay_fft", self.txt_delay_fft)
        _sv("doppler_fft", self.txt_doppler_fft)
        _sv("delay_view", self.txt_delay_view)
        _sv("doppler_view", self.txt_doppler_view)
        _sv("clean_loop_gain", self.txt_clean_loop_gain)
        _sv("clean_max_targets", self.txt_clean_max_targets)
        _sv("clean_min_range", self.txt_clean_min_range)
        _sv("clean_min_power", self.txt_clean_min_power)
        _sv("clean_dc_excl", self.txt_clean_dc_excl)
        _sv("alignment_delay", self.txt_delay)
        _sv("strd", self.txt_strd)

        self.btn_md.clicked.connect(lambda: s.set("show_micro_doppler", self.btn_md.isChecked()))
        self.btn_cfar.clicked.connect(lambda: s.set("cfar_enabled", self.btn_cfar.isChecked()))
        self.btn_mti.clicked.connect(lambda: s.set("enable_mti", enable_mti))
        self.btn_range_win.clicked.connect(lambda: s.set("enable_range_window", enable_range_window))
        self.btn_doppler_win.clicked.connect(lambda: s.set("enable_doppler_window", enable_doppler_window))

        # Persist section collapsed states on toggle
        for sec_name, sec_attr in [
            ("display", self.section_display),
            ("detection", self.section_detection),
            ("align", self.section_align),
            ("hw", self.section_hw),
            ("toggles", self.section_toggles),
        ]:
            sec_attr.header_button().clicked.connect(
                lambda _checked, n=sec_name, sec=sec_attr: s.set(
                    f"section_{n}_collapsed", sec.is_collapsed()
                )
            )

    def _set_status_label_text(self, label, text):
        elided = label.fontMetrics().elidedText(
            text,
            QtCore.Qt.TextElideMode.ElideRight,
            self._status_label_text_width
        )
        label.setText(elided)
        label.setToolTip(text)

    def _connect_current_backend(self, auto=False):
        host = self.txt_backend_host.text().strip() or "127.0.0.1"
        self.txt_backend_host.setText(host)
        if auto and host == HOST:
            self._set_status_label_text(self.lbl_sender, f"Sender: {host}:{CONTROL_PORT}")
            return

        self.btn_connect_backend.setEnabled(False)
        try:
            reconnect_backend(host)
            self._set_status_label_text(self.lbl_sender, f"Sender: {host}:{CONTROL_PORT}")
            self._set_status_label_text(self.lbl_params, "Params: waiting...")
        finally:
            self.btn_connect_backend.setEnabled(True)

    def connect_backend(self):
        self._connect_current_backend(auto=False)

    def update_toggle_style(self, btn, state):
        set_button_active(btn, state)

    def update_top_targets_text(self, latest):
        if latest is None:
            self.targets_text.setPlainText("Top Targets: N/A")
            return

        clusters = latest.get('target_clusters')
        if clusters is None:
            clusters = []
        rd_data = latest.get('rd')
        cfar_points = latest.get('cfar_points')
        if not clusters and (rd_data is None or cfar_points is None or len(cfar_points) == 0):
            self.targets_text.setPlainText("Top Targets: none")
            return

        if not clusters:
            clusters = cluster_detected_targets(cfar_points, rd_data)
        if not clusters:
            self.targets_text.setPlainText("Top Targets: none")
            return

        frame_id = latest.get('frame_id', -1)
        lines = [f"Top Targets F{frame_id}"]
        for rank, cluster in enumerate(clusters[:5], start=1):
            d_idx = int(cluster['peak_doppler_idx'])
            r_idx = int(cluster['peak_range_idx'])
            strength_db = float(cluster['peak_strength_db'])
            centroid_d = float(cluster['centroid_doppler_idx'])
            centroid_r = float(cluster['centroid_range_idx'])
            lines.append(
                f"{rank}. R={r_idx:4d} D={d_idx:+5d} "
                f"S={strength_db:6.1f}dB N={int(cluster['cluster_size']):2d} "
                f"C=({centroid_r:5.1f},{centroid_d:5.1f})"
            )
        self.targets_text.setPlainText("\n".join(lines))

    def set_range_bin(self):
        global selected_range_bin
        try:
            max_range_bin = min(get_display_range_bin_limit(), get_viewer_params().max_range_bin()) - 1
            val = max(0, min(max_range_bin, int(self.txt_range_bin.text())))
            selected_range_bin = val
            with buffer_lock:
                micro_doppler_buffer.clear()
            print(f"Range bin set to: {val}")
        except ValueError:
            print("Invalid range bin")

    def set_delay_fft_size(self):
        global PROCESS_RANGE_FFT_SIZE
        try:
            PROCESS_RANGE_FFT_SIZE = max(1, int(self.txt_delay_fft.text()))
            self.txt_delay_fft.setText(str(PROCESS_RANGE_FFT_SIZE))
            reset_processing_state()
            print(f"Delay FFT size set to: {PROCESS_RANGE_FFT_SIZE}")
        except ValueError:
            print("Invalid delay FFT size")

    def set_doppler_fft_size(self):
        global PROCESS_DOPPLER_FFT_SIZE
        try:
            PROCESS_DOPPLER_FFT_SIZE = max(1, int(self.txt_doppler_fft.text()))
            self.txt_doppler_fft.setText(str(PROCESS_DOPPLER_FFT_SIZE))
            reset_processing_state()
            print(f"Doppler FFT size set to: {PROCESS_DOPPLER_FFT_SIZE}")
        except ValueError:
            print("Invalid doppler FFT size")

    def set_delay_view_bins(self):
        global DISPLAY_RANGE_BIN_LIMIT, selected_range_bin
        try:
            DISPLAY_RANGE_BIN_LIMIT = max(1, int(self.txt_delay_view.text()))
            self.txt_delay_view.setText(str(DISPLAY_RANGE_BIN_LIMIT))
            selected_range_bin = min(selected_range_bin, DISPLAY_RANGE_BIN_LIMIT - 1)
            self.txt_range_bin.setText(str(selected_range_bin))
            reset_processing_state()
            print(f"Delay display bins set to: {DISPLAY_RANGE_BIN_LIMIT}")
        except ValueError:
            print("Invalid delay display bins")

    def set_doppler_view_bins(self):
        global DISPLAY_DOPPLER_BIN_LIMIT
        try:
            DISPLAY_DOPPLER_BIN_LIMIT = max(1, int(self.txt_doppler_view.text()))
            self.txt_doppler_view.setText(str(DISPLAY_DOPPLER_BIN_LIMIT))
            reset_processing_state()
            print(f"Doppler display bins set to: {DISPLAY_DOPPLER_BIN_LIMIT}")
        except ValueError:
            print("Invalid doppler display bins")

    def toggle_micro_doppler(self):
        global show_micro_doppler
        show_micro_doppler = self.btn_md.isChecked()
        self.btn_md.setText(f"Micro-Doppler: {'ON' if show_micro_doppler else 'OFF'}")

    def toggle_cfar(self):
        global cfar_enabled
        cfar_enabled = self.btn_cfar.isChecked()
        self.btn_cfar.setText(f"Local CLEAN: {'ON' if cfar_enabled else 'OFF'}")
        set_button_active(self.btn_cfar, cfar_enabled)
        if not cfar_enabled:
            current_display_data['cfar_points'] = np.empty((0, 2), dtype=np.int32)
            self.rd_cfar_marker.setData([], [])
            self.update_top_targets_text(current_display_data)

    def apply_clean_settings(self):
        global clean_loop_gain, clean_max_targets, clean_min_power_db
        global cfar_min_range_bin, cfar_dc_exclusion_bins
        try:
            clean_loop_gain = float(np.clip(float(self.txt_clean_loop_gain.text()), 1e-3, 1.0))
            clean_max_targets = max(1, int(self.txt_clean_max_targets.text()))
            cfar_min_range_bin = max(0, int(self.txt_clean_min_range.text()))
            cfar_dc_exclusion_bins = max(0, int(self.txt_clean_dc_excl.text()))
            clean_min_power_db = float(self.txt_clean_min_power.text())
            self.txt_clean_loop_gain.setText(f"{clean_loop_gain:.2f}")
            self.txt_clean_max_targets.setText(str(clean_max_targets))
            self.txt_clean_min_range.setText(str(cfar_min_range_bin))
            self.txt_clean_dc_excl.setText(str(cfar_dc_exclusion_bins))
            self.txt_clean_min_power.setText(f"{clean_min_power_db:.1f}")
            reset_processing_state()
            print(
                "Local CLEAN updated: "
                f"loop_gain={clean_loop_gain:.2f} "
                f"max_targets={clean_max_targets} "
                f"min_range={cfar_min_range_bin} "
                f"min_power_db={clean_min_power_db:.1f} "
                f"dc_excl={cfar_dc_exclusion_bins}"
            )
        except ValueError:
            print("Invalid local CLEAN settings")

    def apply_alignment(self):
        try:
            send_alignment_command(int(self.txt_delay.text()))
        except ValueError:
            print("Invalid alignment value")

    def set_strd(self):
        send_strd_command(self.txt_strd.text())

    def toggle_mti(self):
        global enable_mti
        enable_mti = not enable_mti
        self.update_toggle_style(self.btn_mti, enable_mti)
        send_mti_command(enable_mti)

    def toggle_range_win(self):
        global enable_range_window
        enable_range_window = not enable_range_window
        self.update_toggle_style(self.btn_range_win, enable_range_window)
        reset_processing_state()

    def toggle_doppler_win(self):
        global enable_doppler_window
        enable_doppler_window = not enable_doppler_window
        self.update_toggle_style(self.btn_doppler_win, enable_doppler_window)
        reset_processing_state()

    def save_raw(self):
        if 'raw' in current_display_data:
            self._save_mat('raw_frame', current_display_data['raw'], "raw")

    def save_rd(self):
        if 'rd' in current_display_data:
            self._save_mat('rd_map', current_display_data['rd'], "rd")

    def save_md(self):
        if 'md' in current_display_data:
            self._save_mat('md_map', current_display_data['md'], "md")

    def _save_mat(self, key, data, suffix):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"./capture/capture_{suffix}_{ts}.mat"
        try:
            os.makedirs("./capture", exist_ok=True)
            sio.savemat(fname, {key: data})
            print(f"Saved {suffix} to {fname}")
        except Exception as e:
            print(f"Error saving {suffix}: {e}")

    def update_plots(self):
        global current_display_data, sender_ip, last_param_error
        while True:
            try:
                startup_sync_queue.get_nowait()
            except Exception:
                break
            strd_value = self.txt_strd.text()
            mti_enabled = bool(enable_mti)
            threading.Thread(
                target=send_startup_backend_sync,
                args=(strd_value, mti_enabled),
                daemon=True,
            ).start()

        with sender_lock:
            if sender_ip:
                self._set_status_label_text(self.lbl_sender, f"Sender: {sender_ip}")
            else:
                self._set_status_label_text(self.lbl_sender, "Sender: Detecting...")

        params_text = (
            f"Params: {get_viewer_params().describe()} | "
            f"local_fft={get_processing_range_fft_size()}x{get_processing_doppler_fft_size()} | "
            f"view={get_display_range_bin_limit()}x{get_display_doppler_bin_limit()}"
        )
        if last_param_error:
            params_text += f" | last_error={last_param_error}"
        self._set_status_label_text(self.lbl_params, params_text)

        latest = None
        while not display_queue.empty():
            latest = display_queue.get()
        
        if latest:
            current_display_data = latest
            rd_data = latest['rd']
            self.rd_img.setImage(rd_data, autoLevels=False)
            cfar_points = latest.get('cfar_points')
            if cfar_points is not None and len(cfar_points) > 0:
                self.rd_cfar_marker.setData(
                    cfar_points[:, 0].astype(np.float32),
                    cfar_points[:, 1].astype(np.float32),
                )
            else:
                self.rd_cfar_marker.setData([], [])

            if latest['md'] is not None:
                self.md_img.setImage(latest['md'], autoLevels=False)

            if 'dsp_time' in latest:
                self.total_dsp_time += latest['dsp_time']
            self.frame_count += 1
            self._set_status_label_text(self.lbl_buffer, f"MD Buffer: {len(micro_doppler_buffer)}/{BUFFER_LENGTH}")
            cfar_status = f"Detector: CLEAN {'on' if cfar_enabled else 'off'}"
            if cfar_enabled:
                cfar_stats = latest.get('cfar_stats') or {}
                cfar_status += (
                    f" raw={int(latest.get('cfar_hits', 0))}"
                    f" shown={int(latest.get('cfar_shown_hits', 0))}"
                    f" pmin={cfar_stats.get('power_min_db', clean_min_power_db):.1f}dB"
                    f" gain={cfar_stats.get('loop_gain', clean_loop_gain):.2f}"
                    f" psf={int(cfar_stats.get('psf_rows', 0))}x{int(cfar_stats.get('psf_cols', 0))}"
                    f" stop={cfar_stats.get('stop_reason', 'n/a')}"
                    f" resid={cfar_stats.get('residual_peak_db', 0.0):.1f}dB"
                )
            cfar_status_short = f"Detector: CLEAN {'on' if cfar_enabled else 'off'}"
            if cfar_enabled:
                cfar_stats = latest.get('cfar_stats') or {}
                cfar_status_short += (
                    f" raw={int(latest.get('cfar_hits', 0))}"
                    f" sh={int(latest.get('cfar_shown_hits', 0))}"
                    f" p={cfar_stats.get('power_min_db', clean_min_power_db):.0f}"
                    f" g={cfar_stats.get('loop_gain', clean_loop_gain):.2f}"
                )
            self._set_status_label_text(self.lbl_cfar, cfar_status_short)
            self.lbl_cfar.setToolTip(cfar_status)

        self.update_top_targets_text(current_display_data if current_display_data else latest)

        current_time = time.time()
        elapsed = current_time - self.last_update_time
        if elapsed > 0.5:
            fps = self.frame_count / elapsed
            avg_dsp = (self.total_dsp_time / self.frame_count * 1000) if self.frame_count > 0 else 0
            self._set_status_label_text(self.lbl_fps, f"FPS: {fps:.1f} | DSP: {avg_dsp:.1f}ms")
            self._set_status_label_text(self.lbl_queue, f"RawQ: {raw_frame_queue.qsize()} | DispQ: {display_queue.qsize()}")
            self.frame_count = 0
            self.total_dsp_time = 0
            self.last_update_time = current_time

    def closeEvent(self, event):
        self._settings.save_now()
        self.targets_window.close()
        print("Closing application...")
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    apply_viewer_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
