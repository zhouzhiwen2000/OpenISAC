import argparse
import numpy as np
import sys
import os
import platform
import subprocess
import time
import threading
from queue import Queue, Empty
from collections import deque
import socket
import struct
import datetime
import scipy.io as sio

# PyQt6 + PyQtGraph Imports
from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

from sensing_runtime_protocol import (
    CTRL_HEADER,
    PARAMS_COMMAND,
    READY_COMMAND,
    ViewerRuntimeParams,
    build_params_request,
    decode_sensing_payload,
    parse_params_packet,
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
    if hasattr(arr, 'asnumpy'):
        return arr.asnumpy()
    if hasattr(arr, '__array__'):
        return np.asarray(arr)
    return arr


try:
    from scipy.signal import stft
except Exception:
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
DEFAULT_CONTROL_PORT = 9999
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

enable_mti = True
enable_range_window = True
enable_doppler_window = True

RAW_QUEUE_SIZE = 20
DISPLAY_QUEUE_SIZE = 5
DISPLAY_DOWNSAMPLE = 1
BUFFER_LENGTH = 5000
C_LIGHT_MPS = 299792458.0
ANTENNA_SPACING_M = 42.83e-3

selected_range_bin = 0
show_micro_doppler = True
display_channel = 0

PHASE_SYNC_CACHE_SIZE = 128

running = True


def get_processing_range_fft_size():
    return max(1, int(PROCESS_RANGE_FFT_SIZE))


def get_processing_doppler_fft_size():
    return max(1, int(PROCESS_DOPPLER_FFT_SIZE))


def get_display_range_bin_limit():
    return max(1, int(DISPLAY_RANGE_BIN_LIMIT))


def get_display_doppler_bin_limit():
    return max(1, int(DISPLAY_DOPPLER_BIN_LIMIT))


def reset_processing_state():
    for ch in CHANNELS:
        with ch.range_time_lock:
            ch.range_time_data = None
        with ch.micro_lock:
            ch.micro_doppler_buffer.clear()
        while True:
            try:
                ch.display_queue.get_nowait()
            except Empty:
                break
            except Exception:
                break
        ch.current_display_data = {}
    for cache in phase_sync_cache:
        cache.clear()
    for order in phase_sync_order:
        order.clear()


def parse_args():
    parser = argparse.ArgumentParser(description="Monostatic sensing viewer (fast, multi-channel)")
    parser.add_argument(
        "--ports",
        type=str,
        default="8888,8889",
        help="Comma-separated UDP ports for sensing channels, e.g. 8888,8889",
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=DEFAULT_CONTROL_PORT,
        help="Fallback control port before heartbeat detection",
    )
    return parser.parse_args()


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

    def assemble_frame(self, viewer_params):
        byte_data = b"".join(self.buffer[:self.total_chunks])
        decoded = decode_sensing_payload(self.frame_id, byte_data, viewer_params)
        return decoded.frame_id, decoded.matrix


class ChannelRuntime:
    def __init__(self, ch_id, udp_port, control_port):
        self.ch_id = ch_id
        self.udp_port = udp_port
        self.control_port = control_port

        self.frame_buffer = FrameBuffer()
        self.buffer_lock = threading.Lock()

        self.raw_frame_queue = Queue(maxsize=RAW_QUEUE_SIZE)
        self.display_queue = Queue(maxsize=DISPLAY_QUEUE_SIZE)

        self.sender_ip = None
        self.sender_lock = threading.Lock()

        self.range_time_data = None
        self.range_time_lock = threading.Lock()

        self.micro_doppler_buffer = deque(maxlen=BUFFER_LENGTH)
        self.micro_lock = threading.Lock()

        self.current_display_data = {}
        self.last_frame_id = None
        self.viewer_params = LEGACY_VIEWER_PARAMS
        self.viewer_params_lock = threading.Lock()
        self.last_param_summary = self.viewer_params.describe()
        self.last_param_error = None


args = parse_args()
UDP_PORTS = [int(x.strip()) for x in args.ports.split(",") if x.strip()]
if not UDP_PORTS:
    UDP_PORTS = [8888, 8889]
CHANNELS = [ChannelRuntime(idx, port, args.control_port) for idx, port in enumerate(UDP_PORTS)]
if not CHANNELS:
    CHANNELS = [ChannelRuntime(0, 8888, args.control_port)]

phase_sync_cache = [dict() for _ in CHANNELS]  # ch_id => frame_id -> rd_complex
phase_sync_order = [deque(maxlen=PHASE_SYNC_CACHE_SIZE) for _ in CHANNELS]


def get_viewer_params(ch):
    with ch.viewer_params_lock:
        return ch.viewer_params


def set_viewer_params(ch, viewer_params):
    with ch.viewer_params_lock:
        ch.viewer_params = viewer_params
        ch.last_param_summary = viewer_params.describe()
        ch.last_param_error = None


# ====== Network + Control ======
def _get_target_channels(target_ch_id=None):
    if target_ch_id is None:
        return CHANNELS
    if target_ch_id < 0 or target_ch_id >= len(CHANNELS):
        return []
    return [CHANNELS[target_ch_id]]


def _current_control_channel_id():
    if not CHANNELS:
        return None
    return max(0, min(display_channel, len(CHANNELS) - 1))


def send_control_command(cmd_id, value, target_ch_id=None):
    targets = _get_target_channels(target_ch_id)
    if not targets:
        print("Error: Invalid target channel")
        return

    sent_any = False
    for ch in targets:
        with ch.sender_lock:
            ip = ch.sender_ip
            cport = ch.control_port
        if ip is None:
            print(f"Error: CH{ch.ch_id + 1} sender not detected; skip {cmd_id.decode(errors='ignore').strip()}")
            continue

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            command_data = struct.pack("!4s4si", b"CMD ", cmd_id, int(value))
            sock.sendto(command_data, (ip, cport))
            sock.close()
            print(f"Sent {cmd_id.decode(errors='ignore').strip()}={value} to CH{ch.ch_id + 1} {ip}:{cport}")
            sent_any = True
        except Exception as e:
            print(f"Failed to send command to CH{ch.ch_id + 1}: {e}")

    if not sent_any:
        print("Error: No valid sender for command delivery")


def request_viewer_params(target_ch_id=None):
    targets = _get_target_channels(target_ch_id)
    if not targets:
        return

    request_packet = build_params_request(0)
    for ch in targets:
        with ch.sender_lock:
            ip = ch.sender_ip
            cport = ch.control_port
        if ip is None:
            continue
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(request_packet, (ip, cport))
            sock.close()
        except Exception as e:
            print(f"Failed to request viewer params for CH{ch.ch_id + 1}: {e}")


def send_skip_command():
    ch_id = _current_control_channel_id()
    if ch_id is None:
        return

    for _ in range(100):
        ch = CHANNELS[ch_id]
        with ch.sender_lock:
            if ch.sender_ip is not None:
                break
        time.sleep(0.1)

    request_viewer_params(ch_id)
    send_control_command(b"SKIP", 1, ch_id)
    time.sleep(0.05)
    request_viewer_params(ch_id)


def udp_receiver(ch):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_BUFFER_SIZE)
    except Exception:
        pass

    try:
        sock.bind((UDP_IP, ch.udp_port))
        print(f"[CH{ch.ch_id + 1}] Listening on UDP port {ch.udp_port}")
    except Exception as e:
        print(f"[CH{ch.ch_id + 1}] Socket bind error: {e}")
        return

    while running:
        try:
            data, addr = sock.recvfrom(MAX_CHUNK_SIZE + HEADER_SIZE)

            with ch.sender_lock:
                if ch.sender_ip is None:
                    ch.sender_ip = addr[0]
                    print(f"[CH{ch.ch_id + 1}] Detected sender IP: {ch.sender_ip}")

            if len(data) >= 8 and data[:4] == CTRL_HEADER:
                command = data[4:8]
                with ch.sender_lock:
                    if ch.control_port != addr[1]:
                        ch.control_port = addr[1]
                        print(f"[CH{ch.ch_id + 1}] Control port updated: {ch.sender_ip}:{ch.control_port}")
                if command == PARAMS_COMMAND:
                    params = parse_params_packet(data)
                    if params is not None:
                        set_viewer_params(ch, params)
                        print(f"[CH{ch.ch_id + 1}] Viewer params: {params.describe()}")
                elif command == READY_COMMAND:
                    if get_viewer_params(ch).version == 0 or ch.last_param_error is not None:
                        request_viewer_params(ch.ch_id)
                continue

            if len(data) < HEADER_SIZE:
                continue

            try:
                frame_id, total_chunks, chunk_id = struct.unpack("!III", data[:HEADER_SIZE])
            except struct.error:
                continue

            chunk_data = data[HEADER_SIZE:]
            with ch.buffer_lock:
                if ch.frame_buffer.frame_id != frame_id:
                    ch.frame_buffer.init(frame_id, total_chunks)
                if ch.frame_buffer.add_chunk(chunk_id, chunk_data):
                    viewer_params = get_viewer_params(ch)
                    frame_item = ch.frame_buffer.assemble_frame(viewer_params)
                    if ch.raw_frame_queue.full():
                        try:
                            ch.raw_frame_queue.get_nowait()
                        except Exception:
                            pass
                    ch.raw_frame_queue.put(frame_item)

        except socket.error as e:
            if getattr(e, "errno", None) != 10054:
                print(f"[CH{ch.ch_id + 1}] Socket error: {e}")
        except Exception as e:
            print(f"[CH{ch.ch_id + 1}] Receiver error: {e}")
            ch.last_param_error = str(e)
            request_viewer_params(ch.ch_id)

    sock.close()


receiver_threads = []
for _ch in CHANNELS:
    _t = threading.Thread(target=udp_receiver, args=(_ch,), daemon=True)
    _t.start()
    receiver_threads.append(_t)


# ====== FFT Processing ======
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


def process_range_doppler(frame_data, viewer_params, max_view_range_bins=None):
    if viewer_params.is_dense_range_doppler():
        rd_complex = np.asarray(
            frame_data[:viewer_params.wire_rows, :viewer_params.wire_cols],
            dtype=np.complex64,
        )
        rd_shifted = np.fft.fftshift(rd_complex, axes=0)
        magnitude_db = 20.0 * np.log10(np.abs(rd_shifted) + 1e-12)
        return magnitude_db.astype(np.float32, copy=False), None, rd_shifted.astype(np.complex64, copy=False)

    if not viewer_params.raw_fft_locally_supported():
        raise ValueError(f"Unsupported sensing frame format: {viewer_params.describe()}")

    raw_rows = max(1, int(viewer_params.active_rows))
    raw_cols = max(1, int(viewer_params.active_cols))
    range_fft_size = max(raw_cols, get_processing_range_fft_size())
    doppler_fft_size = max(raw_rows, get_processing_doppler_fft_size())
    raw_frame = np.asarray(frame_data[:raw_rows, :raw_cols], dtype=np.complex64)

    if max_view_range_bins is None:
        max_view_range_bins = get_display_range_bin_limit()
    max_view_range_bins = min(max_view_range_bins, range_fft_size)

    if USE_NVIDIA_GPU or USE_INTEL_GPU:
        range_win = get_range_window(raw_cols) if enable_range_window else cp.ones((1, raw_cols), dtype=cp.float32)
        doppler_win = get_doppler_window(raw_rows) if enable_doppler_window else cp.ones((raw_rows, 1), dtype=cp.float32)
        frame_data_gpu = cp.array(raw_frame)
        shifted_data = cp.fft.fftshift(frame_data_gpu, axes=1)
        windowed_data = shifted_data * range_win

        padded_data = cp.zeros((raw_rows, range_fft_size), dtype=cp.complex64)
        padded_data[:, :raw_cols] = windowed_data
        range_time = cp.fft.ifft(padded_data, axis=1) * range_fft_size

        range_time_view = range_time[:, :max_view_range_bins] if max_view_range_bins < range_fft_size else range_time
        range_time_cpu = to_numpy(range_time_view)

        range_time_gpu = cp.array(range_time_cpu, dtype=cp.complex64)
        doppler_windowed = range_time_gpu * doppler_win
        view_width = range_time_view.shape[1]

        padded_doppler = cp.zeros((doppler_fft_size, view_width), dtype=cp.complex64)
        padded_doppler[:raw_rows, :] = doppler_windowed
        doppler_fft = cp.fft.fft(padded_doppler, axis=0)
        doppler_shifted = cp.fft.fftshift(doppler_fft, axes=0)

        magnitude = cp.abs(doppler_shifted) / np.sqrt(raw_rows * raw_cols)
        magnitude_db = 20.0 * cp.log10(magnitude + 1e-12)
        return to_numpy(magnitude_db), range_time_cpu, to_numpy(doppler_shifted)

    if USE_APPLE_GPU:
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
            np.array(range_time_view, copy=False),
            np.array(doppler_shifted, copy=False).astype(np.complex64, copy=False),
        )

    range_win = get_range_window(raw_cols) if enable_range_window else np.ones((1, raw_cols), dtype=np.float32)
    doppler_win = get_doppler_window(raw_rows) if enable_doppler_window else np.ones((raw_rows, 1), dtype=np.float32)

    if HAVE_NUMBA:
        padded_data = cpu_prep_range_fft(raw_frame, range_win, raw_cols, range_fft_size)
        range_time = np.fft.ifft(padded_data, axis=1) * range_fft_size
        range_time_view = range_time[:, :max_view_range_bins] if max_view_range_bins < range_fft_size else range_time
        view_width = range_time_view.shape[1]

        padded_doppler = cpu_prep_doppler_fft(range_time_view, doppler_win, raw_rows, doppler_fft_size, view_width)
        doppler_fft = np.fft.fft(padded_doppler, axis=0)
        doppler_shifted = np.fft.fftshift(doppler_fft, axes=0)
        magnitude_db = 20.0 * np.log10(np.abs(doppler_shifted) / np.sqrt(raw_rows * raw_cols) + 1e-12)
        return magnitude_db.astype(np.float32, copy=False), range_time_view, doppler_shifted.astype(np.complex64, copy=False)

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

    magnitude_db = 20.0 * np.log10(np.abs(doppler_shifted) / np.sqrt(raw_rows * raw_cols) + 1e-12)
    return magnitude_db.astype(np.float32, copy=False), range_time_view, doppler_shifted.astype(np.complex64, copy=False)


def accumulate_range_time_data(ch):
    with ch.range_time_lock:
        data = ch.range_time_data
    if data is not None and data.size > 0:
        range_idx = min(selected_range_bin, data.shape[1] - 1)
        with ch.micro_lock:
            ch.micro_doppler_buffer.extend(data[:, range_idx])


def calculate_micro_doppler(ch):
    with ch.micro_lock:
        if len(ch.micro_doppler_buffer) < 256:
            return None, None, None
        complex_signal = np.array(ch.micro_doppler_buffer)

    f, t, Zxx = stft(
        complex_signal,
        fs=1.0,
        window='hamming',
        nperseg=256,
        noverlap=192,
        nfft=256,
        return_onesided=False,
    )
    Pxx_db = 20 * np.log10(np.abs(Zxx) + 1e-12)
    Pxx_db_shifted = np.fft.fftshift(Pxx_db, axes=0)
    f_shifted = np.fft.fftshift(f)
    f_idx = (f_shifted > -0.5) & (f_shifted < 0.5)
    return f_shifted[f_idx], t, Pxx_db_shifted[f_idx, :]


def _process_one_raw_item(ch, raw_item):
    if isinstance(raw_item, tuple) and len(raw_item) == 2:
        frame_id, raw_frame = raw_item
    else:
        frame_id, raw_frame = -1, raw_item

    t_start = time.time()
    viewer_params = get_viewer_params(ch)

    rd_spectrum, range_time_view, rd_complex = process_range_doppler(raw_frame, viewer_params)
    with ch.range_time_lock:
        ch.range_time_data = range_time_view

    display_range_bins = get_display_range_bin_limit()
    display_doppler_bins = get_display_doppler_bin_limit()

    if rd_spectrum.shape[1] > display_range_bins:
        rd_spectrum = rd_spectrum[:, :display_range_bins]
        rd_complex = rd_complex[:, :display_range_bins]

    center_idx = rd_spectrum.shape[0] // 2
    start_idx = max(0, center_idx - display_doppler_bins // 2)
    end_idx = min(rd_spectrum.shape[0], start_idx + display_doppler_bins)
    rd_spectrum = rd_spectrum[start_idx:end_idx, :]
    rd_complex = rd_complex[start_idx:end_idx, :]

    # Keep matrix orientation consistent with DSP output (no transpose)
    rd_spectrum_plot = rd_spectrum
    rd_complex_plot = rd_complex.astype(np.complex64, copy=False)

    if DISPLAY_DOWNSAMPLE > 1:
        rd_spectrum_plot = rd_spectrum_plot[::DISPLAY_DOWNSAMPLE, ::DISPLAY_DOWNSAMPLE]
        rd_complex_plot = rd_complex_plot[::DISPLAY_DOWNSAMPLE, ::DISPLAY_DOWNSAMPLE]

    md_spectrum = None
    md_extent = None
    if show_micro_doppler:
        accumulate_range_time_data(ch)
        f, t, Pxx = calculate_micro_doppler(ch)
        if Pxx is not None:
            # Display orientation: Y=Doppler, X=Time
            md_spectrum = Pxx
            x0 = float(t[0]) if len(t) > 0 else 0.0
            x1 = float(t[-1]) if len(t) > 1 else (x0 + 1.0)
            y0 = float(f[0]) if len(f) > 0 else -0.5
            y1 = float(f[-1]) if len(f) > 1 else (y0 + 1.0)
            md_extent = [x0, x1, y0, y1]

    dsp_time = time.time() - t_start

    result = {
        'ch_id': ch.ch_id,
        'frame_id': int(frame_id),
        'raw': raw_frame,
        'rd': rd_spectrum_plot,
        'rd_complex': rd_complex_plot,
        'md': md_spectrum,
        'md_extent': md_extent,
        'dsp_time': dsp_time,
        'viewer_params': viewer_params,
    }

    if ch.display_queue.full():
        try:
            ch.display_queue.get_nowait()
        except Exception:
            pass
    ch.display_queue.put(result)


def dsp_worker():
    print("DSP Worker started (multi-channel round-robin)")
    rr_idx = 0
    n_ch = len(CHANNELS)

    while running:
        did_work = False

        for step in range(n_ch):
            ch = CHANNELS[(rr_idx + step) % n_ch]
            latest_item = None
            while True:
                try:
                    latest_item = ch.raw_frame_queue.get_nowait()
                except Empty:
                    break
                except Exception:
                    break

            if latest_item is None:
                continue

            try:
                _process_one_raw_item(ch, latest_item)
                did_work = True
            except Exception as e:
                print(f"[CH{ch.ch_id + 1}] DSP Worker Error: {e}")
                import traceback
                traceback.print_exc()

        if n_ch > 0:
            rr_idx = (rr_idx + 1) % n_ch

        if not did_work:
            time.sleep(0.001)


dsp_thread = threading.Thread(target=dsp_worker, daemon=True)
dsp_thread.start()


# ====== Channel phase synchronization ======
def cache_phase_frame(ch_id, frame_id, rd_complex):
    if ch_id < 0 or ch_id >= len(CHANNELS) or frame_id is None or frame_id < 0 or rd_complex is None:
        return
    cache = phase_sync_cache[ch_id]
    order = phase_sync_order[ch_id]
    if frame_id in cache:
        cache[frame_id] = rd_complex
        return
    if len(order) == order.maxlen:
        old_id = order.popleft()
        cache.pop(old_id, None)
    order.append(frame_id)
    cache[frame_id] = rd_complex


def trim_phase_cache(min_keep):
    for ch_id in range(len(CHANNELS)):
        order = phase_sync_order[ch_id]
        cache = phase_sync_cache[ch_id]
        while order and order[0] < min_keep:
            old_id = order.popleft()
            cache.pop(old_id, None)


# ====== Command Functions ======
def send_alignment_command(val):
    target = _current_control_channel_id()
    if target is None:
        return
    # ALGN in backend is applied to the currently selected ALCH target.
    send_control_command(b"ALCH", int(target), target)
    send_control_command(b"ALGN", int(val), target)


def send_strd_command(val):
    target = _current_control_channel_id()
    if target is None:
        return
    try:
        strd_limit = get_viewer_params(CHANNELS[target]).max_strd_value()
        strd_val = max(1, min(strd_limit, int(val)))
        send_control_command(b"STRD", strd_val, target)
        time.sleep(0.02)
        request_viewer_params(target)
    except ValueError:
        print(f"Invalid STRD value: {val}")


def send_mti_command(enabled):
    target = _current_control_channel_id()
    if target is None:
        return
    send_control_command(b"MTI ", 1 if enabled else 0, target)


def send_tx_gain_command(val):
    target = _current_control_channel_id()
    if target is None:
        return
    try:
        gain_db = float(val)
        gain_x10 = int(round(gain_db * 10.0))
        send_control_command(b"TXGN", gain_x10, target)
        print(f"Requested TX gain: {gain_db:.1f} dB (CH{target + 1})")
    except ValueError:
        print(f"Invalid TX gain value: {val}")


def send_rx_gain_command(val):
    target = _current_control_channel_id()
    if target is None:
        return
    try:
        gain_db = float(val)
        gain_x10 = int(round(gain_db * 10.0))
        # Select target sensing channel in modulator, then apply RX gain.
        send_control_command(b"ALCH", int(target), target)
        send_control_command(b"RXGN", gain_x10, target)
        print(f"Requested RX gain: {gain_db:.1f} dB (CH{target + 1})")
    except ValueError:
        print(f"Invalid RX gain value: {val}")


# ====== MainWindow with PyQt6 + PyQtGraph ======
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenISAC Sensing - PyQtGraph (Multi-Channel)")
        self.resize(1600, 900)
        self._control_panel_width = 520
        self._status_label_text_width = self._control_panel_width - 24

        self.phase_ready = False
        self.synced_frame_id = None
        self.synced_rd_complex = None
        self.phase_curve_raw = None
        self.phase_curve_comp = None
        self.phase_bias_per_channel = np.zeros(len(CHANNELS), dtype=np.float64)
        self.phase_calibrated = False
        self.center_freq_hz = 2.4e9
        self.clicked_range_idx = None
        self.clicked_doppler_idx = None
        self.rd_doppler_min = 0.0
        self.rd_doppler_span = 1.0

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # Plot Area
        plot_widget = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_widget)
        main_layout.addWidget(plot_widget, stretch=4)

        # Range-Doppler Plot
        self.rd_plot = pg.PlotWidget(title="Range-Doppler Spectrum")
        # RD uses col-major mapping: x <- first index(doppler), y <- second index(range)
        self.rd_img = pg.ImageItem(axisOrder='col-major')
        self.rd_plot.addItem(self.rd_img)
        self.rd_plot.setLabel('left', 'Range Bin')
        self.rd_plot.setLabel('bottom', 'Doppler Bin')
        self.rd_img.setLookupTable(pg.colormap.get('turbo').getLookupTable())
        self.rd_colorbar = pg.ColorBarItem(values=(0, 60), colorMap=pg.colormap.get('turbo'), interactive=False)
        self.rd_colorbar.setImageItem(self.rd_img, insert_in=self.rd_plot.plotItem)

        self.rd_click_marker = pg.ScatterPlotItem([], [], symbol='x', size=12, pen=pg.mkPen('w', width=2))
        self.rd_plot.addItem(self.rd_click_marker)
        self.rd_plot.scene().sigMouseClicked.connect(self.on_rd_mouse_clicked)

        plot_layout.addWidget(self.rd_plot)

        # Micro-Doppler Plot
        self.md_plot = pg.PlotWidget(title="Micro-Doppler Spectrum")
        # MD uses row-major mapping: x <- second index(time), y <- first index(doppler)
        self.md_img = pg.ImageItem(axisOrder='row-major')
        self.md_plot.addItem(self.md_img)
        self.md_plot.setLabel('left', 'Doppler')
        self.md_plot.setLabel('bottom', 'Time')
        self.md_img.setLookupTable(pg.colormap.get('turbo').getLookupTable())
        self.md_colorbar = pg.ColorBarItem(values=(0, 60), colorMap=pg.colormap.get('turbo'), interactive=False)
        self.md_colorbar.setImageItem(self.md_img, insert_in=self.md_plot.plotItem)
        plot_layout.addWidget(self.md_plot)

        # Phase-Channel Curve Plot
        self.phase_curve_plot = pg.PlotWidget(title="Phase-Channel Curve @ Clicked RD Point")
        self.phase_curve_plot.setLabel('left', 'Phase (rad, unwrapped)')
        self.phase_curve_plot.setLabel('bottom', 'Channel Index')
        self.phase_curve_plot.showGrid(x=True, y=True, alpha=0.3)
        self.phase_curve_plot.addLegend(offset=(8, 8))
        self.phase_curve_raw_item = self.phase_curve_plot.plot(
            [], [], pen=pg.mkPen('#f7d154', width=2), symbol='o', symbolSize=6, name='Raw'
        )
        self.phase_curve_comp_item = self.phase_curve_plot.plot(
            [], [], pen=pg.mkPen('#4cc9f0', width=2), symbol='x', symbolSize=7, name='Calibrated'
        )
        plot_layout.addWidget(self.phase_curve_plot)

        # Control Panel
        control_panel = QtWidgets.QWidget()
        control_panel.setFixedWidth(self._control_panel_width)
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        control_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        main_layout.addWidget(control_panel, stretch=1)

        # Status Labels
        self.lbl_display = QtWidgets.QLabel("Display: CH1")
        self.lbl_fps = QtWidgets.QLabel("FPS: 0.0")
        self.lbl_queue = QtWidgets.QLabel("Queue: 0")
        self.lbl_sender = QtWidgets.QLabel("Sender: Detecting...")
        self.lbl_params = QtWidgets.QLabel("Params: waiting...")
        self.lbl_buffer = QtWidgets.QLabel("MD Buffer: 0/5000")
        self.lbl_phase_sync = QtWidgets.QLabel("Phase Sync: waiting same frame across channels")
        self.lbl_phase_clicked = QtWidgets.QLabel("Phase@Clicked: click RD map to query")
        self.lbl_aoa_status = QtWidgets.QLabel("AoA: waiting calibration/click")

        for lbl in [
            self.lbl_display,
            self.lbl_fps,
            self.lbl_queue,
            self.lbl_sender,
            self.lbl_params,
            self.lbl_buffer,
            self.lbl_phase_sync,
            self.lbl_phase_clicked,
            self.lbl_aoa_status,
        ]:
            lbl.setWordWrap(False)
            lbl.setTextFormat(QtCore.Qt.TextFormat.PlainText)
            lbl.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
            control_layout.addWidget(lbl)

        fixed_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        for lbl in [
            self.lbl_display,
            self.lbl_fps,
            self.lbl_queue,
            self.lbl_sender,
            self.lbl_params,
            self.lbl_buffer,
            self.lbl_phase_sync,
            self.lbl_phase_clicked,
            self.lbl_aoa_status,
        ]:
            lbl.setFont(fixed_font)
            lbl.setFixedWidth(self._status_label_text_width)
            self._set_status_label_text(lbl, lbl.text())

        control_layout.addSpacing(16)

        # Display channel selector
        ch_select_layout = QtWidgets.QHBoxLayout()
        ch_select_layout.addWidget(QtWidgets.QLabel("Display CH:"))
        self.combo_display_channel = QtWidgets.QComboBox()
        self.combo_display_channel.currentIndexChanged.connect(self.on_display_channel_changed)
        ch_select_layout.addWidget(self.combo_display_channel)
        control_layout.addLayout(ch_select_layout)
        self.refresh_display_button_text()

        control_layout.addSpacing(16)

        # Range Bin
        rb_layout = QtWidgets.QHBoxLayout()
        rb_layout.addWidget(QtWidgets.QLabel("Range Bin:"))
        self.txt_range_bin = QtWidgets.QLineEdit(str(selected_range_bin))
        btn_set_rb = QtWidgets.QPushButton("Set")
        btn_set_rb.clicked.connect(self.set_range_bin)
        rb_layout.addWidget(self.txt_range_bin)
        rb_layout.addWidget(btn_set_rb)
        control_layout.addLayout(rb_layout)

        fft_layout = QtWidgets.QHBoxLayout()
        fft_layout.addWidget(QtWidgets.QLabel("Delay FFT:"))
        self.txt_delay_fft = QtWidgets.QLineEdit(str(get_processing_range_fft_size()))
        btn_delay_fft = QtWidgets.QPushButton("Set")
        btn_delay_fft.clicked.connect(self.set_delay_fft_size)
        fft_layout.addWidget(self.txt_delay_fft)
        fft_layout.addWidget(btn_delay_fft)
        control_layout.addLayout(fft_layout)

        doppler_fft_layout = QtWidgets.QHBoxLayout()
        doppler_fft_layout.addWidget(QtWidgets.QLabel("Doppler FFT:"))
        self.txt_doppler_fft = QtWidgets.QLineEdit(str(get_processing_doppler_fft_size()))
        btn_doppler_fft = QtWidgets.QPushButton("Set")
        btn_doppler_fft.clicked.connect(self.set_doppler_fft_size)
        doppler_fft_layout.addWidget(self.txt_doppler_fft)
        doppler_fft_layout.addWidget(btn_doppler_fft)
        control_layout.addLayout(doppler_fft_layout)

        delay_view_layout = QtWidgets.QHBoxLayout()
        delay_view_layout.addWidget(QtWidgets.QLabel("Delay View:"))
        self.txt_delay_view = QtWidgets.QLineEdit(str(get_display_range_bin_limit()))
        btn_delay_view = QtWidgets.QPushButton("Set")
        btn_delay_view.clicked.connect(self.set_delay_view_bins)
        delay_view_layout.addWidget(self.txt_delay_view)
        delay_view_layout.addWidget(btn_delay_view)
        control_layout.addLayout(delay_view_layout)

        doppler_view_layout = QtWidgets.QHBoxLayout()
        doppler_view_layout.addWidget(QtWidgets.QLabel("Doppler View:"))
        self.txt_doppler_view = QtWidgets.QLineEdit(str(get_display_doppler_bin_limit()))
        btn_doppler_view = QtWidgets.QPushButton("Set")
        btn_doppler_view.clicked.connect(self.set_doppler_view_bins)
        doppler_view_layout.addWidget(self.txt_doppler_view)
        doppler_view_layout.addWidget(btn_doppler_view)
        control_layout.addLayout(doppler_view_layout)

        # Micro-Doppler Toggle
        self.btn_md = QtWidgets.QPushButton("Micro-Doppler: ON")
        self.btn_md.setCheckable(True)
        self.btn_md.setChecked(True)
        self.btn_md.clicked.connect(self.toggle_micro_doppler)
        self.btn_md.setStyleSheet("QPushButton:checked { background-color: lightgreen; }")
        control_layout.addWidget(self.btn_md)
        control_layout.addSpacing(20)

        # Alignment
        align_layout = QtWidgets.QHBoxLayout()
        align_layout.addWidget(QtWidgets.QLabel("Delay:"))
        self.txt_delay = QtWidgets.QLineEdit("0")
        btn_apply = QtWidgets.QPushButton("Apply")
        btn_apply.clicked.connect(self.apply_alignment)
        align_layout.addWidget(self.txt_delay)
        align_layout.addWidget(btn_apply)
        control_layout.addLayout(align_layout)

        quick_layout = QtWidgets.QHBoxLayout()
        for label, val in [('+57600', 57600), ('-57600', -57600), ('+10', 10), ('-10', -10), ('+1', 1), ('-1', -1)]:
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(lambda ch, v=val: send_alignment_command(v))
            quick_layout.addWidget(btn)
        control_layout.addLayout(quick_layout)
        control_layout.addSpacing(20)

        # STRD
        strd_layout = QtWidgets.QHBoxLayout()
        strd_layout.addWidget(QtWidgets.QLabel("STRD:"))
        self.txt_strd = QtWidgets.QLineEdit("20")
        btn_strd = QtWidgets.QPushButton("Set")
        btn_strd.clicked.connect(self.set_strd)
        strd_layout.addWidget(self.txt_strd)
        strd_layout.addWidget(btn_strd)
        control_layout.addLayout(strd_layout)
        
        tx_gain_layout = QtWidgets.QHBoxLayout()
        tx_gain_layout.addWidget(QtWidgets.QLabel("TX Gain(dB):"))
        self.txt_tx_gain = QtWidgets.QLineEdit("20.0")
        btn_tx_gain = QtWidgets.QPushButton("Set")
        btn_tx_gain.clicked.connect(self.set_tx_gain)
        tx_gain_layout.addWidget(self.txt_tx_gain)
        tx_gain_layout.addWidget(btn_tx_gain)
        control_layout.addLayout(tx_gain_layout)

        rx_gain_layout = QtWidgets.QHBoxLayout()
        rx_gain_layout.addWidget(QtWidgets.QLabel("RX Gain(dB):"))
        self.txt_rx_gain = QtWidgets.QLineEdit("30.0")
        btn_rx_gain = QtWidgets.QPushButton("Set")
        btn_rx_gain.clicked.connect(self.set_rx_gain)
        rx_gain_layout.addWidget(self.txt_rx_gain)
        rx_gain_layout.addWidget(btn_rx_gain)
        control_layout.addLayout(rx_gain_layout)

        aoa_freq_layout = QtWidgets.QHBoxLayout()
        aoa_freq_layout.addWidget(QtWidgets.QLabel("Center Freq(GHz):"))
        self.txt_center_freq_ghz = QtWidgets.QLineEdit("2.4")
        btn_center_freq = QtWidgets.QPushButton("Set")
        btn_center_freq.clicked.connect(self.set_center_freq)
        aoa_freq_layout.addWidget(self.txt_center_freq_ghz)
        aoa_freq_layout.addWidget(btn_center_freq)
        control_layout.addLayout(aoa_freq_layout)

        self.btn_calibrate = QtWidgets.QPushButton("Calibrate Phase")
        self.btn_calibrate.clicked.connect(self.calibrate_phase_bias)
        control_layout.addWidget(self.btn_calibrate)
        control_layout.addSpacing(20)

        # Toggles
        self.btn_mti = QtWidgets.QPushButton("MTI")
        self.update_toggle_style(self.btn_mti, enable_mti)
        self.btn_mti.clicked.connect(self.toggle_mti)
        control_layout.addWidget(self.btn_mti)

        self.btn_range_win = QtWidgets.QPushButton("Range Window")
        self.update_toggle_style(self.btn_range_win, enable_range_window)
        self.btn_range_win.clicked.connect(self.toggle_range_win)
        control_layout.addWidget(self.btn_range_win)

        self.btn_doppler_win = QtWidgets.QPushButton("Doppler Window")
        self.update_toggle_style(self.btn_doppler_win, enable_doppler_window)
        self.btn_doppler_win.clicked.connect(self.toggle_doppler_win)
        control_layout.addWidget(self.btn_doppler_win)
        control_layout.addSpacing(20)

        # Save
        save_layout = QtWidgets.QHBoxLayout()
        for name, fn in [("Save Raw", self.save_raw), ("Save RD", self.save_rd), ("Save MD", self.save_md)]:
            btn = QtWidgets.QPushButton(name)
            btn.clicked.connect(fn)
            save_layout.addWidget(btn)
        control_layout.addLayout(save_layout)
        control_layout.addStretch()

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        # Avoid overwhelming Qt event loop under sustained high-rate streams.
        self.timer.start(5)

        self.last_update_time = time.time()
        self.frame_count = 0
        self.total_dsp_time = 0.0

        threading.Thread(target=send_skip_command, daemon=True).start()

    def _set_status_label_text(self, label, text):
        elided = label.fontMetrics().elidedText(
            text,
            QtCore.Qt.TextElideMode.ElideRight,
            self._status_label_text_width
        )
        label.setText(elided)
        label.setToolTip(text)

    def update_toggle_style(self, btn, state):
        btn.setStyleSheet("background-color: lightgreen;" if state else "background-color: lightgray;")

    def refresh_display_button_text(self):
        global display_channel
        n_ch = len(CHANNELS)
        if n_ch <= 0:
            display_channel = 0
            self.combo_display_channel.setEnabled(False)
            self._set_status_label_text(self.lbl_display, "Display: N/A")
            self.phase_curve_plot.setVisible(False)
            if hasattr(self, "btn_calibrate"):
                self.btn_calibrate.setEnabled(False)
            return

        display_channel = max(0, min(display_channel, n_ch - 1))

        if self.combo_display_channel.count() != n_ch:
            self.combo_display_channel.blockSignals(True)
            self.combo_display_channel.clear()
            self.combo_display_channel.addItems([f"CH{i + 1}" for i in range(n_ch)])
            self.combo_display_channel.blockSignals(False)

        if self.combo_display_channel.currentIndex() != display_channel:
            self.combo_display_channel.blockSignals(True)
            self.combo_display_channel.setCurrentIndex(display_channel)
            self.combo_display_channel.blockSignals(False)

        self.combo_display_channel.setEnabled(n_ch > 1)
        self.phase_curve_plot.setVisible(n_ch >= 2)
        if hasattr(self, "btn_calibrate"):
            self.btn_calibrate.setEnabled(n_ch >= 2)
        self._set_status_label_text(self.lbl_display, f"Display: CH{display_channel + 1}")

    def _get_display_channel_runtime(self):
        if not CHANNELS:
            return None
        ch_idx = max(0, min(display_channel, len(CHANNELS) - 1))
        return CHANNELS[ch_idx]

    def on_display_channel_changed(self, idx):
        global display_channel
        if idx < 0 or idx >= len(CHANNELS):
            return
        display_channel = idx
        self.refresh_display_button_text()
        print(f"Display channel switched to CH{display_channel + 1}")

    def toggle_display_channel(self):
        global display_channel
        if len(CHANNELS) < 2:
            return
        display_channel = (display_channel + 1) % len(CHANNELS)
        self.refresh_display_button_text()
        print(f"Display channel switched to CH{display_channel + 1}")

    def set_range_bin(self):
        global selected_range_bin
        try:
            ch = self._get_display_channel_runtime()
            if ch is not None:
                max_range_bin = min(get_display_range_bin_limit(), get_viewer_params(ch).max_range_bin()) - 1
            else:
                max_range_bin = get_display_range_bin_limit() - 1
            val = max(0, min(max_range_bin, int(self.txt_range_bin.text())))
            selected_range_bin = val
            for ch in CHANNELS:
                with ch.micro_lock:
                    ch.micro_doppler_buffer.clear()
            print(f"Range bin set to: {val} (all channel MD buffers cleared)")
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

    def apply_alignment(self):
        try:
            send_alignment_command(int(self.txt_delay.text()))
        except ValueError:
            print("Invalid alignment value")

    def set_strd(self):
        send_strd_command(self.txt_strd.text())

    def set_tx_gain(self):
        send_tx_gain_command(self.txt_tx_gain.text())

    def set_rx_gain(self):
        send_rx_gain_command(self.txt_rx_gain.text())

    def set_center_freq(self):
        try:
            freq_ghz = float(self.txt_center_freq_ghz.text())
            if freq_ghz <= 0.0:
                raise ValueError
            self.center_freq_hz = freq_ghz * 1e9
            print(f"AoA center frequency set to {freq_ghz:.6f} GHz")
            self.update_phase_probe_text()
        except ValueError:
            print(f"Invalid center frequency: {self.txt_center_freq_ghz.text()}")
            self.txt_center_freq_ghz.setText(f"{self.center_freq_hz / 1e9:.6f}")

    def calibrate_phase_bias(self):
        if len(CHANNELS) < 2:
            self._set_status_label_text(self.lbl_aoa_status, "AoA: unavailable (need >=2 channels)")
            print("Calibration skipped: need >=2 channels")
            return
        phase_result = self._compute_phase_vectors()
        if phase_result is None:
            self._set_status_label_text(self.lbl_aoa_status, "AoA: calibration failed (sync/click not ready)")
            print("Calibration skipped: phase vector not ready")
            return

        _, r_idx, doppler_bin, phase_raw, _ = phase_result
        self.phase_bias_per_channel = phase_raw.astype(np.float64, copy=True)
        self.phase_bias_per_channel[0] = 0.0
        self.phase_calibrated = True
        print(
            f"Phase calibrated at frame {self.synced_frame_id}, R={r_idx}, D={doppler_bin:+.0f}, "
            f"bias(rad, CH1-ref)={np.array2string(self.phase_bias_per_channel, precision=3, separator=',')}"
        )
        self.update_phase_probe_text()

    def _sync_phase_frame(self):
        self.phase_ready = False
        self.synced_frame_id = None
        self.synced_rd_complex = None

        if len(CHANNELS) < 2:
            return

        common_frame_ids = None
        for ch_cache in phase_sync_cache:
            if not ch_cache:
                common_frame_ids = set()
                break
            ch_ids = set(ch_cache.keys())
            if common_frame_ids is None:
                common_frame_ids = ch_ids
            else:
                common_frame_ids &= ch_ids
            if not common_frame_ids:
                break

        if not common_frame_ids:
            return

        synced_frame_id = int(max(common_frame_ids))
        rd_complex_list = []
        ref_shape = None
        for ch_id in range(len(CHANNELS)):
            rd_complex = phase_sync_cache[ch_id].get(synced_frame_id)
            if rd_complex is None:
                return
            if ref_shape is None:
                ref_shape = rd_complex.shape
            elif rd_complex.shape != ref_shape:
                return
            rd_complex_list.append(rd_complex)

        self.phase_ready = True
        self.synced_frame_id = synced_frame_id
        self.synced_rd_complex = rd_complex_list
        trim_phase_cache(synced_frame_id - 8)

    def _compute_phase_vectors(self):
        if (
            not self.phase_ready
            or self.synced_rd_complex is None
            or self.clicked_range_idx is None
            or self.clicked_doppler_idx is None
        ):
            return None

        rd_ref = self.synced_rd_complex[0]
        if rd_ref is None or rd_ref.size == 0:
            return None

        rows, cols = rd_ref.shape
        d_idx = int(np.clip(self.clicked_doppler_idx, 0, rows - 1))
        r_idx = int(np.clip(self.clicked_range_idx, 0, cols - 1))
        doppler_bin = self.rd_doppler_min + d_idx

        z = np.asarray([rd[d_idx, r_idx] for rd in self.synced_rd_complex], dtype=np.complex64)
        phase_rel = np.angle(z * np.conj(z[0]))
        phase_raw = np.unwrap(phase_rel.astype(np.float64))

        if self.phase_bias_per_channel.shape[0] != len(CHANNELS):
            self.phase_bias_per_channel = np.zeros(len(CHANNELS), dtype=np.float64)
            self.phase_calibrated = False
        phase_comp = phase_raw - self.phase_bias_per_channel
        return d_idx, r_idx, doppler_bin, phase_raw, phase_comp

    def _update_phase_curve_plot(self):
        if len(CHANNELS) < 2:
            self.phase_curve_raw_item.setData([], [])
            self.phase_curve_comp_item.setData([], [])
            return
        if self.phase_curve_raw is None:
            self.phase_curve_raw_item.setData([], [])
            self.phase_curve_comp_item.setData([], [])
            return

        x = np.arange(1, len(CHANNELS) + 1, dtype=np.float64)
        self.phase_curve_raw_item.setData(x, self.phase_curve_raw)
        if self.phase_calibrated and self.phase_curve_comp is not None:
            self.phase_curve_comp_item.setData(x, self.phase_curve_comp)
        else:
            self.phase_curve_comp_item.setData([], [])
        self.phase_curve_plot.setXRange(0.5, len(CHANNELS) + 0.5, padding=0.0)

    def _estimate_aoa_from_phase(self, phase_values):
        if phase_values is None or len(phase_values) < 2 or self.center_freq_hz <= 0.0:
            return None, None
        x = np.arange(len(phase_values), dtype=np.float64)
        try:
            slope = float(np.polyfit(x, phase_values, 1)[0])
        except Exception:
            return None, None
        wavelength = C_LIGHT_MPS / self.center_freq_hz
        sin_theta = np.clip(slope * wavelength / (2.0 * np.pi * ANTENNA_SPACING_M), -1.0, 1.0)
        aoa_deg = float(np.degrees(np.arcsin(sin_theta)))
        return aoa_deg, slope

    def toggle_mti(self):
        global enable_mti
        enable_mti = not enable_mti
        self.update_toggle_style(self.btn_mti, enable_mti)
        send_mti_command(enable_mti)

    def toggle_range_win(self):
        global enable_range_window
        enable_range_window = not enable_range_window
        self.update_toggle_style(self.btn_range_win, enable_range_window)

    def toggle_doppler_win(self):
        global enable_doppler_window
        enable_doppler_window = not enable_doppler_window
        self.update_toggle_style(self.btn_doppler_win, enable_doppler_window)

    def save_raw(self):
        ch = self._get_display_channel_runtime()
        if ch and 'raw' in ch.current_display_data:
            self._save_mat('raw_frame', ch.current_display_data['raw'], "raw")

    def save_rd(self):
        ch = self._get_display_channel_runtime()
        if ch and 'rd' in ch.current_display_data:
            self._save_mat('rd_map', ch.current_display_data['rd'], "rd")

    def save_md(self):
        ch = self._get_display_channel_runtime()
        if ch and 'md' in ch.current_display_data and ch.current_display_data.get('md') is not None:
            self._save_mat('md_map', ch.current_display_data['md'], "md")

    def _save_mat(self, key, data, suffix):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"./capture/capture_{suffix}_{ts}.mat"
        try:
            os.makedirs("./capture", exist_ok=True)
            sio.savemat(fname, {key: data})
            print(f"Saved {suffix} to {fname}")
        except Exception as e:
            print(f"Error saving {suffix}: {e}")

    def on_rd_mouse_clicked(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return

        ch = self._get_display_channel_runtime()
        if ch is None:
            return

        rd_data = ch.current_display_data.get('rd')
        if rd_data is None or rd_data.size == 0:
            return

        scene_pos = event.scenePos()
        if not self.rd_plot.sceneBoundingRect().contains(scene_pos):
            return

        mouse_point = self.rd_plot.plotItem.vb.mapSceneToView(scene_pos)
        x_val = float(mouse_point.x())
        y_val = float(mouse_point.y())

        rows, cols = rd_data.shape  # rows=doppler bins, cols=range bins
        doppler_idx = int(np.clip(round(x_val - self.rd_doppler_min), 0, rows - 1))
        range_idx = int(np.clip(round(y_val), 0, cols - 1))
        doppler_bin = self.rd_doppler_min + doppler_idx

        self.clicked_range_idx = range_idx
        self.clicked_doppler_idx = doppler_idx
        self.rd_click_marker.setData([doppler_bin], [range_idx])

        print(f"Clicked RD point: R={range_idx}, D={doppler_bin:+.0f}")
        self.update_phase_probe_text()

    def update_phase_status(self):
        if len(CHANNELS) < 2:
            self.phase_ready = False
            self.synced_frame_id = None
            self.synced_rd_complex = None
            self._set_status_label_text(self.lbl_phase_sync, "Phase Sync: unavailable (need >=2 channels)")
            self._set_status_label_text(self.lbl_aoa_status, "AoA: unavailable (need >=2 channels)")
            return

        frame_tags = ", ".join([f"CH{i + 1}={CHANNELS[i].last_frame_id}" for i in range(len(CHANNELS))])
        if not self.phase_ready or self.synced_frame_id is None or self.synced_rd_complex is None:
            self._set_status_label_text(
                self.lbl_phase_sync,
                f"Phase Sync: waiting same frame across {len(CHANNELS)} channels ({frame_tags})"
            )
            if self.phase_calibrated:
                self._set_status_label_text(self.lbl_aoa_status, "AoA: waiting phase sync (calibrated)")
            else:
                self._set_status_label_text(self.lbl_aoa_status, "AoA: waiting phase sync (not calibrated)")
            return

        self._set_status_label_text(
            self.lbl_phase_sync,
            f"Phase Sync: locked frame {self.synced_frame_id} across {len(CHANNELS)} channels"
        )

    def update_phase_probe_text(self):
        if len(CHANNELS) < 2:
            self.phase_curve_raw = None
            self.phase_curve_comp = None
            self._update_phase_curve_plot()
            self._set_status_label_text(self.lbl_phase_clicked, "Phase@Clicked: unavailable (need >=2 channels)")
            self._set_status_label_text(self.lbl_aoa_status, "AoA: unavailable (need >=2 channels)")
            return

        if self.clicked_range_idx is None or self.clicked_doppler_idx is None:
            self.phase_curve_raw = None
            self.phase_curve_comp = None
            self._update_phase_curve_plot()
            self._set_status_label_text(self.lbl_phase_clicked, "Phase@Clicked: click RD map to query")
            if self.phase_calibrated:
                self._set_status_label_text(self.lbl_aoa_status, "AoA: calibrated; click RD map to query")
            else:
                self._set_status_label_text(self.lbl_aoa_status, "AoA: not calibrated; click RD map then calibrate")
            return

        phase_result = self._compute_phase_vectors()
        if phase_result is None:
            self.phase_curve_raw = None
            self.phase_curve_comp = None
            self._update_phase_curve_plot()
            self._set_status_label_text(
                self.lbl_phase_clicked,
                "Phase@Clicked: waiting same-frame data across all channels"
            )
            if self.phase_calibrated:
                self._set_status_label_text(self.lbl_aoa_status, "AoA: waiting same-frame phase (calibrated)")
            else:
                self._set_status_label_text(self.lbl_aoa_status, "AoA: waiting same-frame phase")
            return

        _, r_idx, doppler_bin, phase_raw, phase_comp = phase_result
        self.phase_curve_raw = phase_raw
        self.phase_curve_comp = phase_comp
        self._update_phase_curve_plot()

        disp_idx = max(0, min(display_channel, len(CHANNELS) - 1))
        ph_raw_disp = float(phase_raw[disp_idx])
        ph_comp_disp = float(phase_comp[disp_idx])
        aoa_deg, slope = self._estimate_aoa_from_phase(phase_comp)

        calib_tag = "calibrated" if self.phase_calibrated else "uncalibrated"
        self._set_status_label_text(
            self.lbl_phase_clicked,
            f"Phase@R={r_idx},D={doppler_bin:+.0f},F={self.synced_frame_id}: "
            f"CH{disp_idx + 1} raw={ph_raw_disp:+.3f}, comp={ph_comp_disp:+.3f} rad ({calib_tag})"
        )

        if aoa_deg is None or slope is None:
            self._set_status_label_text(self.lbl_aoa_status, "AoA: estimation unavailable")
            return
        self._set_status_label_text(
            self.lbl_aoa_status,
            f"AoA@R={r_idx},D={doppler_bin:+.0f}: {aoa_deg:+.2f} deg, slope={slope:+.4f} rad/ch, "
            f"fc={self.center_freq_hz/1e9:.3f} GHz ({calib_tag})"
        )

    def update_plots(self):
        updated_any = False

        # Pull latest display item from each channel
        for ch in CHANNELS:
            latest = None
            while not ch.display_queue.empty():
                latest = ch.display_queue.get()

            if latest is not None:
                ch.current_display_data = latest
                ch.last_frame_id = latest.get('frame_id', ch.last_frame_id)
                cache_phase_frame(ch.ch_id, latest.get('frame_id', -1), latest.get('rd_complex'))
                if 'dsp_time' in latest:
                    self.total_dsp_time += latest['dsp_time']
                updated_any = True

        self._sync_phase_frame()

        # Update phase status text only (no heavy compute in UI thread).
        self.update_phase_status()

        # Update currently displayed channel
        ch = self._get_display_channel_runtime()
        if ch is None:
            self._set_status_label_text(self.lbl_sender, "Sender: N/A")
            self._set_status_label_text(self.lbl_params, "Params: N/A")
            self._set_status_label_text(self.lbl_buffer, "MD Buffer: N/A")
            self.rd_click_marker.setData([], [])
        else:
            latest_disp = ch.current_display_data
            rd_data = latest_disp.get('rd')
            md_data = latest_disp.get('md')

            if rd_data is not None:
                self.rd_img.setImage(rd_data, autoLevels=False)
                rows, cols = rd_data.shape  # rows=doppler, cols=range
                self.rd_doppler_min = -0.5 * float(rows)
                self.rd_doppler_span = float(rows)
                self.rd_img.setRect(QtCore.QRectF(self.rd_doppler_min, 0.0, self.rd_doppler_span, float(cols)))
                if self.clicked_range_idx is not None and self.clicked_doppler_idx is not None:
                    d_idx = int(np.clip(self.clicked_doppler_idx, 0, rows - 1))
                    r_idx = int(np.clip(self.clicked_range_idx, 0, cols - 1))
                    doppler_bin = self.rd_doppler_min + d_idx
                    self.rd_click_marker.setData([doppler_bin], [r_idx])
            if md_data is not None:
                self.md_img.setImage(md_data, autoLevels=False)
                md_extent = latest_disp.get('md_extent')
                if md_extent is not None and len(md_extent) == 4:
                    x0, x1, y0, y1 = [float(v) for v in md_extent]
                    self.md_img.setRect(
                        QtCore.QRectF(x0, y0, max(1e-6, x1 - x0), max(1e-6, y1 - y0))
                    )

            with ch.sender_lock:
                if ch.sender_ip:
                    self._set_status_label_text(
                        self.lbl_sender,
                        f"Sender(CH{ch.ch_id + 1}): {ch.sender_ip}:{ch.control_port} | Frame: {ch.last_frame_id}"
                    )
                else:
                    self._set_status_label_text(self.lbl_sender, f"Sender(CH{ch.ch_id + 1}): Detecting...")

            params = get_viewer_params(ch)
            params_text = (
                f"Params(CH{ch.ch_id + 1}): {params.describe()} | "
                f"local_fft={get_processing_range_fft_size()}x{get_processing_doppler_fft_size()} | "
                f"view={get_display_range_bin_limit()}x{get_display_doppler_bin_limit()}"
            )
            if ch.last_param_error:
                params_text += f" | last_error={ch.last_param_error}"
            self._set_status_label_text(self.lbl_params, params_text)

            with ch.micro_lock:
                self._set_status_label_text(
                    self.lbl_buffer,
                    f"MD Buffer(CH{ch.ch_id + 1}): {len(ch.micro_doppler_buffer)}/{BUFFER_LENGTH}"
                )

        self.update_phase_probe_text()
        self.refresh_display_button_text()

        # Queue status
        queue_states = " ".join(
            [f"CH{i + 1}:{CHANNELS[i].raw_frame_queue.qsize()}/{CHANNELS[i].display_queue.qsize()}" for i in range(len(CHANNELS))]
        )
        self._set_status_label_text(self.lbl_queue, f"Q(raw/disp) {queue_states}")

        # FPS + DSP time
        if updated_any:
            self.frame_count += 1

        current_time = time.time()
        elapsed = current_time - self.last_update_time
        if elapsed > 0.5:
            fps = self.frame_count / elapsed
            avg_dsp = (self.total_dsp_time / self.frame_count * 1000) if self.frame_count > 0 else 0.0
            self._set_status_label_text(self.lbl_fps, f"FPS: {fps:.1f} | DSP: {avg_dsp:.1f}ms")
            self.frame_count = 0
            self.total_dsp_time = 0.0
            self.last_update_time = current_time

    def closeEvent(self, event):
        global running
        running = False
        print("Closing application...")
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
