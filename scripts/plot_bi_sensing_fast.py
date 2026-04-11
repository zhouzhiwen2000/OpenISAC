
import numpy as np
import sys
import os
import platform
import subprocess
import time
import threading
from queue import Queue
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
    elif hasattr(arr, 'asnumpy'):
        return arr.asnumpy()
    elif hasattr(arr, '__array__'):
        return np.asarray(arr)
    return arr

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
CONTROL_PORT = 9999
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
cfar_max_points = 256
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


def get_processing_range_fft_size():
    return max(1, int(PROCESS_RANGE_FFT_SIZE))


def get_processing_doppler_fft_size():
    return max(1, int(PROCESS_DOPPLER_FFT_SIZE))


def get_display_range_bin_limit():
    return max(1, int(DISPLAY_RANGE_BIN_LIMIT))


def get_display_doppler_bin_limit():
    return max(1, int(DISPLAY_DOPPLER_BIN_LIMIT))


def _integral_image_2d(arr, backend):
    rows, cols = arr.shape
    if backend == "cupy":
        integral = cp.cumsum(cp.cumsum(arr, axis=0), axis=1)
        out = cp.zeros((rows + 1, cols + 1), dtype=arr.dtype)
        out[1:, 1:] = integral
        return out
    if backend == "mlx":
        integral = mx.cumsum(mx.cumsum(arr, axis=0), axis=1)
        out = mx.zeros((rows + 1, cols + 1), dtype=arr.dtype)
        out[1:, 1:] = integral
        return out
    integral = np.cumsum(np.cumsum(arr, axis=0), axis=1, dtype=arr.dtype)
    out = np.zeros((rows + 1, cols + 1), dtype=arr.dtype)
    out[1:, 1:] = integral
    return out


def _compute_cfar_points_from_mask(mask, rd_db, backend_name):
    if backend_name in {"cupy", "mlx"}:
        mask_cpu = to_numpy(mask).astype(bool, copy=False)
    else:
        mask_cpu = np.asarray(mask, dtype=bool)

    hit_idx = np.argwhere(mask_cpu)
    if hit_idx.size == 0:
        return np.empty((0, 2), dtype=np.int32), 0, 0

    raw_hit_count = int(hit_idx.shape[0])
    values = rd_db[mask_cpu]
    order = np.argsort(values)[::-1]
    hit_idx = hit_idx[order]
    if hit_idx.shape[0] > cfar_max_points:
        hit_idx = hit_idx[:cfar_max_points]
    shown_hit_count = int(hit_idx.shape[0])
    return hit_idx.astype(np.int32, copy=False), raw_hit_count, shown_hit_count


def build_cfar_views(rd_db, display_range_bins, display_doppler_bins, downsample):
    rows, cols = rd_db.shape
    ds = max(1, int(downsample))
    outer_h = max(0, int(cfar_train_doppler)) + max(0, int(cfar_guard_doppler))
    outer_w = max(0, int(cfar_train_range)) + max(0, int(cfar_guard_range))
    extra_rows = outer_h * ds
    extra_cols = outer_w * ds

    display_col_stop = min(max(1, int(display_range_bins)), cols)
    center_idx = rows // 2
    display_row_start = max(0, center_idx - max(1, int(display_doppler_bins)) // 2)
    display_row_stop = min(rows, display_row_start + max(1, int(display_doppler_bins)))

    display_row_indices = np.arange(display_row_start, display_row_stop, ds, dtype=np.int32)
    display_col_indices = np.arange(0, display_col_stop, ds, dtype=np.int32)
    if display_row_indices.size == 0 or display_col_indices.size == 0:
        empty = np.empty((0, 0), dtype=rd_db.dtype)
        return empty, empty, 0, 0

    wide_row_start = max(0, display_row_start - extra_rows)
    wide_row_stop = min(rows, display_row_stop + extra_rows)
    wide_row_indices = np.arange(wide_row_start, wide_row_stop, ds, dtype=np.int32)
    raw_wide_col_indices = np.arange(-extra_cols, display_col_stop + extra_cols, ds, dtype=np.int32)
    wide_col_indices = np.mod(raw_wide_col_indices, cols).astype(np.int32, copy=False)

    display_view = rd_db[np.ix_(display_row_indices, display_col_indices)]
    wide_view = rd_db[np.ix_(wide_row_indices, wide_col_indices)]
    row_offset = int(np.searchsorted(wide_row_indices, display_row_indices[0]))
    col_offset = int(np.count_nonzero(raw_wide_col_indices < 0))
    return display_view, wide_view, row_offset, col_offset


def run_ca_cfar_2d(
    rd_db,
    active_row_start=None,
    active_row_stop=None,
    active_col_start=None,
    active_col_stop=None,
    dc_center_row=None,
):
    rows, cols = rd_db.shape
    td = max(0, int(cfar_train_doppler))
    tr = max(0, int(cfar_train_range))
    gd = max(0, int(cfar_guard_doppler))
    gr = max(0, int(cfar_guard_range))
    alpha_db = float(cfar_alpha_db)
    alpha = max(1e-12, float(np.power(10.0, alpha_db / 10.0)))
    min_range = max(0, int(cfar_min_range_bin))
    dc_excl = max(0, int(cfar_dc_exclusion_bins))
    eps = 1e-12

    outer_h = td + gd
    outer_w = tr + gr
    if rows == 0 or cols == 0 or rows <= 2 * outer_h or cols <= 2 * outer_w:
        return np.empty((0, 2), dtype=np.int32), 0, 0, "off", {
            'noise_min': 0.0,
            'noise_max': 0.0,
            'thresh_min': 0.0,
            'thresh_max': 0.0,
            'invalid_cells': 0,
            'nonfinite_cells': 0,
            'nonpositive_cells': 0,
        }

    backend_name = "cpu"
    if USE_NVIDIA_GPU or USE_INTEL_GPU:
        backend_name = "cupy"
        rd_db_backend = cp.asarray(rd_db, dtype=cp.float64)
        power = cp.power(cp.float64(10.0), rd_db_backend / cp.float64(10.0))
        integral = _integral_image_2d(power, backend_name)
        mask = cp.zeros((rows, cols), dtype=cp.bool_)
    elif USE_APPLE_GPU:
        try:
            backend_name = "mlx"
            rd_db_backend = mx.array(rd_db, dtype=mx.float64)
            power = mx.power(mx.array(10.0, dtype=mx.float64), rd_db_backend / 10.0)
            integral = _integral_image_2d(power, backend_name)
            mask = mx.zeros((rows, cols), dtype=mx.bool_)
        except Exception:
            backend_name = "cpu"
            power = np.power(np.float64(10.0), rd_db.astype(np.float64) / np.float64(10.0), dtype=np.float64)
            integral = _integral_image_2d(power, backend_name)
            mask = np.zeros((rows, cols), dtype=bool)
    else:
        power = np.power(np.float64(10.0), rd_db.astype(np.float64) / np.float64(10.0), dtype=np.float64)
        integral = _integral_image_2d(power, backend_name)
        mask = np.zeros((rows, cols), dtype=bool)

    outer_cells = (2 * outer_h + 1) * (2 * outer_w + 1)
    guard_cells = (2 * gd + 1) * (2 * gr + 1)
    training_cells = max(1, outer_cells - guard_cells)

    row_start = outer_h if active_row_start is None else max(outer_h, int(active_row_start))
    row_stop = (rows - outer_h) if active_row_stop is None else min(rows - outer_h, int(active_row_stop))
    col_start = outer_w if active_col_start is None else max(outer_w, int(active_col_start))
    col_stop = (cols - outer_w) if active_col_stop is None else min(cols - outer_w, int(active_col_stop))
    if row_start >= row_stop or col_start >= col_stop:
        return np.empty((0, 2), dtype=np.int32), 0, 0, backend_name, {
            'noise_min': 0.0,
            'noise_max': 0.0,
            'thresh_min': 0.0,
            'thresh_max': 0.0,
            'invalid_cells': 0,
            'nonfinite_cells': 0,
            'nonpositive_cells': 0,
        }

    outer_row_top = row_start - outer_h
    outer_row_bottom = row_stop + outer_h + 1
    outer_col_left = col_start - outer_w
    outer_col_right = col_stop + outer_w + 1
    outer_sum = (
        integral[(outer_row_top + 2 * outer_h + 1):outer_row_bottom, (outer_col_left + 2 * outer_w + 1):outer_col_right]
        - integral[outer_row_top:(row_stop - outer_h), (outer_col_left + 2 * outer_w + 1):outer_col_right]
        - integral[(outer_row_top + 2 * outer_h + 1):outer_row_bottom, outer_col_left:(col_stop - outer_w)]
        + integral[outer_row_top:(row_stop - outer_h), outer_col_left:(col_stop - outer_w)]
    )

    guard_row_top = row_start - gd
    guard_row_bottom = row_stop + gd + 1
    guard_col_left = col_start - gr
    guard_col_right = col_stop + gr + 1
    guard_sum = (
        integral[(guard_row_top + 2 * gd + 1):guard_row_bottom, (guard_col_left + 2 * gr + 1):guard_col_right]
        - integral[guard_row_top:(row_stop - gd), (guard_col_left + 2 * gr + 1):guard_col_right]
        - integral[(guard_row_top + 2 * gd + 1):guard_row_bottom, guard_col_left:(col_stop - gr)]
        + integral[guard_row_top:(row_stop - gd), guard_col_left:(col_stop - gr)]
    )
    noise_mean = (outer_sum - guard_sum) / training_cells
    cut_power = power[row_start:row_stop, col_start:col_stop]
    if backend_name == "cupy":
        finite_mask = cp.isfinite(noise_mean) & cp.isfinite(cut_power)
        positive_mask = noise_mean > cp.float64(eps)
        compare_mask = finite_mask & positive_mask
        safe_noise_mean = cp.where(compare_mask, noise_mean, cp.float64(eps))
        threshold = cp.float64(alpha) * safe_noise_mean
        valid_mask = compare_mask & (cut_power > threshold)
        nonfinite_cells = int(cp.count_nonzero(~finite_mask).item())
        nonpositive_cells = int(cp.count_nonzero(finite_mask & (~positive_mask)).item())
        invalid_cells = nonfinite_cells + nonpositive_cells
        noise_valid = cp.where(compare_mask, safe_noise_mean, cp.nan)
        thresh_valid = cp.where(compare_mask, threshold, cp.nan)
        noise_min = float(cp.nanmin(noise_valid).item()) if invalid_cells < noise_valid.size else 0.0
        noise_max = float(cp.nanmax(noise_valid).item()) if invalid_cells < noise_valid.size else 0.0
        thresh_min = float(cp.nanmin(thresh_valid).item()) if invalid_cells < thresh_valid.size else 0.0
        thresh_max = float(cp.nanmax(thresh_valid).item()) if invalid_cells < thresh_valid.size else 0.0
    elif backend_name == "mlx":
        noise_mean_np = np.asarray(to_numpy(noise_mean), dtype=np.float64)
        cut_power_np = np.asarray(to_numpy(cut_power), dtype=np.float64)
        compare_mask_np = np.isfinite(noise_mean_np) & np.isfinite(cut_power_np) & (noise_mean_np > eps)
        finite_mask_np = np.isfinite(noise_mean_np) & np.isfinite(cut_power_np)
        positive_mask_np = noise_mean_np > eps
        safe_noise_mean_np = np.where(compare_mask_np, noise_mean_np, eps).astype(np.float64, copy=False)
        threshold_np = (float(alpha) * safe_noise_mean_np).astype(np.float64, copy=False)
        valid_mask_np = compare_mask_np & (cut_power_np > threshold_np)
        nonfinite_cells = int((~finite_mask_np).sum())
        nonpositive_cells = int((finite_mask_np & (~positive_mask_np)).sum())
        invalid_cells = nonfinite_cells + nonpositive_cells
        noise_vals = safe_noise_mean_np[compare_mask_np]
        thresh_vals = threshold_np[compare_mask_np]
        noise_min = float(noise_vals.min()) if noise_vals.size > 0 else 0.0
        noise_max = float(noise_vals.max()) if noise_vals.size > 0 else 0.0
        thresh_min = float(thresh_vals.min()) if thresh_vals.size > 0 else 0.0
        thresh_max = float(thresh_vals.max()) if thresh_vals.size > 0 else 0.0
        valid_mask = mx.array(valid_mask_np)
    else:
        compare_mask = np.isfinite(noise_mean) & np.isfinite(cut_power) & (noise_mean > eps)
        finite_mask = np.isfinite(noise_mean) & np.isfinite(cut_power)
        positive_mask = noise_mean > eps
        safe_noise_mean = np.where(compare_mask, noise_mean, eps).astype(np.float64, copy=False)
        threshold = (float(alpha) * safe_noise_mean).astype(np.float64, copy=False)
        valid_mask = compare_mask & (cut_power > threshold)
        nonfinite_cells = int((~finite_mask).sum())
        nonpositive_cells = int((finite_mask & (~positive_mask)).sum())
        invalid_cells = nonfinite_cells + nonpositive_cells
        noise_vals = safe_noise_mean[compare_mask]
        thresh_vals = threshold[compare_mask]
        noise_min = float(noise_vals.min()) if noise_vals.size > 0 else 0.0
        noise_max = float(noise_vals.max()) if noise_vals.size > 0 else 0.0
        thresh_min = float(thresh_vals.min()) if thresh_vals.size > 0 else 0.0
        thresh_max = float(thresh_vals.max()) if thresh_vals.size > 0 else 0.0

    if min_range > 0:
        valid_mask[:, :min(valid_mask.shape[1], min_range)] = False

    center_row = (rows // 2) if dc_center_row is None else int(dc_center_row)
    center_row_local = center_row - row_start
    if dc_excl > 0 and 0 <= center_row_local < valid_mask.shape[0]:
        lo = max(0, center_row_local - dc_excl)
        hi = min(valid_mask.shape[0], center_row_local + dc_excl + 1)
        valid_mask[lo:hi, :] = False

    mask[row_start:row_stop, col_start:col_stop] = valid_mask
    points, raw_hits, shown_hits = _compute_cfar_points_from_mask(mask, rd_db, backend_name)
    stats = {
        'noise_min': noise_min,
        'noise_max': noise_max,
        'thresh_min': thresh_min,
        'thresh_max': thresh_max,
        'invalid_cells': invalid_cells,
        'nonfinite_cells': nonfinite_cells,
        'nonpositive_cells': nonpositive_cells,
    }
    return points, raw_hits, shown_hits, backend_name, stats


def cluster_detected_targets(cfar_points, rd_data):
    if cfar_points is None:
        return []

    points = np.asarray(cfar_points, dtype=np.int32)
    if points.size == 0 or rd_data is None:
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


def get_viewer_params():
    with viewer_params_lock:
        return viewer_params


def set_viewer_params(params):
    global viewer_params, last_param_error
    with viewer_params_lock:
        viewer_params = params
    last_param_error = None

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

def send_skip_command():
    global sender_ip
    for _ in range(100):
        with sender_lock:
            if sender_ip is not None:
                break
        time.sleep(0.1)
    if sender_ip is None:
        print("Error: Cannot send command - sender IP not detected.")
        return
    try:
        request_viewer_params()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        command_data = struct.pack("!4s4si", b"CMD ", b"SKIP", 1)
        sock.sendto(command_data, (sender_ip, CONTROL_PORT))
        print(f"Sent skip FFT command to {sender_ip}: SKIP 1")
        sock.close()
        time.sleep(0.05)
        request_viewer_params()
    except Exception as e:
        print(f"Failed to send skip FFT command: {str(e)}")


def request_viewer_params():
    global sender_ip
    if sender_ip is None:
        return
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(build_params_request(0), (sender_ip, CONTROL_PORT))
        sock.close()
    except Exception as e:
        print(f"Failed to request viewer params: {e}")

def udp_receiver():
    global sender_ip, CONTROL_PORT, last_param_error
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_BUFFER_SIZE)
    except:
        pass
    try:
        sock.bind((UDP_IP, UDP_PORT))
        print(f"Listening on UDP port {UDP_PORT}")
    except Exception as e:
        print(f"Socket bind error: {e}")
        sys.exit(1)
    
    while True:
        try:
            data, addr = sock.recvfrom(MAX_CHUNK_SIZE + HEADER_SIZE)
            with sender_lock:
                if sender_ip is None:
                    sender_ip = addr[0]
                    print(f"Detected data sender IP: {sender_ip}")
            if len(data) >= 8 and data[:4] == CTRL_HEADER:
                command = data[4:8]
                if CONTROL_PORT != addr[1]:
                    CONTROL_PORT = addr[1]
                if command == PARAMS_COMMAND:
                    params = parse_params_packet(data)
                    if params is not None:
                        set_viewer_params(params)
                        print(f"Viewer params: {params.describe()}")
                elif command == READY_COMMAND:
                    if get_viewer_params().version == 0 or last_param_error is not None:
                        request_viewer_params()
                continue
            if len(data) < HEADER_SIZE:
                continue
            try:
                frame_id, total_chunks, chunk_id = struct.unpack("!III", data[:HEADER_SIZE])
            except struct.error:
                continue
            chunk_data = data[HEADER_SIZE:]
            with buffer_lock:
                if current_frame.frame_id != frame_id:
                    current_frame.init(frame_id, total_chunks)
                if current_frame.add_chunk(chunk_id, chunk_data):
                    frame_data = current_frame.assemble_frame(get_viewer_params())
                    if raw_frame_queue.full():
                        try:
                            raw_frame_queue.get_nowait()
                        except:
                            pass
                    raw_frame_queue.put(frame_data)
        except socket.error as e:
            if e.errno != 10054:
                print(f"Socket error: {e}")
        except Exception as e:
            print(f"Unexpected error in receiver: {str(e)}")
            last_param_error = str(e)
            request_viewer_params()

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
    global range_time_data

    if runtime_params.is_dense_range_doppler():
        rd_complex = np.asarray(
            frame_data[:runtime_params.wire_rows, :runtime_params.wire_cols],
            dtype=np.complex64,
        )
        rd_shifted = np.fft.fftshift(rd_complex, axes=0)
        with range_time_lock:
            range_time_data = None
        return 20.0 * np.log10(np.abs(rd_shifted) + 1e-12)

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
        magnitude_db = 20 * cp.log10(magnitude + 1e-12)
        with range_time_lock:
            range_time_data = range_time_cpu
        return to_numpy(magnitude_db)
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
        range_time_cpu = np.array(range_time_view, copy=False)
        with range_time_lock:
            range_time_data = range_time_cpu
        return np.array(magnitude_db, copy=False).astype(np.float32, copy=False)
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
        with range_time_lock:
            range_time_data = range_time_view
        return magnitude_db.astype(np.float32, copy=False)
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
        with range_time_lock:
            range_time_data = range_time_view
        return magnitude_db.astype(np.float32, copy=False)

def accumulate_range_time_data():
    with range_time_lock:
        if range_time_data is not None and range_time_data.size > 0:
            range_idx = min(selected_range_bin, range_time_data.shape[1] - 1)
            with buffer_lock:
                micro_doppler_buffer.extend(range_time_data[:, range_idx])

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
            rd_spectrum = process_range_doppler(raw_frame, runtime_params)
            display_range_bins = get_display_range_bin_limit()
            display_doppler_bins = get_display_doppler_bin_limit()
            rd_spectrum_plot, rd_spectrum_cfar, cfar_row_offset, cfar_col_offset = build_cfar_views(
                rd_spectrum, display_range_bins, display_doppler_bins, DISPLAY_DOWNSAMPLE
            )
            cfar_points = np.empty((0, 2), dtype=np.int32)
            cfar_hits = 0
            cfar_shown_hits = 0
            cfar_backend = "off"
            cfar_stats = None
            if cfar_enabled and rd_spectrum_plot.size > 0:
                cfar_points, cfar_hits, cfar_shown_hits, cfar_backend, cfar_stats = run_ca_cfar_2d(
                    rd_spectrum_cfar,
                    active_row_start=cfar_row_offset,
                    active_row_stop=cfar_row_offset + rd_spectrum_plot.shape[0],
                    active_col_start=cfar_col_offset,
                    active_col_stop=cfar_col_offset + rd_spectrum_plot.shape[1],
                    dc_center_row=cfar_row_offset + (rd_spectrum_plot.shape[0] // 2),
                )
                if cfar_points.size > 0:
                    cfar_points = cfar_points - np.array([cfar_row_offset, cfar_col_offset], dtype=np.int32)
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
                'rd': rd_spectrum_plot,
                'cfar_points': cfar_points,
                'cfar_hits': cfar_hits,
                'cfar_shown_hits': cfar_shown_hits,
                'cfar_backend': cfar_backend,
                'cfar_stats': cfar_stats,
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
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(struct.pack("!4s4si", header, cmd_id, int(value)), (sender_ip, CONTROL_PORT))
        print(f"Sent command {cmd_id} to {sender_ip}: {value}")
        sock.close()
    except Exception as e:
        print(f"Failed to send command: {str(e)}")

def send_alignment_command(val):
    send_command(b"CMD ", b"ALGN", val)

def send_strd_command(val):
    try:
        strd_val = max(1, min(get_viewer_params().max_strd_value(), int(val)))
        send_command(b"CMD ", b"STRD", strd_val)
        time.sleep(0.02)
        request_viewer_params()
    except ValueError:
        print(f"Invalid STRD value: {val}")

def send_mti_command(enabled):
    send_command(b"CMD ", b"MTI ", 1 if enabled else 0)

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
        self.rd_img = pg.ImageItem()
        self.rd_plot.addItem(self.rd_img)
        self.rd_plot.setLabel('left', 'Doppler Bin')
        self.rd_plot.setLabel('bottom', 'Range Bin')
        # Set colormap
        self.rd_img.setLookupTable(pg.colormap.get('turbo').getLookupTable())
        # Add color bar
        self.rd_colorbar = pg.ColorBarItem(values=(-40, 20), colorMap=pg.colormap.get('turbo'), interactive=False)
        self.rd_colorbar.setImageItem(self.rd_img, insert_in=self.rd_plot.plotItem)
        self.rd_cfar_marker = pg.ScatterPlotItem(
            [], [], symbol='o', size=7,
            pen=pg.mkPen('#00e5ff', width=1.5),
            brush=pg.mkBrush(0, 229, 255, 70)
        )
        self.rd_plot.addItem(self.rd_cfar_marker)
        plot_layout.addWidget(self.rd_plot)

        # Micro-Doppler Plot
        self.md_plot = pg.PlotWidget(title="Micro-Doppler Spectrum")
        self.md_img = pg.ImageItem()
        self.md_plot.addItem(self.md_img)
        self.md_plot.setLabel('left', 'Time')
        self.md_plot.setLabel('bottom', 'Frequency')
        self.md_img.setLookupTable(pg.colormap.get('turbo').getLookupTable())
        # Add color bar
        self.md_colorbar = pg.ColorBarItem(values=(-30, 30), colorMap=pg.colormap.get('turbo'), interactive=False)
        self.md_colorbar.setImageItem(self.md_img, insert_in=self.md_plot.plotItem)
        plot_layout.addWidget(self.md_plot)

        # Control Panel
        control_panel = QtWidgets.QWidget()
        control_panel.setFixedWidth(self._control_panel_width)
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        control_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        main_layout.addWidget(control_panel, stretch=1)

        # Status Labels
        self.lbl_fps = QtWidgets.QLabel("FPS: 0.0")
        self.lbl_queue = QtWidgets.QLabel("Queue: 0")
        self.lbl_sender = QtWidgets.QLabel("Sender: Detecting...")
        self.lbl_params = QtWidgets.QLabel("Params: waiting...")
        self.lbl_buffer = QtWidgets.QLabel("MD Buffer: 0/5000")
        self.lbl_cfar = QtWidgets.QLabel("CFAR: off")
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
        self.targets_text.setPlainText("Top Targets: waiting CFAR detections")
        control_layout.addSpacing(20)

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
        control_layout.addSpacing(12)

        self.btn_cfar = QtWidgets.QPushButton("CA-CFAR: OFF")
        self.btn_cfar.setCheckable(True)
        self.btn_cfar.setChecked(False)
        self.btn_cfar.clicked.connect(self.toggle_cfar)
        control_layout.addWidget(self.btn_cfar)

        cfar_train_layout = QtWidgets.QHBoxLayout()
        cfar_train_layout.addWidget(QtWidgets.QLabel("Train D:"))
        self.txt_cfar_train_d = QtWidgets.QLineEdit(str(cfar_train_doppler))
        cfar_train_layout.addWidget(self.txt_cfar_train_d)
        cfar_train_layout.addWidget(QtWidgets.QLabel("Train R:"))
        self.txt_cfar_train_r = QtWidgets.QLineEdit(str(cfar_train_range))
        cfar_train_layout.addWidget(self.txt_cfar_train_r)
        control_layout.addLayout(cfar_train_layout)

        cfar_guard_layout = QtWidgets.QHBoxLayout()
        cfar_guard_layout.addWidget(QtWidgets.QLabel("Guard D:"))
        self.txt_cfar_guard_d = QtWidgets.QLineEdit(str(cfar_guard_doppler))
        cfar_guard_layout.addWidget(self.txt_cfar_guard_d)
        cfar_guard_layout.addWidget(QtWidgets.QLabel("Guard R:"))
        self.txt_cfar_guard_r = QtWidgets.QLineEdit(str(cfar_guard_range))
        cfar_guard_layout.addWidget(self.txt_cfar_guard_r)
        control_layout.addLayout(cfar_guard_layout)

        cfar_misc_layout = QtWidgets.QHBoxLayout()
        cfar_misc_layout.addWidget(QtWidgets.QLabel("Alpha(dB):"))
        self.txt_cfar_alpha = QtWidgets.QLineEdit(f"{cfar_alpha_db:.2f}")
        cfar_misc_layout.addWidget(self.txt_cfar_alpha)
        cfar_misc_layout.addWidget(QtWidgets.QLabel("Min R:"))
        self.txt_cfar_min_range = QtWidgets.QLineEdit(str(cfar_min_range_bin))
        cfar_misc_layout.addWidget(self.txt_cfar_min_range)
        control_layout.addLayout(cfar_misc_layout)

        cfar_dc_layout = QtWidgets.QHBoxLayout()
        cfar_dc_layout.addWidget(QtWidgets.QLabel("DC Excl:"))
        self.txt_cfar_dc_excl = QtWidgets.QLineEdit(str(cfar_dc_exclusion_bins))
        btn_cfar_apply = QtWidgets.QPushButton("Apply CFAR")
        btn_cfar_apply.clicked.connect(self.apply_cfar_settings)
        cfar_dc_layout.addWidget(self.txt_cfar_dc_excl)
        cfar_dc_layout.addWidget(btn_cfar_apply)
        control_layout.addLayout(cfar_dc_layout)
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
        for label, val in [('+10', 10), ('-10', -10), ('+1', 1), ('-1', -1)]:
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
        self.timer.start(1)  # Max FPS

        send_skip_command()
        self.last_update_time = time.time()
        self.frame_count = 0
        self.total_dsp_time = 0.0
        self.targets_window.show()

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

    def update_top_targets_text(self, latest):
        if latest is None:
            self.targets_text.setPlainText("Top Targets: N/A")
            return

        rd_data = latest.get('rd')
        cfar_points = latest.get('cfar_points')
        if rd_data is None or cfar_points is None or len(cfar_points) == 0:
            self.targets_text.setPlainText("Top Targets: none")
            return

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
        self.btn_cfar.setText(f"CA-CFAR: {'ON' if cfar_enabled else 'OFF'}")
        self.btn_cfar.setStyleSheet("QPushButton:checked { background-color: lightgreen; }")

    def apply_cfar_settings(self):
        global cfar_train_doppler, cfar_train_range
        global cfar_guard_doppler, cfar_guard_range
        global cfar_alpha_db, cfar_min_range_bin, cfar_dc_exclusion_bins
        try:
            cfar_train_doppler = max(0, int(self.txt_cfar_train_d.text()))
            cfar_train_range = max(0, int(self.txt_cfar_train_r.text()))
            cfar_guard_doppler = max(0, int(self.txt_cfar_guard_d.text()))
            cfar_guard_range = max(0, int(self.txt_cfar_guard_r.text()))
            cfar_alpha_db = float(self.txt_cfar_alpha.text())
            cfar_min_range_bin = max(0, int(self.txt_cfar_min_range.text()))
            cfar_dc_exclusion_bins = max(0, int(self.txt_cfar_dc_excl.text()))
            alpha_linear = float(np.power(10.0, cfar_alpha_db / 10.0))
            print(
                "CA-CFAR updated: "
                f"train=({cfar_train_doppler},{cfar_train_range}) "
                f"guard=({cfar_guard_doppler},{cfar_guard_range}) "
                f"alpha_db={cfar_alpha_db:.2f} alpha_linear={alpha_linear:.3f} "
                f"min_range={cfar_min_range_bin} dc_excl={cfar_dc_exclusion_bins}"
            )
        except ValueError:
            print("Invalid CA-CFAR settings")

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

    def toggle_doppler_win(self):
        global enable_doppler_window
        enable_doppler_window = not enable_doppler_window
        self.update_toggle_style(self.btn_doppler_win, enable_doppler_window)

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
            cfar_status = f"CFAR: {'on' if cfar_enabled else 'off'}"
            if cfar_enabled:
                cfar_stats = latest.get('cfar_stats') or {}
                cfar_status += (
                    f" raw={int(latest.get('cfar_hits', 0))}"
                    f" shown={int(latest.get('cfar_shown_hits', 0))}"
                    f" backend={latest.get('cfar_backend', 'cpu')}"
                    f" invalid={int(cfar_stats.get('invalid_cells', 0))}"
                    f" nonpos={int(cfar_stats.get('nonpositive_cells', 0))}"
                    f" nonfinite={int(cfar_stats.get('nonfinite_cells', 0))}"
                    f" noise=[{cfar_stats.get('noise_min', 0.0):.2e},{cfar_stats.get('noise_max', 0.0):.2e}]"
                )
            cfar_status_short = f"CFAR: {'on' if cfar_enabled else 'off'}"
            if cfar_enabled:
                cfar_stats = latest.get('cfar_stats') or {}
                cfar_status_short += (
                    f" raw={int(latest.get('cfar_hits', 0))}"
                    f" sh={int(latest.get('cfar_shown_hits', 0))}"
                    f" inv={int(cfar_stats.get('invalid_cells', 0))}"
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
        self.targets_window.close()
        print("Closing application...")
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    send_skip_command()
    sys.exit(app.exec())
