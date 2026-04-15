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
from pathlib import Path
import socket
import struct
import datetime
import scipy.io as sio
import yaml

# PyQt6 + PyQtGraph Imports
from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

from sensing_runtime_protocol import (
    AGGREGATE_HEADER_STRUCT,
    AGGREGATE_MAGIC_VERSION,
    CTRL_HEADER,
    PARAMS_COMMAND,
    READY_COMMAND,
    ViewerRuntimeParams,
    build_params_request,
    decode_aggregate_sensing_payload,
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
target_cluster_doppler_gap = 2
target_cluster_range_gap = 2
PHASE_CALIBRATION_TARGET_SAMPLES = 300
PHASE_CALIBRATION_PROGRESS_INTERVAL = 25
PHASE_CALIBRATION_MAD_SCALE = 3.5
PHASE_CALIBRATION_MIN_ERROR = 0.05
PHASE_CALIBRATION_MIN_INLIERS = 80
PHASE_CALIBRATION_AUTO_SAVE = True
TARGET_SECTOR_HALF_ANGLE_DEG = 90.0
TARGET_SECTOR_POINT_LIMIT = 5
TARGET_SECTOR_RANGE_RINGS = 4
TARGET_SECTOR_DEFAULT_ZERO_RANGE_BIN = 0
TARGET_SECTOR_DEFAULT_MAX_RANGE_BINS = 100
DISPLAY_TIMER_INTERVAL_MS = 5

RAW_QUEUE_SIZE = 20
DISPLAY_QUEUE_SIZE = 5
DISPLAY_DOWNSAMPLE = 1
BUFFER_LENGTH = 5000
C_LIGHT_MPS = 299792458.0
ANTENNA_SPACING_M = 42.83e-3
AGGREGATE_FRAME_QUEUE_SIZE = 20

selected_range_bin = 0
show_micro_doppler = True
display_channel = 0

running = True


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


def _window_sum_from_integral(integral, top, left, bottom, right):
    return (
        integral[bottom + 1, right + 1]
        - integral[top, right + 1]
        - integral[bottom + 1, left]
        + integral[top, left]
    )


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
    min_power_db = float(cfar_min_power_db)
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
        valid_mask &= rd_db_backend[row_start:row_stop, col_start:col_stop] >= cp.float64(min_power_db)
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
        valid_mask_np &= rd_db[row_start:row_stop, col_start:col_stop] >= min_power_db
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
        valid_mask &= rd_db[row_start:row_stop, col_start:col_stop] >= min_power_db
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
        'power_min_db': min_power_db,
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
    global latest_phase_bundle_frame_id, latest_phase_bundle_rd_complex
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
    while True:
        try:
            aggregate_frame_queue.get_nowait()
        except Empty:
            break
        except Exception:
            break
    with phase_bundle_lock:
        latest_phase_bundle_frame_id = None
        latest_phase_bundle_rd_complex = None


def parse_args():
    parser = argparse.ArgumentParser(description="Monostatic sensing viewer (fast, multi-channel)")
    parser.add_argument(
        "--port",
        type=int,
        default=8888,
        help="UDP port for aggregated multi-channel sensing frames",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=2,
        help="Logical channel count carried by the aggregated sensing stream",
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

    def assemble_payload(self):
        return self.frame_id, b"".join(self.buffer[:self.total_chunks])


class ChannelRuntime:
    def __init__(self, ch_id, udp_port, control_port):
        self.ch_id = ch_id
        self.udp_port = udp_port
        self.control_port = control_port

        self.frame_buffer = FrameBuffer()
        self.buffer_lock = threading.Lock()

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
UDP_PORT = int(args.port)
initial_channel_count = max(1, int(args.channels))
CHANNELS = [ChannelRuntime(idx, UDP_PORT, args.control_port) for idx in range(initial_channel_count)]
if not CHANNELS:
    CHANNELS = [ChannelRuntime(0, 8888, args.control_port)]

aggregate_frame_queue = Queue(maxsize=AGGREGATE_FRAME_QUEUE_SIZE)
phase_bundle_lock = threading.Lock()
latest_phase_bundle_frame_id = None
latest_phase_bundle_rd_complex = None


def get_viewer_params(ch):
    with ch.viewer_params_lock:
        return ch.viewer_params


def set_viewer_params(ch, viewer_params):
    with ch.viewer_params_lock:
        ch.viewer_params = viewer_params
        ch.last_param_summary = viewer_params.describe()
        ch.last_param_error = None


def _clear_queue(queue_obj):
    while True:
        try:
            queue_obj.get_nowait()
        except Empty:
            break
        except Exception:
            break


def _ensure_channel_count(expected_count, reason="auto-detect"):
    global CHANNELS, display_channel, latest_phase_bundle_frame_id, latest_phase_bundle_rd_complex
    expected_count = max(1, int(expected_count))
    current_count = len(CHANNELS)
    if expected_count == current_count:
        return False

    base_control_port = CHANNELS[0].control_port if CHANNELS else args.control_port
    base_sender_ip = CHANNELS[0].sender_ip if CHANNELS else None
    base_params = get_viewer_params(CHANNELS[0]) if CHANNELS else LEGACY_VIEWER_PARAMS

    new_channels = []
    for ch_idx in range(expected_count):
        if ch_idx < current_count:
            ch = CHANNELS[ch_idx]
        else:
            ch = ChannelRuntime(ch_idx, UDP_PORT, base_control_port)
            with ch.sender_lock:
                ch.sender_ip = base_sender_ip
            set_viewer_params(ch, base_params)
        new_channels.append(ch)

    for ch_idx, ch in enumerate(new_channels):
        ch.ch_id = ch_idx
        _clear_queue(ch.display_queue)
        ch.current_display_data = {}
        ch.last_frame_id = None
        with ch.range_time_lock:
            ch.range_time_data = None
        with ch.micro_lock:
            ch.micro_doppler_buffer.clear()

    CHANNELS = new_channels
    display_channel = max(0, min(display_channel, len(CHANNELS) - 1))
    _clear_queue(aggregate_frame_queue)
    with phase_bundle_lock:
        latest_phase_bundle_frame_id = None
        latest_phase_bundle_rd_complex = None
    print(f"[AGG] Auto-adjusted logical channel count: {current_count} -> {expected_count} ({reason})")
    return True


# ====== Network + Control ======
def _get_target_channels(target_ch_id=None):
    if target_ch_id is None:
        return CHANNELS
    if target_ch_id < 0 or target_ch_id >= len(CHANNELS):
        return []
    return [CHANNELS[target_ch_id]]


def _get_control_endpoint(target_ch_id=None):
    targets = _get_target_channels(target_ch_id)
    if not targets:
        return None
    for ch in targets:
        with ch.sender_lock:
            if ch.sender_ip is not None:
                return ch.sender_ip, ch.control_port
    return None


def _current_control_channel_id():
    if not CHANNELS:
        return None
    return max(0, min(display_channel, len(CHANNELS) - 1))


def send_control_command(cmd_id, value, target_ch_id=None):
    endpoint = _get_control_endpoint(target_ch_id)
    if endpoint is None:
        print("Error: Invalid target channel")
        return

    ip, cport = endpoint
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        command_data = struct.pack("!4s4si", b"CMD ", cmd_id, int(value))
        sock.sendto(command_data, (ip, cport))
        sock.close()
        target_desc = "all channels" if target_ch_id is None else f"CH{target_ch_id + 1}"
        print(f"Sent {cmd_id.decode(errors='ignore').strip()}={value} to {target_desc} via {ip}:{cport}")
    except Exception as e:
        print(f"Failed to send command via {ip}:{cport}: {e}")


def send_shared_control_command(cmd_id, value):
    endpoint = _get_control_endpoint(None)
    if endpoint is None:
        print("Error: Shared sensing sender not detected yet.")
        return

    ip, cport = endpoint
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        command_data = struct.pack("!4s4si", b"CMD ", cmd_id, int(value))
        sock.sendto(command_data, (ip, cport))
        sock.close()
        print(f"Sent shared {cmd_id.decode(errors='ignore').strip()}={value} via {ip}:{cport}")
    except Exception as e:
        print(f"Failed to send shared command via {ip}:{cport}: {e}")


def request_shared_viewer_params():
    endpoint = _get_control_endpoint(None)
    if endpoint is None:
        return

    request_packet = build_params_request(0)
    ip, cport = endpoint
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(request_packet, (ip, cport))
        sock.close()
    except Exception as e:
        print(f"Failed to request shared viewer params via {ip}:{cport}: {e}")


def request_viewer_params(target_ch_id=None):
    endpoint = _get_control_endpoint(target_ch_id)
    if endpoint is None:
        return

    request_packet = build_params_request(0)
    ip, cport = endpoint
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(request_packet, (ip, cport))
        sock.close()
    except Exception as e:
        print(f"Failed to request viewer params via {ip}:{cport}: {e}")


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

    request_shared_viewer_params()
    send_shared_control_command(b"SKIP", 1)
    time.sleep(0.05)
    request_shared_viewer_params()


def _queue_display_frame(ch, frame_item):
    if ch.display_queue.full():
        try:
            ch.display_queue.get_nowait()
        except Exception:
            pass
    ch.display_queue.put(frame_item)


def _queue_aggregate_frame(frame_id, channel_frames):
    if aggregate_frame_queue.full():
        try:
            aggregate_frame_queue.get_nowait()
        except Exception:
            pass
    aggregate_frame_queue.put((int(frame_id), channel_frames))


def _update_detected_sender(target_channels, addr, update_control_port=False):
    for ch in target_channels:
        with ch.sender_lock:
            if ch.sender_ip is None:
                ch.sender_ip = addr[0]
                print(f"[CH{ch.ch_id + 1}] Detected sender IP: {ch.sender_ip}")
            if update_control_port and ch.control_port != addr[1]:
                ch.control_port = addr[1]
                print(f"[CH{ch.ch_id + 1}] Control port updated: {ch.sender_ip}:{ch.control_port}")


def _handle_viewer_params(target_channels, params, prefix):
    if params.aggregated_stream():
        _ensure_channel_count(params.stream_channel_count, reason="viewer params")
        target_channels = CHANNELS
    for ch in target_channels:
        set_viewer_params(ch, params)
    print(f"{prefix} Viewer params: {params.describe()}")


def _aggregate_logical_channel_count(payload):
    if len(payload) < AGGREGATE_HEADER_STRUCT.size:
        return None
    try:
        magic_version, channel_count, _, channel_mask, _ = AGGREGATE_HEADER_STRUCT.unpack_from(payload)
    except struct.error:
        return None
    if magic_version != AGGREGATE_MAGIC_VERSION:
        return None
    highest_channel = max(
        [bit for bit in range(32) if channel_mask & (1 << bit)],
        default=(int(channel_count) - 1),
    )
    return max(int(channel_count), highest_channel + 1)


def _dispatch_aggregate_payload(frame_id_hint, aggregate_payload, viewer_params, prefix):
    detected_count = _aggregate_logical_channel_count(aggregate_payload)
    if detected_count is not None:
        _ensure_channel_count(detected_count, reason="aggregate frame")

    _, decoded_frames = decode_aggregate_sensing_payload(frame_id_hint, aggregate_payload, viewer_params)
    bundle_frame_id = int(frame_id_hint)
    channel_frames = [None] * len(CHANNELS)
    for ch_id, decoded in decoded_frames:
        if ch_id < 0 or ch_id >= len(CHANNELS):
            print(
                f"{prefix} Dropping channel {ch_id} from aggregate frame {decoded.frame_id}: "
                f"viewer only has {len(CHANNELS)} channels"
            )
            continue
        bundle_frame_id = int(decoded.frame_id)
        channel_frames[ch_id] = decoded.matrix
    if any(frame is not None for frame in channel_frames):
        _queue_aggregate_frame(bundle_frame_id, channel_frames)


def udp_receiver(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_BUFFER_SIZE)
    except Exception:
        pass

    try:
        sock.bind((UDP_IP, port))
        print(f"[AGG] Listening on UDP port {port} for aggregated sensing frames")
    except Exception as e:
        print(f"[AGG] Socket bind error: {e}")
        return

    logged_waiting_for_params = False
    while running:
        try:
            data, addr = sock.recvfrom(MAX_CHUNK_SIZE + HEADER_SIZE)

            _update_detected_sender(CHANNELS, addr)

            if len(data) >= 8 and data[:4] == CTRL_HEADER:
                command = data[4:8]
                if command == PARAMS_COMMAND:
                    params = parse_params_packet(data)
                    if params is not None:
                        _update_detected_sender(CHANNELS, addr, update_control_port=True)
                        _handle_viewer_params(CHANNELS, params, "[AGG]")
                        logged_waiting_for_params = False
                elif command == READY_COMMAND:
                    _update_detected_sender(CHANNELS, addr, update_control_port=True)
                    if CHANNELS and (get_viewer_params(CHANNELS[0]).version == 0 or CHANNELS[0].last_param_error is not None):
                        request_viewer_params(0)
                continue

            if len(data) < HEADER_SIZE:
                continue

            try:
                frame_id, total_chunks, chunk_id = struct.unpack("!III", data[:HEADER_SIZE])
            except struct.error:
                continue

            chunk_data = data[HEADER_SIZE:]
            frame_buffer = CHANNELS[0].frame_buffer
            with CHANNELS[0].buffer_lock:
                if frame_buffer.frame_id != frame_id:
                    frame_buffer.init(frame_id, total_chunks)
                if not frame_buffer.add_chunk(chunk_id, chunk_data):
                    continue

                viewer_params = get_viewer_params(CHANNELS[0]) if CHANNELS else LEGACY_VIEWER_PARAMS
                frame_id_done, payload = frame_buffer.assemble_payload()
                if not viewer_params.aggregated_stream() and viewer_params.version == 0:
                    if not logged_waiting_for_params:
                        print("[AGG] Aggregate frame detected before viewer params arrived; requesting params")
                        logged_waiting_for_params = True
                    request_viewer_params(0)
                    continue
                if len(payload) < 4 or struct.unpack("!I", payload[:4])[0] != AGGREGATE_MAGIC_VERSION:
                    print("[AGG] Ignoring non-aggregated sensing frame; legacy per-channel UDP mode is no longer supported")
                    continue
                _dispatch_aggregate_payload(frame_id_done, payload, viewer_params, "[AGG]")

        except socket.error as e:
            if getattr(e, "errno", None) != 10054:
                print(f"[AGG] Socket error: {e}")
        except Exception as e:
            print(f"[AGG] Receiver error: {e}")
            for ch in CHANNELS:
                ch.last_param_error = str(e)
            request_viewer_params(0 if CHANNELS else None)

    sock.close()


receiver_threads = []
_t = threading.Thread(target=udp_receiver, args=(UDP_PORT,), daemon=True)
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


def process_range_doppler_batch(channel_frames, viewer_params, max_view_range_bins=None):
    active_items = [(ch_idx, frame) for ch_idx, frame in channel_frames if frame is not None]
    if not active_items:
        return {}
    if len(active_items) == 1:
        ch_idx, frame = active_items[0]
        rd_spectrum, range_time_view, rd_complex = process_range_doppler(
            frame,
            viewer_params,
            max_view_range_bins=max_view_range_bins,
        )
        return {ch_idx: (rd_spectrum, range_time_view, rd_complex)}

    results = {}

    if viewer_params.is_dense_range_doppler():
        wire_rows = max(1, int(viewer_params.wire_rows))
        wire_cols = max(1, int(viewer_params.wire_cols))
        raw_batch = np.stack(
            [np.asarray(frame[:wire_rows, :wire_cols], dtype=np.complex64) for _, frame in active_items],
            axis=0,
        )
        rd_shifted = np.fft.fftshift(raw_batch, axes=1)
        magnitude_db = 20.0 * np.log10(np.abs(rd_shifted) + 1e-12)
        for batch_idx, (ch_idx, _) in enumerate(active_items):
            results[ch_idx] = (
                magnitude_db[batch_idx].astype(np.float32, copy=False),
                None,
                rd_shifted[batch_idx].astype(np.complex64, copy=False),
            )
        return results

    if not viewer_params.raw_fft_locally_supported():
        raise ValueError(f"Unsupported sensing frame format: {viewer_params.describe()}")

    raw_rows = max(1, int(viewer_params.active_rows))
    raw_cols = max(1, int(viewer_params.active_cols))
    range_fft_size = max(raw_cols, get_processing_range_fft_size())
    doppler_fft_size = max(raw_rows, get_processing_doppler_fft_size())

    if max_view_range_bins is None:
        max_view_range_bins = get_display_range_bin_limit()
    max_view_range_bins = min(max_view_range_bins, range_fft_size)

    raw_batch = np.stack(
        [np.asarray(frame[:raw_rows, :raw_cols], dtype=np.complex64) for _, frame in active_items],
        axis=0,
    )

    if USE_NVIDIA_GPU or USE_INTEL_GPU:
        range_win = get_range_window(raw_cols) if enable_range_window else cp.ones((1, raw_cols), dtype=cp.float32)
        doppler_win = get_doppler_window(raw_rows) if enable_doppler_window else cp.ones((raw_rows, 1), dtype=cp.float32)
        range_win_3d = cp.reshape(range_win, (1, 1, raw_cols))
        doppler_win_3d = cp.reshape(doppler_win, (1, raw_rows, 1))
        batch_gpu = cp.asarray(raw_batch, dtype=cp.complex64)
        shifted_batch = cp.fft.fftshift(batch_gpu, axes=2)
        windowed_batch = shifted_batch * range_win_3d

        padded_data = cp.zeros((batch_gpu.shape[0], raw_rows, range_fft_size), dtype=cp.complex64)
        padded_data[:, :, :raw_cols] = windowed_batch
        range_time = cp.fft.ifft(padded_data, axis=2) * range_fft_size
        range_time_view = range_time[:, :, :max_view_range_bins] if max_view_range_bins < range_fft_size else range_time

        padded_doppler = cp.zeros((batch_gpu.shape[0], doppler_fft_size, range_time_view.shape[2]), dtype=cp.complex64)
        padded_doppler[:, :raw_rows, :] = range_time_view * doppler_win_3d
        doppler_fft = cp.fft.fft(padded_doppler, axis=1)
        doppler_shifted = cp.fft.fftshift(doppler_fft, axes=1)

        magnitude_db = 20.0 * cp.log10(cp.abs(doppler_shifted) / np.sqrt(raw_rows * raw_cols) + 1e-12)
        magnitude_db_cpu = to_numpy(magnitude_db)
        range_time_cpu = to_numpy(range_time_view)
        doppler_shifted_cpu = to_numpy(doppler_shifted)
        for batch_idx, (ch_idx, _) in enumerate(active_items):
            results[ch_idx] = (
                np.asarray(magnitude_db_cpu[batch_idx], dtype=np.float32),
                np.asarray(range_time_cpu[batch_idx], dtype=np.complex64),
                np.asarray(doppler_shifted_cpu[batch_idx], dtype=np.complex64),
            )
        return results

    if USE_APPLE_GPU:
        range_win = get_range_window(raw_cols) if enable_range_window else mx.ones((1, raw_cols), dtype=mx.float32)
        doppler_win = get_doppler_window(raw_rows) if enable_doppler_window else mx.ones((raw_rows, 1), dtype=mx.float32)
        range_win_3d = mx.reshape(range_win, (1, 1, raw_cols))
        doppler_win_3d = mx.reshape(doppler_win, (1, raw_rows, 1))
        batch_gpu = mx.array(raw_batch, dtype=mx.complex64)
        shifted_batch = mx.fft.fftshift(batch_gpu, axes=(2,))
        windowed_batch = shifted_batch * range_win_3d
        range_time = mx.fft.ifft(windowed_batch, n=range_fft_size, axis=2) * range_fft_size
        range_time_view = range_time[:, :, :max_view_range_bins] if max_view_range_bins < range_fft_size else range_time
        doppler_windowed = range_time_view * doppler_win_3d
        doppler_fft = mx.fft.fft(doppler_windowed, n=doppler_fft_size, axis=1)
        doppler_shifted = mx.fft.fftshift(doppler_fft, axes=(1,))
        magnitude_db = 20.0 * mx.log10(mx.abs(doppler_shifted) / np.sqrt(raw_rows * raw_cols) + 1e-12)
        magnitude_db_cpu = np.asarray(magnitude_db, dtype=np.float32)
        range_time_cpu = np.asarray(range_time_view, dtype=np.complex64)
        doppler_shifted_cpu = np.asarray(doppler_shifted, dtype=np.complex64)
        for batch_idx, (ch_idx, _) in enumerate(active_items):
            results[ch_idx] = (
                magnitude_db_cpu[batch_idx],
                range_time_cpu[batch_idx],
                doppler_shifted_cpu[batch_idx],
            )
        return results

    range_win = get_range_window(raw_cols) if enable_range_window else np.ones((1, raw_cols), dtype=np.float32)
    doppler_win = get_doppler_window(raw_rows) if enable_doppler_window else np.ones((raw_rows, 1), dtype=np.float32)
    shifted_batch = np.fft.fftshift(raw_batch, axes=2)
    windowed_batch = shifted_batch * range_win.reshape(1, 1, raw_cols)
    padded_data = np.zeros((raw_batch.shape[0], raw_rows, range_fft_size), dtype=np.complex64)
    padded_data[:, :, :raw_cols] = windowed_batch
    range_time = np.fft.ifft(padded_data, axis=2) * range_fft_size
    range_time_view = range_time[:, :, :max_view_range_bins] if max_view_range_bins < range_fft_size else range_time
    padded_doppler = np.zeros((raw_batch.shape[0], doppler_fft_size, range_time_view.shape[2]), dtype=np.complex64)
    padded_doppler[:, :raw_rows, :] = range_time_view * doppler_win.reshape(1, raw_rows, 1)
    doppler_fft = np.fft.fft(padded_doppler, axis=1)
    doppler_shifted = np.fft.fftshift(doppler_fft, axes=1)
    magnitude_db = 20.0 * np.log10(np.abs(doppler_shifted) / np.sqrt(raw_rows * raw_cols) + 1e-12)
    for batch_idx, (ch_idx, _) in enumerate(active_items):
        results[ch_idx] = (
            magnitude_db[batch_idx].astype(np.float32, copy=False),
            range_time_view[batch_idx].astype(np.complex64, copy=False),
            doppler_shifted[batch_idx].astype(np.complex64, copy=False),
        )
    return results


def accumulate_range_time_data_batch(channel_range_time_views):
    updated_channels = []
    for ch_idx, range_time_view in channel_range_time_views:
        if range_time_view is None or getattr(range_time_view, "size", 0) == 0:
            continue
        range_idx = min(selected_range_bin, range_time_view.shape[1] - 1)
        signal_slice = np.asarray(range_time_view[:, range_idx], dtype=np.complex64)
        with CHANNELS[ch_idx].micro_lock:
            CHANNELS[ch_idx].micro_doppler_buffer.extend(signal_slice)
        updated_channels.append(ch_idx)
    return updated_channels


def _batched_micro_doppler_stft(signal_batch, fs=1.0, nperseg=256, noverlap=192, nfft=256):
    signal_batch = np.asarray(signal_batch, dtype=np.complex64)
    if signal_batch.ndim != 2:
        raise ValueError("signal_batch must be 2D: channels x time")
    batch_size, signal_len = signal_batch.shape
    step = nperseg - noverlap
    if step <= 0:
        raise ValueError("STFT step must be positive")
    n_frames = 1 + max(0, (signal_len - nperseg) // step)
    if n_frames <= 0:
        return None, None, None

    window = np.hamming(nperseg).astype(np.float32)
    Zxx = np.empty((batch_size, nfft, n_frames), dtype=np.complex64)
    for frame_idx in range(n_frames):
        start = frame_idx * step
        seg = signal_batch[:, start:start + nperseg]
        if seg.shape[1] < nperseg:
            padded_seg = np.zeros((batch_size, nperseg), dtype=np.complex64)
            padded_seg[:, :seg.shape[1]] = seg
            seg = padded_seg
        buf = np.zeros((batch_size, nfft), dtype=np.complex64)
        buf[:, :nperseg] = seg * window[None, :]
        Zxx[:, :, frame_idx] = np.fft.fft(buf, axis=1)

    f = np.fft.fftfreq(nfft, d=1.0 / fs)
    t = (np.arange(n_frames) * step) / fs
    Pxx_db = 20 * np.log10(np.abs(Zxx) + 1e-12)
    Pxx_db_shifted = np.fft.fftshift(Pxx_db, axes=1)
    f_shifted = np.fft.fftshift(f)
    f_idx = (f_shifted > -0.5) & (f_shifted < 0.5)
    return f_shifted[f_idx], t, Pxx_db_shifted[:, f_idx, :]


def calculate_micro_doppler_batch(channel_indices):
    if not channel_indices:
        return {}

    min_len = None
    for ch_idx in channel_indices:
        with CHANNELS[ch_idx].micro_lock:
            buf_len = len(CHANNELS[ch_idx].micro_doppler_buffer)
        if buf_len < 256:
            continue
        min_len = buf_len if min_len is None else min(min_len, buf_len)

    if min_len is None or min_len < 256:
        return {}

    eligible_channels = []
    signal_batch = []
    for ch_idx in channel_indices:
        with CHANNELS[ch_idx].micro_lock:
            if len(CHANNELS[ch_idx].micro_doppler_buffer) < min_len:
                continue
            signal = np.asarray(list(CHANNELS[ch_idx].micro_doppler_buffer)[-min_len:], dtype=np.complex64)
        eligible_channels.append(ch_idx)
        signal_batch.append(signal)

    if not eligible_channels:
        return {}

    f, t, Pxx_batch = _batched_micro_doppler_stft(np.stack(signal_batch, axis=0))
    if Pxx_batch is None:
        return {}

    x0 = float(t[0]) if len(t) > 0 else 0.0
    x1 = float(t[-1]) if len(t) > 1 else (x0 + 1.0)
    y0 = float(f[0]) if len(f) > 0 else -0.5
    y1 = float(f[-1]) if len(f) > 1 else (y0 + 1.0)
    md_extent = [x0, x1, y0, y1]

    outputs = {}
    for batch_idx, ch_idx in enumerate(eligible_channels):
        outputs[ch_idx] = (f, t, Pxx_batch[batch_idx], md_extent)
    return outputs


def accumulate_range_time_data(ch):
    with ch.range_time_lock:
        data = ch.range_time_data
    updated = accumulate_range_time_data_batch([(ch.ch_id, data)])
    return bool(updated)


def calculate_micro_doppler(ch):
    outputs = calculate_micro_doppler_batch([ch.ch_id])
    if ch.ch_id not in outputs:
        return None, None, None
    f, t, Pxx, _ = outputs[ch.ch_id]
    return f, t, Pxx


def _finalize_channel_result(
    ch,
    frame_id,
    raw_frame,
    viewer_params,
    rd_spectrum,
    range_time_view,
    rd_complex,
    t_start,
    md_result=None,
):
    with ch.range_time_lock:
        ch.range_time_data = range_time_view

    display_range_bins = get_display_range_bin_limit()
    display_doppler_bins = get_display_doppler_bin_limit()

    rd_spectrum_plot, rd_spectrum_cfar, cfar_row_offset, cfar_col_offset = build_cfar_views(
        rd_spectrum, display_range_bins, display_doppler_bins, DISPLAY_DOWNSAMPLE
    )
    rd_complex_plot, _, _, _ = build_cfar_views(
        rd_complex, display_range_bins, display_doppler_bins, DISPLAY_DOWNSAMPLE
    )
    rd_complex_plot = rd_complex_plot.astype(np.complex64, copy=False)

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
    target_clusters = cluster_detected_targets(cfar_points, rd_spectrum_plot) if cfar_points.size > 0 else []

    md_spectrum = None
    md_extent = None
    if show_micro_doppler:
        if md_result is None:
            accumulate_range_time_data(ch)
            f, t, Pxx = calculate_micro_doppler(ch)
            if Pxx is not None:
                md_spectrum = Pxx
                x0 = float(t[0]) if len(t) > 0 else 0.0
                x1 = float(t[-1]) if len(t) > 1 else (x0 + 1.0)
                y0 = float(f[0]) if len(f) > 0 else -0.5
                y1 = float(f[-1]) if len(f) > 1 else (y0 + 1.0)
                md_extent = [x0, x1, y0, y1]
        else:
            _, _, Pxx, md_extent = md_result
            md_spectrum = Pxx

    dsp_time = time.time() - t_start

    result = {
        'ch_id': ch.ch_id,
        'frame_id': int(frame_id),
        'raw': raw_frame,
        'rd': rd_spectrum_plot,
        'rd_complex': rd_complex_plot,
        'cfar_points': cfar_points,
        'cfar_hits': cfar_hits,
        'cfar_shown_hits': cfar_shown_hits,
        'cfar_backend': cfar_backend,
        'cfar_stats': cfar_stats,
        'target_clusters': target_clusters,
        'md': md_spectrum,
        'md_extent': md_extent,
        'dsp_time': dsp_time,
        'viewer_params': viewer_params,
    }

    return result


def _process_one_raw_item(ch, raw_item):
    if isinstance(raw_item, tuple) and len(raw_item) == 2:
        frame_id, raw_frame = raw_item
    else:
        frame_id, raw_frame = -1, raw_item

    t_start = time.time()
    viewer_params = get_viewer_params(ch)
    rd_spectrum, range_time_view, rd_complex = process_range_doppler(raw_frame, viewer_params)
    return _finalize_channel_result(
        ch,
        frame_id,
        raw_frame,
        viewer_params,
        rd_spectrum,
        range_time_view,
        rd_complex,
        t_start,
    )


def dsp_worker():
    global latest_phase_bundle_frame_id, latest_phase_bundle_rd_complex
    print("DSP Worker started (aggregate-frame synchronized)")

    while running:
        latest_bundle = None
        while True:
            try:
                latest_bundle = aggregate_frame_queue.get_nowait()
            except Empty:
                break
            except Exception:
                break

        if latest_bundle is None:
            time.sleep(0.001)
            continue

        frame_id, raw_frames = latest_bundle
        bundle_results = [None] * len(CHANNELS)
        bundle_rd_complex = [None] * len(CHANNELS)
        viewer_params = get_viewer_params(CHANNELS[0]) if CHANNELS else LEGACY_VIEWER_PARAMS
        active_frames = [
            (ch_idx, raw_frame)
            for ch_idx, raw_frame in enumerate(raw_frames)
            if raw_frame is not None and ch_idx < len(CHANNELS)
        ]
        batch_t_start = time.time()

        try:
            batch_outputs = process_range_doppler_batch(
                active_frames,
                viewer_params,
                max_view_range_bins=get_display_range_bin_limit(),
            )
        except Exception as e:
            print(f"[DSP] Batched range-doppler processing error: {e}")
            import traceback
            traceback.print_exc()
            batch_outputs = {}

        md_outputs = {}
        if show_micro_doppler and batch_outputs:
            updated_channels = accumulate_range_time_data_batch(
                [
                    (ch_idx, batch_outputs[ch_idx][1])
                    for ch_idx, _ in active_frames
                    if ch_idx in batch_outputs
                ]
            )
            md_outputs = calculate_micro_doppler_batch(updated_channels)

        for ch_idx, raw_frame in active_frames:
            if ch_idx not in batch_outputs:
                continue
            ch = CHANNELS[ch_idx]
            rd_spectrum, range_time_view, rd_complex = batch_outputs[ch_idx]
            try:
                result = _finalize_channel_result(
                    ch,
                    frame_id,
                    raw_frame,
                    viewer_params,
                    rd_spectrum,
                    range_time_view,
                    rd_complex,
                    batch_t_start,
                    md_result=md_outputs.get(ch_idx),
                )
                bundle_results[ch_idx] = result
                bundle_rd_complex[ch_idx] = result.get('rd_complex')
            except Exception as e:
                print(f"[CH{ch.ch_id + 1}] DSP Worker Error: {e}")
                import traceback
                traceback.print_exc()

        if all(rd is not None for rd in bundle_rd_complex[:len(CHANNELS)]):
            with phase_bundle_lock:
                latest_phase_bundle_frame_id = int(frame_id)
                latest_phase_bundle_rd_complex = bundle_rd_complex

        for ch_idx, result in enumerate(bundle_results):
            if result is None or ch_idx >= len(CHANNELS):
                continue
            _queue_display_frame(CHANNELS[ch_idx], result)


dsp_thread = threading.Thread(target=dsp_worker, daemon=True)
dsp_thread.start()


# ====== Command Functions ======
def send_alignment_command(val):
    target = _current_control_channel_id()
    if target is None:
        return
    # ALGN in backend is applied to the currently selected ALCH target.
    send_control_command(b"ALCH", int(target), target)
    send_control_command(b"ALGN", int(val), target)


def send_strd_command(val):
    try:
        ref_idx = _current_control_channel_id()
        if ref_idx is None:
            return
        strd_limit = get_viewer_params(CHANNELS[ref_idx]).max_strd_value()
        strd_val = max(1, min(strd_limit, int(val)))
        send_shared_control_command(b"STRD", strd_val)
        time.sleep(0.02)
        request_shared_viewer_params()
    except ValueError:
        print(f"Invalid STRD value: {val}")


def send_mti_command(enabled):
    send_shared_control_command(b"MTI ", 1 if enabled else 0)


def send_tx_gain_command(val):
    try:
        gain_db = float(val)
        gain_x10 = int(round(gain_db * 10.0))
        send_shared_control_command(b"TXGN", gain_x10)
        print(f"Requested shared TX gain: {gain_db:.1f} dB")
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
        self._last_known_channel_count = len(CHANNELS)
        self.channel_bias_vector = np.ones(len(CHANNELS), dtype=np.complex128)
        self.phase_calibrated = False
        self.phase_target_range_idx = None
        self.phase_target_doppler_idx = None
        self.phase_calibration_active = False
        self.phase_calibration_samples = []
        self.phase_calibration_frame_ids = []
        self.phase_calibration_range_indices = []
        self.phase_calibration_doppler_bins = []
        self.phase_calibration_target_samples = PHASE_CALIBRATION_TARGET_SAMPLES
        self.phase_calibration_last_frame_id = None
        self.phase_calibration_last_target_desc = "waiting synchronized target"
        self.phase_calibration_last_range_idx = None
        self.phase_calibration_last_doppler_bin = None
        self.phase_calibration_last_saved_path = None
        self.center_freq_hz = 2.4e9
        self.range_scale_sample_rate_hz = None
        self.range_scale_source = "range bin"
        self.target_sector_zero_range_bin = TARGET_SECTOR_DEFAULT_ZERO_RANGE_BIN
        self.target_sector_max_range_bins = TARGET_SECTOR_DEFAULT_MAX_RANGE_BINS
        self.clicked_range_idx = None
        self.clicked_doppler_idx = None
        self.rd_doppler_min = 0.0
        self.rd_doppler_span = 1.0
        self.targets_window = QtWidgets.QWidget()
        self.targets_window.setWindowTitle("OpenISAC Top Targets")
        self.targets_window.resize(760, 720)
        targets_layout = QtWidgets.QVBoxLayout(self.targets_window)
        self.target_sector_plot = pg.PlotWidget(title="Target Sector View")
        self.target_sector_plot.setLabel('left', 'Forward Range (bin)')
        self.target_sector_plot.setLabel('bottom', 'Cross-Range (bin)')
        self.target_sector_plot.showGrid(x=True, y=True, alpha=0.2)
        self.target_sector_plot.setAspectLocked(True)
        self.target_sector_plot.setMouseEnabled(x=False, y=False)
        self.target_sector_plot.setMenuEnabled(False)
        self.target_sector_plot.setMinimumHeight(360)
        self.target_sector_sensor_item = pg.ScatterPlotItem(
            [0.0], [0.0], symbol='t', size=14,
            pen=pg.mkPen('#ffffff', width=1.5),
            brush=pg.mkBrush(255, 255, 255, 220)
        )
        self.target_sector_outline_item = pg.PlotCurveItem(pen=pg.mkPen((140, 160, 180, 180), width=1.4))
        self.target_sector_base_item = pg.PlotCurveItem(pen=pg.mkPen((90, 105, 120, 150), width=1.0))
        self.target_sector_ring_items = [
            pg.PlotCurveItem(pen=pg.mkPen((90, 105, 120, 90), width=1.0))
            for _ in range(TARGET_SECTOR_RANGE_RINGS)
        ]
        self.target_sector_spoke_items = [
            pg.PlotCurveItem(pen=pg.mkPen((90, 105, 120, 110), width=1.0))
            for _ in (-60.0, -30.0, 0.0, 30.0, 60.0)
        ]
        self.target_sector_points_item = pg.ScatterPlotItem(
            [], [], size=14,
            pen=pg.mkPen('#0b132b', width=1.2),
            brush=pg.mkBrush(255, 145, 77, 210)
        )
        self.target_sector_label_items = []
        self.target_sector_plot.addItem(self.target_sector_outline_item)
        self.target_sector_plot.addItem(self.target_sector_base_item)
        for item in self.target_sector_ring_items:
            self.target_sector_plot.addItem(item)
        for item in self.target_sector_spoke_items:
            self.target_sector_plot.addItem(item)
        self.target_sector_plot.addItem(self.target_sector_points_item)
        self.target_sector_plot.addItem(self.target_sector_sensor_item)
        targets_layout.addWidget(self.target_sector_plot)
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
        self.rd_cfar_marker = pg.ScatterPlotItem(
            [], [], symbol='o', size=7,
            pen=pg.mkPen('#00e5ff', width=1.5),
            brush=pg.mkBrush(0, 229, 255, 70)
        )
        self.rd_plot.addItem(self.rd_cfar_marker)
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
        self.phase_curve_plot = pg.PlotWidget(title="Phase-Channel Curve @ Top Target")
        self.phase_curve_plot.setLabel('left', 'Phase (rad, unwrapped)')
        self.phase_curve_plot.setLabel('bottom', 'Channel Index')
        self.phase_curve_plot.showGrid(x=True, y=True, alpha=0.3)
        self.phase_curve_plot.setYRange(-10.0, 10.0, padding=0.0)
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
        self.lbl_cfar = QtWidgets.QLabel("CFAR: off")
        self.lbl_phase_sync = QtWidgets.QLabel("Phase Sync: waiting synchronized aggregate frame")
        self.lbl_phase_clicked = QtWidgets.QLabel("Phase@Clicked: click RD map to query")
        self.lbl_aoa_status = QtWidgets.QLabel("AoA: waiting calibration/click")

        for lbl in [
            self.lbl_display,
            self.lbl_fps,
            self.lbl_queue,
            self.lbl_sender,
            self.lbl_params,
            self.lbl_buffer,
            self.lbl_cfar,
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
            self.lbl_cfar,
            self.lbl_phase_sync,
            self.lbl_phase_clicked,
            self.lbl_aoa_status,
        ]:
            lbl.setFont(fixed_font)
            lbl.setFixedWidth(self._status_label_text_width)
            self._set_status_label_text(lbl, lbl.text())
        self.targets_text.setFont(fixed_font)
        self.targets_text.setPlainText("Top Targets: waiting CFAR detections")
        self._load_range_scale_config()
        self._update_target_sector_background(
            max_range=float(self.target_sector_max_range_bins),
            axis_unit="bin",
            scale_text=self._format_target_sector_scale_text("range bin"),
        )

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
        cfar_misc_layout.addWidget(QtWidgets.QLabel("Min P(dB):"))
        self.txt_cfar_min_power = QtWidgets.QLineEdit(f"{cfar_min_power_db:.1f}")
        cfar_misc_layout.addWidget(self.txt_cfar_min_power)
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

        sector_range_layout = QtWidgets.QHBoxLayout()
        sector_range_layout.addWidget(QtWidgets.QLabel("Sector Zero(bin):"))
        self.txt_sector_zero_bin = QtWidgets.QLineEdit(str(self.target_sector_zero_range_bin))
        sector_range_layout.addWidget(self.txt_sector_zero_bin)
        sector_range_layout.addWidget(QtWidgets.QLabel("Sector Max(bin):"))
        self.txt_sector_max_bin = QtWidgets.QLineEdit(str(self.target_sector_max_range_bins))
        btn_sector_range = QtWidgets.QPushButton("Apply")
        btn_sector_range.clicked.connect(self.set_target_sector_range)
        sector_range_layout.addWidget(self.txt_sector_max_bin)
        sector_range_layout.addWidget(btn_sector_range)
        control_layout.addLayout(sector_range_layout)

        self.btn_calibrate = QtWidgets.QPushButton("Calibrate Phase")
        self.btn_calibrate.clicked.connect(self.calibrate_phase_bias)
        control_layout.addWidget(self.btn_calibrate)
        self._refresh_phase_calibration_button()
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

        self._auto_load_latest_phase_calibration()

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        # Keep the high polling cadence, but skip the body when no new frame arrived.
        self.timer.start(DISPLAY_TIMER_INTERVAL_MS)

        self.last_update_time = time.time()
        self.frame_count = 0
        self.total_dsp_time = 0.0
        self._last_rendered_display_key = None
        self._last_top_targets_key = None
        self._force_ui_refresh = True
        self.targets_window.show()

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
            self._refresh_phase_calibration_button()
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
        self._refresh_phase_calibration_button()
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
        self._force_ui_refresh = True
        self.refresh_display_button_text()
        print(f"Display channel switched to CH{display_channel + 1}")

    def toggle_display_channel(self):
        global display_channel
        if len(CHANNELS) < 2:
            return
        display_channel = (display_channel + 1) % len(CHANNELS)
        self._force_ui_refresh = True
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

    def toggle_cfar(self):
        global cfar_enabled
        cfar_enabled = self.btn_cfar.isChecked()
        self.btn_cfar.setText(f"CA-CFAR: {'ON' if cfar_enabled else 'OFF'}")
        self.btn_cfar.setStyleSheet("QPushButton:checked { background-color: lightgreen; }")

    def apply_cfar_settings(self):
        global cfar_train_doppler, cfar_train_range
        global cfar_guard_doppler, cfar_guard_range
        global cfar_alpha_db, cfar_min_range_bin, cfar_dc_exclusion_bins, cfar_min_power_db
        try:
            cfar_train_doppler = max(0, int(self.txt_cfar_train_d.text()))
            cfar_train_range = max(0, int(self.txt_cfar_train_r.text()))
            cfar_guard_doppler = max(0, int(self.txt_cfar_guard_d.text()))
            cfar_guard_range = max(0, int(self.txt_cfar_guard_r.text()))
            cfar_alpha_db = float(self.txt_cfar_alpha.text())
            cfar_min_range_bin = max(0, int(self.txt_cfar_min_range.text()))
            cfar_dc_exclusion_bins = max(0, int(self.txt_cfar_dc_excl.text()))
            cfar_min_power_db = float(self.txt_cfar_min_power.text())
            alpha_linear = float(np.power(10.0, cfar_alpha_db / 10.0))
            print(
                "CA-CFAR updated: "
                f"train=({cfar_train_doppler},{cfar_train_range}) "
                f"guard=({cfar_guard_doppler},{cfar_guard_range}) "
                f"alpha_db={cfar_alpha_db:.2f} alpha_linear={alpha_linear:.3f} "
                f"min_range={cfar_min_range_bin} min_power_db={cfar_min_power_db:.1f} "
                f"dc_excl={cfar_dc_exclusion_bins}"
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
            self._last_top_targets_key = None
            ch = self._get_display_channel_runtime()
            latest_disp = ch.current_display_data if ch is not None else None
            self.update_top_targets_text(ch, latest_disp)
            self.update_phase_probe_text()
        except ValueError:
            print(f"Invalid center frequency: {self.txt_center_freq_ghz.text()}")
            self.txt_center_freq_ghz.setText(f"{self.center_freq_hz / 1e9:.6f}")

    def set_target_sector_range(self):
        try:
            zero_bin = max(0, int(self.txt_sector_zero_bin.text()))
            max_bin = max(1, int(self.txt_sector_max_bin.text()))
            self.target_sector_zero_range_bin = zero_bin
            self.target_sector_max_range_bins = max_bin
            self.txt_sector_zero_bin.setText(str(self.target_sector_zero_range_bin))
            self.txt_sector_max_bin.setText(str(self.target_sector_max_range_bins))
            print(
                "Target sector range updated: "
                f"zero_range_bin={self.target_sector_zero_range_bin}, "
                f"max_range_bins={self.target_sector_max_range_bins}"
            )
            self._last_top_targets_key = None
            ch = self._get_display_channel_runtime()
            latest_disp = ch.current_display_data if ch is not None else None
            self.update_top_targets_text(ch, latest_disp)
        except ValueError:
            print(
                "Invalid target sector range settings: "
                f"zero={self.txt_sector_zero_bin.text()} max={self.txt_sector_max_bin.text()}"
            )
            self.txt_sector_zero_bin.setText(str(self.target_sector_zero_range_bin))
            self.txt_sector_max_bin.setText(str(self.target_sector_max_range_bins))

    def _refresh_phase_calibration_button(self):
        if not hasattr(self, "btn_calibrate"):
            return

        if len(CHANNELS) < 2:
            self.btn_calibrate.setEnabled(False)
            self.btn_calibrate.setText("Calibrate Phase")
            self.btn_calibrate.setStyleSheet("")
            return

        self.btn_calibrate.setEnabled(True)
        if self.phase_calibration_active:
            count = len(self.phase_calibration_samples)
            self.btn_calibrate.setText(
                f"Stop Calibration ({count}/{self.phase_calibration_target_samples})"
            )
            self.btn_calibrate.setStyleSheet("background-color: khaki;")
        else:
            self.btn_calibrate.setText("Start Phase Calibration")
            self.btn_calibrate.setStyleSheet("")

    def _format_complex_bias_vector(self, bias_vector):
        mags = np.abs(bias_vector)
        phases = np.unwrap(np.angle(bias_vector).astype(np.float64))
        return (
            f"mag={np.array2string(mags, precision=3, separator=',')}, "
            f"phase(rad, CH1-ref)={np.array2string(phases, precision=3, separator=',')}"
        )

    def _normalize_channel_bias_vector(self, bias_vector):
        vec = np.asarray(bias_vector, dtype=np.complex128)
        if vec.ndim != 1 or vec.shape[0] != len(CHANNELS):
            return np.ones(len(CHANNELS), dtype=np.complex128)
        if not np.all(np.isfinite(vec.real)) or not np.all(np.isfinite(vec.imag)):
            return np.ones(len(CHANNELS), dtype=np.complex128)

        ref = vec[0]
        if np.abs(ref) <= 1e-12:
            return np.ones(len(CHANNELS), dtype=np.complex128)

        vec = vec / ref
        vec[0] = 1.0 + 0.0j
        return vec

    def _reset_phase_calibration_collection(self):
        self.phase_calibration_samples = []
        self.phase_calibration_frame_ids = []
        self.phase_calibration_range_indices = []
        self.phase_calibration_doppler_bins = []
        self.phase_calibration_last_frame_id = None
        self.phase_calibration_last_target_desc = "waiting synchronized target"
        self.phase_calibration_last_range_idx = None
        self.phase_calibration_last_doppler_bin = None

    def _save_phase_calibration_data(
        self,
        status,
        samples,
        frame_ids,
        range_indices,
        doppler_bins,
        inlier_mask,
        threshold,
        bias_vector,
    ):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fname = f"./capture/capture_phase_calibration_{status}_{ts}.mat"
        try:
            os.makedirs("./capture", exist_ok=True)
            bias_vec = self._normalize_channel_bias_vector(bias_vector)
            payload = {
                'status': status,
                'channel_bias_vector': np.asarray(bias_vec, dtype=np.complex128),
                'channel_bias_magnitude': np.abs(bias_vec).astype(np.float64, copy=False),
                'channel_bias_phase_rad': np.unwrap(np.angle(bias_vec).astype(np.float64)),
                'calibration_samples': np.asarray(samples, dtype=np.complex128),
                'calibration_frame_ids': np.asarray(frame_ids, dtype=np.int64),
                'calibration_range_indices': np.asarray(range_indices, dtype=np.int32),
                'calibration_doppler_bins': np.asarray(doppler_bins, dtype=np.float64),
                'calibration_inlier_mask': np.asarray(inlier_mask, dtype=bool),
                'calibration_threshold': np.array(float(threshold), dtype=np.float64),
                'calibration_sample_count': np.array(int(len(samples)), dtype=np.int32),
                'calibration_inlier_count': np.array(int(np.count_nonzero(inlier_mask)), dtype=np.int32),
                'center_freq_hz': np.array(float(self.center_freq_hz), dtype=np.float64),
                'display_channel': np.array(int(display_channel), dtype=np.int32),
                'target_sector_zero_range_bin': np.array(int(self.target_sector_zero_range_bin), dtype=np.int32),
                'target_sector_max_range_bins': np.array(int(self.target_sector_max_range_bins), dtype=np.int32),
                'range_scale_sample_rate_hz': np.array(
                    float(self.range_scale_sample_rate_hz) if self.range_scale_sample_rate_hz is not None else np.nan,
                    dtype=np.float64,
                ),
            }
            sio.savemat(fname, payload)
            self.phase_calibration_last_saved_path = fname
            print(f"Saved phase calibration data to {fname}")
        except Exception as exc:
            print(f"Error saving phase calibration data: {exc}")

    def _load_phase_calibration_data_from_file(self, path):
        try:
            data = sio.loadmat(path)
        except Exception as exc:
            print(f"Phase calibration auto-load skipped for {path}: {exc}")
            return False

        bias_raw = data.get('channel_bias_vector')
        if bias_raw is None:
            print(f"Phase calibration auto-load skipped for {path}: missing channel_bias_vector")
            return False

        bias_vec = np.asarray(bias_raw, dtype=np.complex128).ravel()
        if bias_vec.size != len(CHANNELS):
            print(
                "Phase calibration auto-load skipped for "
                f"{path}: channel count mismatch ({bias_vec.size} saved vs {len(CHANNELS)} active)"
            )
            return False
        if not np.all(np.isfinite(bias_vec.real)) or not np.all(np.isfinite(bias_vec.imag)):
            print(f"Phase calibration auto-load skipped for {path}: non-finite bias vector")
            return False

        self.channel_bias_vector = self._normalize_channel_bias_vector(bias_vec)
        self.phase_calibrated = True
        self.phase_calibration_last_saved_path = str(path)

        center_freq_raw = data.get('center_freq_hz')
        if center_freq_raw is not None:
            center_freq_arr = np.asarray(center_freq_raw, dtype=np.float64).ravel()
            if center_freq_arr.size > 0 and np.isfinite(center_freq_arr[0]) and float(center_freq_arr[0]) > 0.0:
                self.center_freq_hz = float(center_freq_arr[0])
                if hasattr(self, "txt_center_freq_ghz"):
                    self.txt_center_freq_ghz.setText(f"{self.center_freq_hz / 1e9:.6f}")

        print(
            "Auto-loaded phase calibration from "
            f"{path}: {self._format_complex_bias_vector(self.channel_bias_vector)}"
        )
        return True

    def _auto_load_latest_phase_calibration(self):
        repo_root = Path(__file__).resolve().parent.parent
        candidate_dirs = []
        for candidate in [Path.cwd() / "capture", repo_root / "capture"]:
            if candidate not in candidate_dirs:
                candidate_dirs.append(candidate)

        calibration_files = []
        for directory in candidate_dirs:
            if not directory.is_dir():
                continue
            calibration_files.extend(directory.glob("capture_phase_calibration_success_*.mat"))

        if not calibration_files:
            print("Phase calibration auto-load: no saved success files found")
            return False

        seen_paths = set()
        unique_files = []
        for path in sorted(calibration_files, key=lambda item: item.stat().st_mtime, reverse=True):
            path_str = str(path.resolve())
            if path_str in seen_paths:
                continue
            seen_paths.add(path_str)
            unique_files.append(path)

        for path in unique_files:
            if self._load_phase_calibration_data_from_file(path):
                return True

        print("Phase calibration auto-load: no compatible saved calibration found")
        return False

    def _load_range_scale_config(self):
        self.range_scale_sample_rate_hz = None
        self.range_scale_source = "range bin"

        repo_root = Path(__file__).resolve().parent.parent
        candidate_paths = []
        for candidate in [
            Path.cwd() / "Modulator.yaml",
            Path.cwd() / "Demodulator.yaml",
            Path.cwd() / "build" / "Modulator.yaml",
            Path.cwd() / "build" / "Demodulator.yaml",
            repo_root / "build" / "Modulator.yaml",
            repo_root / "build" / "Demodulator.yaml",
        ]:
            if candidate not in candidate_paths:
                candidate_paths.append(candidate)

        for path in candidate_paths:
            if not path.is_file():
                continue
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = yaml.safe_load(handle) or {}
            except Exception as exc:
                print(f"Target sector range scale skipped: failed to read {path}: {exc}")
                continue

            sample_rate = data.get("sample_rate")
            if sample_rate is None:
                continue
            try:
                sample_rate_hz = float(sample_rate)
            except (TypeError, ValueError):
                continue
            if sample_rate_hz <= 0.0:
                continue

            self.range_scale_sample_rate_hz = sample_rate_hz
            try:
                self.range_scale_source = str(path.relative_to(repo_root))
            except ValueError:
                self.range_scale_source = str(path)
            print(
                "Target sector range scale: "
                f"using sample_rate={sample_rate_hz:.3f} Hz from {self.range_scale_source}"
            )
            return

        print("Target sector range scale: sample_rate unavailable; falling back to range bins")

    def _resolve_target_range_axis(self, ch):
        if ch is None:
            return 1.0, "bin", "range bin"

        params = get_viewer_params(ch)
        raw_cols = max(1, int(params.active_cols if params.active_cols > 0 else params.wire_cols))
        range_fft_size = max(1, int(params.range_fft_size if params.range_fft_size > 0 else params.wire_cols))

        if self.range_scale_sample_rate_hz is None or self.range_scale_sample_rate_hz <= 0.0:
            return 1.0, "bin", "range bin"

        delay_bin_s = float(raw_cols) / (float(range_fft_size) * float(self.range_scale_sample_rate_hz))
        range_bin_spacing_m = 0.5 * C_LIGHT_MPS * delay_bin_s
        if not np.isfinite(range_bin_spacing_m) or range_bin_spacing_m <= 0.0:
            return 1.0, "bin", "range bin"

        return float(range_bin_spacing_m), "m", f"approx m from {self.range_scale_source}"

    def _format_target_sector_scale_text(self, base_scale_text):
        return (
            f"{base_scale_text}; zero=R{self.target_sector_zero_range_bin}; "
            f"max={self.target_sector_max_range_bins}bin"
        )

    def _target_sector_max_display_range(self, range_scale):
        return max(1.0, float(self.target_sector_max_range_bins) * float(range_scale))

    def _get_target_clusters(self, latest_disp):
        if not latest_disp:
            return []
        clusters = latest_disp.get('target_clusters')
        if clusters is not None:
            return clusters

        rd_data = latest_disp.get('rd')
        cfar_points = latest_disp.get('cfar_points')
        if rd_data is None or cfar_points is None or len(cfar_points) == 0:
            return []

        clusters = cluster_detected_targets(cfar_points, rd_data)
        latest_disp['target_clusters'] = clusters
        return clusters

    def _clear_target_sector_labels(self):
        for item in self.target_sector_label_items:
            self.target_sector_plot.removeItem(item)
        self.target_sector_label_items = []

    def _update_target_sector_background(self, max_range, axis_unit, scale_text):
        max_range = max(1.0, float(max_range))
        theta = np.deg2rad(np.linspace(-TARGET_SECTOR_HALF_ANGLE_DEG, TARGET_SECTOR_HALF_ANGLE_DEG, 181))
        arc_x = max_range * np.sin(theta)
        arc_y = max_range * np.cos(theta)
        outline_x = np.concatenate(([0.0], arc_x, [0.0]))
        outline_y = np.concatenate(([0.0], arc_y, [0.0]))
        self.target_sector_outline_item.setData(outline_x, outline_y)
        self.target_sector_base_item.setData([-max_range, max_range], [0.0, 0.0])

        ring_fracs = np.linspace(1.0 / TARGET_SECTOR_RANGE_RINGS, 1.0, TARGET_SECTOR_RANGE_RINGS)
        for item, frac in zip(self.target_sector_ring_items, ring_fracs):
            ring_range = max_range * float(frac)
            item.setData(ring_range * np.sin(theta), ring_range * np.cos(theta))

        spoke_angles_deg = (-60.0, -30.0, 0.0, 30.0, 60.0)
        for item, angle_deg in zip(self.target_sector_spoke_items, spoke_angles_deg):
            angle_rad = np.deg2rad(angle_deg)
            item.setData(
                [0.0, max_range * np.sin(angle_rad)],
                [0.0, max_range * np.cos(angle_rad)],
            )

        axis_label = f"Range ({axis_unit})"
        self.target_sector_plot.setLabel('left', axis_label)
        self.target_sector_plot.setLabel('bottom', f"Cross-Range ({axis_unit})")
        self.target_sector_plot.setXRange(-1.08 * max_range, 1.08 * max_range, padding=0.0)
        self.target_sector_plot.setYRange(0.0, 1.08 * max_range, padding=0.0)
        self.target_sector_plot.setTitle(f"Target Sector View [{scale_text}]")

    def _update_target_sector_plot(self, entries, axis_unit, scale_text, range_scale):
        self._clear_target_sector_labels()

        max_range = self._target_sector_max_display_range(range_scale)
        self._update_target_sector_background(max_range=max_range, axis_unit=axis_unit, scale_text=scale_text)

        if not entries:
            self.target_sector_points_item.setData([], [])
            return

        valid_entries = [
            entry for entry in entries
            if entry.get('aoa_deg') is not None and float(entry.get('range_value', -1.0)) >= 0.0
        ]
        if not valid_entries:
            self.target_sector_points_item.setData([], [])
            self.target_sector_plot.setTitle(
                f"Target Sector View [{scale_text}; AoA pending or target before zero bin]"
            )
            return

        spots = []
        for entry in valid_entries:
            theta_rad = np.deg2rad(float(entry['aoa_deg']))
            radius = float(entry['range_value'])
            x_pos = radius * np.sin(theta_rad)
            y_pos = max(0.0, radius * np.cos(theta_rad))
            color = '#ff7f50' if int(entry['rank']) == 1 else '#4cc9f0'
            spots.append({
                'pos': (x_pos, y_pos),
                'size': 16 if int(entry['rank']) == 1 else 13,
                'brush': pg.mkBrush(color),
                'pen': pg.mkPen('#0b132b', width=1.2),
            })

            label = pg.TextItem(text=f"{int(entry['rank'])}", color='w', anchor=(0.5, 1.0))
            label.setPos(x_pos, y_pos)
            self.target_sector_plot.addItem(label)
            self.target_sector_label_items.append(label)

        self.target_sector_points_item.setData(spots)

    def _build_top_target_entries(self, ch, latest_disp):
        if ch is None or latest_disp is None:
            return None, "Top Targets: N/A", "bin", "range bin", 1.0

        range_scale, axis_unit, base_scale_text = self._resolve_target_range_axis(ch)
        scale_text = self._format_target_sector_scale_text(base_scale_text)
        clusters = self._get_target_clusters(latest_disp)
        if not clusters:
            return [], "Top Targets: none", axis_unit, scale_text, range_scale

        frame_id = latest_disp.get('frame_id', -1)
        have_auto_aoa = (
            len(CHANNELS) >= 2
            and self.phase_ready
            and self.synced_frame_id is not None
            and int(frame_id) == int(self.synced_frame_id)
        )

        header = f"Top Targets CH{ch.ch_id + 1} F{frame_id} [{scale_text}]"
        lines = [header]
        entries = []
        for rank, cluster in enumerate(clusters[:TARGET_SECTOR_POINT_LIMIT], start=1):
            d_idx = int(cluster['peak_doppler_idx'])
            r_idx = int(cluster['peak_range_idx'])
            strength_db = float(cluster['peak_strength_db'])
            doppler_bin = self.rd_doppler_min + d_idx
            centroid_d = float(cluster['centroid_doppler_idx'])
            centroid_r = float(cluster['centroid_range_idx'])
            relative_range_bins = float(r_idx) - float(self.target_sector_zero_range_bin)
            range_value = relative_range_bins * float(range_scale)
            if axis_unit == "m":
                range_text = f"{range_value:+6.2f}m"
            else:
                range_text = f"{relative_range_bins:+5.0f}bin"

            aoa_deg = None
            aoa_text = "AoA=wait" if len(CHANNELS) >= 2 else "AoA=n/a"
            if have_auto_aoa:
                phase_result = self._compute_phase_vectors_at(d_idx, r_idx)
                if phase_result is not None:
                    _, _, _, _, phase_comp = phase_result
                    aoa_deg, _ = self._estimate_aoa_from_phase(phase_comp)
                    if aoa_deg is not None:
                        aoa_text = f"AoA={aoa_deg:+6.1f}deg"
                    else:
                        aoa_text = "AoA= n/a"

            line = (
                f"{rank}. R={r_idx:4d} (rel={range_text}) D={doppler_bin:+5.0f}"
                f" S={strength_db:6.1f}dB N={int(cluster['cluster_size']):2d}"
                f" C=({centroid_r:5.1f},{centroid_d:5.1f}) {aoa_text}"
            )
            lines.append(line)
            entries.append({
                'rank': rank,
                'range_idx': r_idx,
                'relative_range_bins': relative_range_bins,
                'range_value': range_value,
                'aoa_deg': aoa_deg,
                'strength_db': strength_db,
            })

        return entries, "\n".join(lines), axis_unit, scale_text, range_scale

    def _format_phase_calibration_status(self):
        count = len(self.phase_calibration_samples)
        status = (
            f"AoA: calibrating {count}/{self.phase_calibration_target_samples}; "
            "walk at 0 deg and keep the moving target dominant"
        )
        if self.phase_calibration_last_range_idx is not None and self.phase_calibration_last_doppler_bin is not None:
            status += (
                f"; last {self.phase_calibration_last_target_desc} "
                f"R={self.phase_calibration_last_range_idx},D={self.phase_calibration_last_doppler_bin:+.0f}"
            )
        else:
            status += "; waiting synchronized target"
        if self.phase_calibrated:
            status += " (previous calibration still applied)"
        return status

    def _set_aoa_status_text(self, text):
        if self.phase_calibration_active:
            self._set_status_label_text(self.lbl_aoa_status, self._format_phase_calibration_status())
            return
        self._set_status_label_text(self.lbl_aoa_status, text)

    def _handle_runtime_channel_count_change(self):
        current_count = len(CHANNELS)
        previous_count = self._last_known_channel_count
        self._last_known_channel_count = current_count

        if self.phase_calibration_active:
            self._cancel_phase_calibration(f"channel count changed to {current_count}")

        self.channel_bias_vector = np.ones(current_count, dtype=np.complex128)
        self.phase_calibrated = False
        self.phase_curve_raw = None
        self.phase_curve_comp = None
        self.phase_target_range_idx = None
        self.phase_target_doppler_idx = None
        self._last_rendered_display_key = None
        self._last_top_targets_key = None
        self._force_ui_refresh = True

        print(
            "Viewer detected channel-count change: "
            f"{previous_count} -> {current_count}; attempting to auto-load matching calibration"
        )
        self._auto_load_latest_phase_calibration()

    def _start_phase_calibration(self):
        self.phase_calibration_active = True
        self._reset_phase_calibration_collection()
        self._refresh_phase_calibration_button()
        self._set_aoa_status_text(self._format_phase_calibration_status())
        print(
            "Phase calibration started: "
            f"collecting {self.phase_calibration_target_samples} synchronized complex bias samples while walking at 0 deg"
        )

    def _cancel_phase_calibration(self, reason):
        sample_count = len(self.phase_calibration_samples)
        self.phase_calibration_active = False
        self._reset_phase_calibration_collection()
        self._refresh_phase_calibration_button()
        self._last_top_targets_key = None
        print(f"Phase calibration cancelled: {reason} (collected {sample_count} valid samples)")

    def _estimate_phase_bias_from_samples(self, samples):
        if samples is None or samples.ndim != 2 or samples.shape[0] == 0:
            return None, None, None

        samples = samples.astype(np.complex128, copy=False)
        compare = samples[:, 1:] if samples.shape[1] > 1 else samples
        if compare.shape[1] == 0:
            return None, None, None

        safe_compare = np.where(np.abs(compare) > 1e-12, compare, 1.0 + 0.0j)

        if samples.shape[0] == 1:
            center = samples[0]
        else:
            pairwise_ratio = compare[:, None, :] / safe_compare[None, :, :]
            pairwise_error = np.mean(np.square(np.abs(pairwise_ratio - 1.0)), axis=2)
            center_idx = int(np.argmin(np.median(pairwise_error, axis=1)))
            center = samples[center_idx]

        center_compare = center[1:] if center.shape[0] > 1 else center
        safe_center_compare = np.where(np.abs(center_compare) > 1e-12, center_compare, 1.0 + 0.0j)
        sample_ratio = compare / safe_center_compare[None, :]
        sample_error = np.sqrt(np.mean(np.square(np.abs(sample_ratio - 1.0)), axis=1))

        median_error = float(np.median(sample_error))
        mad_error = float(np.median(np.abs(sample_error - median_error)))
        robust_sigma = max(1.4826 * mad_error, PHASE_CALIBRATION_MIN_ERROR)
        threshold = median_error + PHASE_CALIBRATION_MAD_SCALE * robust_sigma
        inlier_mask = sample_error <= threshold

        min_inliers = max(PHASE_CALIBRATION_MIN_INLIERS, samples.shape[0] // 3)
        if int(np.count_nonzero(inlier_mask)) < min_inliers:
            return None, inlier_mask, threshold

        mean_vector = np.mean(samples[inlier_mask], axis=0)
        if not np.all(np.isfinite(mean_vector.real)) or not np.all(np.isfinite(mean_vector.imag)):
            return None, inlier_mask, threshold
        if np.abs(mean_vector[0]) <= 1e-12:
            return None, inlier_mask, threshold

        bias = self._normalize_channel_bias_vector(mean_vector)
        return bias, inlier_mask, threshold

    def _finish_phase_calibration(self):
        sample_count = len(self.phase_calibration_samples)
        samples = np.asarray(self.phase_calibration_samples, dtype=np.complex128)
        frame_ids = np.asarray(self.phase_calibration_frame_ids, dtype=np.int64)
        range_indices = np.asarray(self.phase_calibration_range_indices, dtype=np.int32)
        doppler_bins = np.asarray(self.phase_calibration_doppler_bins, dtype=np.float64)
        bias, inlier_mask, threshold = self._estimate_phase_bias_from_samples(samples)
        self.phase_calibration_active = False
        self._refresh_phase_calibration_button()

        if bias is None or inlier_mask is None:
            if PHASE_CALIBRATION_AUTO_SAVE and sample_count > 0:
                fail_bias = self.channel_bias_vector if self.phase_calibrated else np.ones(len(CHANNELS), dtype=np.complex128)
                self._save_phase_calibration_data(
                    status="failed",
                    samples=samples,
                    frame_ids=frame_ids,
                    range_indices=range_indices,
                    doppler_bins=doppler_bins,
                    inlier_mask=np.asarray(inlier_mask, dtype=bool) if inlier_mask is not None else np.zeros(sample_count, dtype=bool),
                    threshold=float(threshold) if threshold is not None else np.nan,
                    bias_vector=fail_bias,
                )
            self._reset_phase_calibration_collection()
            print(
                "Phase calibration failed: "
                f"only {sample_count} samples collected but not enough consistent inliers for a stable estimate"
            )
            self.update_phase_probe_text()
            return

        inlier_count = int(np.count_nonzero(inlier_mask))
        outlier_count = int(sample_count - inlier_count)
        self.channel_bias_vector = self._normalize_channel_bias_vector(bias)
        self.phase_calibrated = True
        if PHASE_CALIBRATION_AUTO_SAVE:
            self._save_phase_calibration_data(
                status="success",
                samples=samples,
                frame_ids=frame_ids,
                range_indices=range_indices,
                doppler_bins=doppler_bins,
                inlier_mask=np.asarray(inlier_mask, dtype=bool),
                threshold=float(threshold),
                bias_vector=self.channel_bias_vector,
            )
        self._reset_phase_calibration_collection()
        self._last_top_targets_key = None
        print(
            "Phase calibration finished: "
            f"{inlier_count}/{sample_count} inliers kept, "
            f"{outlier_count} outliers removed, "
            f"threshold={threshold:.3f}, "
            f"bias(CHx/CH1) {self._format_complex_bias_vector(self.channel_bias_vector)}"
        )
        self.update_phase_probe_text()

    def _collect_phase_calibration_sample(self):
        if not self.phase_calibration_active:
            return
        if not self.phase_ready or self.synced_frame_id is None or self.synced_rd_complex is None:
            return

        frame_id = int(self.synced_frame_id)
        if self.phase_calibration_last_frame_id == frame_id:
            return
        self.phase_calibration_last_frame_id = frame_id

        target = self._select_calibration_target()
        if target is None:
            return

        d_idx, r_idx, target_desc = target
        z = np.asarray([rd[d_idx, r_idx] for rd in self.synced_rd_complex], dtype=np.complex128)
        if z.shape[0] != len(CHANNELS):
            return
        if not np.all(np.isfinite(z.real)) or not np.all(np.isfinite(z.imag)):
            return
        if np.abs(z[0]) <= 1e-12:
            return

        doppler_bin = self.rd_doppler_min + int(d_idx)
        sample = z / z[0]
        sample[0] = 1.0 + 0.0j
        self.phase_calibration_samples.append(sample.copy())
        self.phase_calibration_frame_ids.append(frame_id)
        self.phase_calibration_range_indices.append(int(r_idx))
        self.phase_calibration_doppler_bins.append(float(doppler_bin))
        self.phase_calibration_last_target_desc = target_desc
        self.phase_calibration_last_range_idx = int(r_idx)
        self.phase_calibration_last_doppler_bin = float(doppler_bin)
        self._refresh_phase_calibration_button()

        sample_count = len(self.phase_calibration_samples)
        if (
            sample_count == 1
            or sample_count % PHASE_CALIBRATION_PROGRESS_INTERVAL == 0
            or sample_count >= self.phase_calibration_target_samples
        ):
            print(
                "Phase calibration progress: "
                f"{sample_count}/{self.phase_calibration_target_samples} valid samples, "
                f"last {target_desc} at R={r_idx}, D={doppler_bin:+.0f}"
            )

        if sample_count >= self.phase_calibration_target_samples:
            self._finish_phase_calibration()

    def _select_calibration_target(self):
        if not self.phase_ready or self.synced_rd_complex is None:
            return None

        disp_idx = max(0, min(display_channel, len(CHANNELS) - 1))
        synced_rd = self.synced_rd_complex[disp_idx]
        if synced_rd is None or synced_rd.size == 0:
            return None

        ch = self._get_display_channel_runtime()
        latest_disp = ch.current_display_data if ch is not None else {}
        if (
            latest_disp
            and int(latest_disp.get('frame_id', -1)) == int(self.synced_frame_id)
            and latest_disp.get('rd') is not None
        ):
            clusters = self._get_target_clusters(latest_disp)
            if clusters:
                strongest = clusters[0]
                d_idx = int(strongest['peak_doppler_idx'])
                r_idx = int(strongest['peak_range_idx'])
                return d_idx, r_idx, "CFAR strongest target"

        metric = np.abs(np.asarray(synced_rd))
        if metric.size == 0 or not np.isfinite(metric).any():
            return None
        safe_metric = np.where(np.isfinite(metric), metric, -np.inf)
        peak_flat = int(np.argmax(safe_metric))
        d_idx, r_idx = np.unravel_index(peak_flat, safe_metric.shape)
        return int(d_idx), int(r_idx), "global max target"

    def _select_phase_curve_target(self):
        if not self.phase_ready or self.synced_rd_complex is None:
            return None

        ch = self._get_display_channel_runtime()
        latest_disp = ch.current_display_data if ch is not None else {}
        if not latest_disp:
            return None
        if int(latest_disp.get('frame_id', -1)) != int(self.synced_frame_id):
            return None

        clusters = self._get_target_clusters(latest_disp)
        if not clusters:
            return None

        top_target = clusters[0]
        d_idx = int(top_target['peak_doppler_idx'])
        r_idx = int(top_target['peak_range_idx'])
        return d_idx, r_idx, "clustered top target"

    def calibrate_phase_bias(self):
        if len(CHANNELS) < 2:
            self._set_aoa_status_text("AoA: unavailable (need >=2 channels)")
            print("Calibration skipped: need >=2 channels")
            return

        if self.phase_calibration_active:
            self._cancel_phase_calibration("stopped by user")
            self.update_phase_probe_text()
            return

        self._start_phase_calibration()
        self.update_phase_probe_text()

    def _sync_phase_frame(self):
        self.phase_ready = False
        self.synced_frame_id = None
        self.synced_rd_complex = None

        if len(CHANNELS) < 2:
            return

        with phase_bundle_lock:
            synced_frame_id = latest_phase_bundle_frame_id
            synced_rd_complex = latest_phase_bundle_rd_complex

        if synced_frame_id is None or synced_rd_complex is None:
            return

        if len(synced_rd_complex) < len(CHANNELS):
            return

        rd_complex_list = []
        ref_shape = None
        for ch_id in range(len(CHANNELS)):
            rd_complex = synced_rd_complex[ch_id]
            if rd_complex is None:
                return
            if ref_shape is None:
                ref_shape = rd_complex.shape
            elif rd_complex.shape != ref_shape:
                return
            rd_complex_list.append(rd_complex)

        self.phase_ready = True
        self.synced_frame_id = int(synced_frame_id)
        self.synced_rd_complex = rd_complex_list

    def _compute_phase_vectors(self):
        if (
            not self.phase_ready
            or self.synced_rd_complex is None
            or self.clicked_range_idx is None
            or self.clicked_doppler_idx is None
        ):
            return None
        return self._compute_phase_vectors_at(self.clicked_doppler_idx, self.clicked_range_idx)

    def _compute_phase_vectors_at(self, doppler_idx, range_idx):
        if not self.phase_ready or self.synced_rd_complex is None:
            return None

        rd_ref = self.synced_rd_complex[0]
        if rd_ref is None or rd_ref.size == 0:
            return None

        rows, cols = rd_ref.shape
        d_idx = int(np.clip(doppler_idx, 0, rows - 1))
        r_idx = int(np.clip(range_idx, 0, cols - 1))
        doppler_bin = self.rd_doppler_min + d_idx

        z = np.asarray([rd[d_idx, r_idx] for rd in self.synced_rd_complex], dtype=np.complex64)
        phase_rel = np.angle(z * np.conj(z[0]))
        phase_raw = np.unwrap(phase_rel.astype(np.float64))

        if self.channel_bias_vector.shape[0] != len(CHANNELS):
            self.channel_bias_vector = np.ones(len(CHANNELS), dtype=np.complex128)
            self.phase_calibrated = False

        self.channel_bias_vector = self._normalize_channel_bias_vector(self.channel_bias_vector)
        safe_bias = np.where(np.abs(self.channel_bias_vector) > 1e-12, self.channel_bias_vector, 1.0 + 0.0j)
        z_comp = z.astype(np.complex128, copy=False) / safe_bias
        phase_comp_rel = np.angle(z_comp * np.conj(z_comp[0]))
        phase_comp = np.unwrap(phase_comp_rel.astype(np.float64))
        phase_comp[0] = 0.0
        return d_idx, r_idx, doppler_bin, phase_raw, phase_comp

    def _update_phase_curve_plot(self):
        if len(CHANNELS) < 2:
            self.phase_curve_raw_item.setData([], [])
            self.phase_curve_comp_item.setData([], [])
            return
        if self.phase_curve_raw is None:
            self.phase_curve_raw_item.setData([], [])
            self.phase_curve_comp_item.setData([], [])
            self.phase_curve_plot.setYRange(-10.0, 10.0, padding=0.0)
            return

        x = np.arange(1, len(CHANNELS) + 1, dtype=np.float64)
        self.phase_curve_raw_item.setData(x, self.phase_curve_raw)
        if self.phase_calibrated and self.phase_curve_comp is not None:
            self.phase_curve_comp_item.setData(x, self.phase_curve_comp)
        else:
            self.phase_curve_comp_item.setData([], [])
        self.phase_curve_plot.setXRange(0.5, len(CHANNELS) + 0.5, padding=0.0)
        self.phase_curve_plot.setYRange(-10.0, 10.0, padding=0.0)

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

    def update_top_targets_text(self, ch, latest_disp):
        entries, text, axis_unit, scale_text, range_scale = self._build_top_target_entries(ch, latest_disp)
        self.targets_text.setPlainText(text)
        if entries is None:
            self._update_target_sector_plot([], "bin", self._format_target_sector_scale_text("range bin"), 1.0)
            return
        self._update_target_sector_plot(entries, axis_unit, scale_text, range_scale)

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
            if self.phase_calibration_active:
                self._cancel_phase_calibration("need >=2 channels")
            self.phase_ready = False
            self.synced_frame_id = None
            self.synced_rd_complex = None
            self._set_status_label_text(self.lbl_phase_sync, "Phase Sync: unavailable (need >=2 channels)")
            self._set_aoa_status_text("AoA: unavailable (need >=2 channels)")
            return

        frame_tags = ", ".join([f"CH{i + 1}={CHANNELS[i].last_frame_id}" for i in range(len(CHANNELS))])
        if not self.phase_ready or self.synced_frame_id is None or self.synced_rd_complex is None:
            self._set_status_label_text(
                self.lbl_phase_sync,
                f"Phase Sync: waiting synchronized aggregate frame across {len(CHANNELS)} channels ({frame_tags})"
            )
            if self.phase_calibrated:
                self._set_aoa_status_text("AoA: waiting phase sync (calibrated)")
            else:
                self._set_aoa_status_text("AoA: waiting phase sync (not calibrated)")
            return

        self._set_status_label_text(
            self.lbl_phase_sync,
            f"Phase Sync: locked frame {self.synced_frame_id} across {len(CHANNELS)} channels"
        )

    def update_phase_probe_text(self):
        if len(CHANNELS) < 2:
            self.phase_curve_raw = None
            self.phase_curve_comp = None
            self.phase_target_range_idx = None
            self.phase_target_doppler_idx = None
            self._update_phase_curve_plot()
            self._set_status_label_text(self.lbl_phase_clicked, "Phase@Clicked: unavailable (need >=2 channels)")
            self._set_aoa_status_text("AoA: unavailable (need >=2 channels)")
            return

        phase_target = self._select_phase_curve_target()
        if phase_target is None:
            self.phase_curve_raw = None
            self.phase_curve_comp = None
            self.phase_target_range_idx = None
            self.phase_target_doppler_idx = None
            self._update_phase_curve_plot()
            self._set_status_label_text(self.lbl_phase_clicked, "Phase@Clicked: click RD map to query")
            if self.phase_calibrated:
                self._set_aoa_status_text("AoA: calibrated; waiting clustered top target")
            else:
                self._set_aoa_status_text("AoA: not calibrated; waiting clustered top target")
            return

        target_d_idx, target_r_idx, target_desc = phase_target
        phase_result = self._compute_phase_vectors_at(target_d_idx, target_r_idx)
        if phase_result is None:
            self.phase_curve_raw = None
            self.phase_curve_comp = None
            self.phase_target_range_idx = None
            self.phase_target_doppler_idx = None
            self._update_phase_curve_plot()
            self._set_status_label_text(
                self.lbl_phase_clicked,
                "Phase@Clicked: waiting synchronized aggregate frame data"
            )
            if self.phase_calibrated:
                self._set_aoa_status_text("AoA: waiting same-frame phase (calibrated)")
            else:
                self._set_aoa_status_text("AoA: waiting same-frame phase")
            return

        _, r_idx, doppler_bin, phase_raw, phase_comp = phase_result
        self.phase_curve_raw = phase_raw
        self.phase_curve_comp = phase_comp
        self.phase_target_range_idx = r_idx
        self.phase_target_doppler_idx = int(target_d_idx)
        self._update_phase_curve_plot()

        aoa_deg, slope = self._estimate_aoa_from_phase(phase_comp)

        calib_tag = "calibrated" if self.phase_calibrated else "uncalibrated"
        clicked_text = "Phase@Clicked: click RD map to query"
        if self.clicked_range_idx is not None and self.clicked_doppler_idx is not None:
            clicked_phase = self._compute_phase_vectors()
            if clicked_phase is not None:
                _, clicked_r_idx, clicked_doppler_bin, clicked_phase_raw, clicked_phase_comp = clicked_phase
                disp_idx = max(0, min(display_channel, len(CHANNELS) - 1))
                clicked_text = (
                    f"Phase@Clicked R={clicked_r_idx},D={clicked_doppler_bin:+.0f},F={self.synced_frame_id}: "
                    f"CH{disp_idx + 1} raw={float(clicked_phase_raw[disp_idx]):+.3f}, "
                    f"comp={float(clicked_phase_comp[disp_idx]):+.3f} rad"
                )
            else:
                clicked_text = "Phase@Clicked: waiting synchronized aggregate frame data"
        self._set_status_label_text(self.lbl_phase_clicked, clicked_text)

        if aoa_deg is None or slope is None:
            self._set_aoa_status_text(
                f"AoA@Strongest({target_desc}) R={r_idx},D={doppler_bin:+.0f}: estimation unavailable"
            )
            return
        self._set_aoa_status_text(
            f"AoA@Strongest({target_desc}) R={r_idx},D={doppler_bin:+.0f}: {aoa_deg:+.2f} deg, slope={slope:+.4f} rad/ch, "
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
                if 'dsp_time' in latest:
                    self.total_dsp_time += latest['dsp_time']
                updated_any = True

        if len(CHANNELS) != self._last_known_channel_count:
            self._handle_runtime_channel_count_change()
            updated_any = True

        if not updated_any and not self._force_ui_refresh:
            return

        self._sync_phase_frame()
        self._collect_phase_calibration_sample()

        # Update phase status text only (no heavy compute in UI thread).
        self.update_phase_status()

        # Update currently displayed channel
        ch = self._get_display_channel_runtime()
        if ch is None:
            self._set_status_label_text(self.lbl_sender, "Sender: N/A")
            self._set_status_label_text(self.lbl_params, "Params: N/A")
            self._set_status_label_text(self.lbl_buffer, "MD Buffer: N/A")
            self._set_status_label_text(self.lbl_cfar, "CFAR: N/A")
            self.rd_click_marker.setData([], [])
            self.rd_cfar_marker.setData([], [])
            self._last_rendered_display_key = None
            self._last_top_targets_key = None
            self.update_top_targets_text(None, None)
        else:
            latest_disp = ch.current_display_data
            rd_data = latest_disp.get('rd')
            md_data = latest_disp.get('md')
            cfar_points = latest_disp.get('cfar_points')
            frame_id = int(latest_disp.get('frame_id', -1))
            display_render_key = (display_channel, frame_id)
            redraw_display = display_render_key != self._last_rendered_display_key
            top_targets_key = (
                display_channel,
                frame_id,
                int(self.synced_frame_id) if self.synced_frame_id is not None else -1,
                int(bool(self.phase_calibrated)),
                int(round(self.center_freq_hz)),
                int(self.target_sector_zero_range_bin),
                int(self.target_sector_max_range_bins),
            )
            refresh_targets = top_targets_key != self._last_top_targets_key

            if rd_data is not None and redraw_display:
                self.rd_img.setImage(rd_data, autoLevels=False)
                rows, cols = rd_data.shape  # rows=doppler, cols=range
                self.rd_doppler_min = -0.5 * float(rows)
                self.rd_doppler_span = float(rows)
                self.rd_img.setRect(QtCore.QRectF(self.rd_doppler_min, 0.0, self.rd_doppler_span, float(cols)))
                if cfar_points is not None and len(cfar_points) > 0:
                    x_pts = self.rd_doppler_min + cfar_points[:, 0].astype(np.float32)
                    y_pts = cfar_points[:, 1].astype(np.float32)
                    self.rd_cfar_marker.setData(x_pts, y_pts)
                else:
                    self.rd_cfar_marker.setData([], [])
                if self.clicked_range_idx is not None and self.clicked_doppler_idx is not None:
                    d_idx = int(np.clip(self.clicked_doppler_idx, 0, rows - 1))
                    r_idx = int(np.clip(self.clicked_range_idx, 0, cols - 1))
                    doppler_bin = self.rd_doppler_min + d_idx
                    self.rd_click_marker.setData([doppler_bin], [r_idx])
                self._last_rendered_display_key = display_render_key
            if md_data is not None and redraw_display:
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
            cfar_status = f"CFAR(CH{ch.ch_id + 1}): {'on' if cfar_enabled else 'off'}"
            if cfar_enabled:
                cfar_stats = latest_disp.get('cfar_stats') or {}
                cfar_status += (
                    f" raw={int(latest_disp.get('cfar_hits', 0))}"
                    f" shown={int(latest_disp.get('cfar_shown_hits', 0))}"
                    f" backend={latest_disp.get('cfar_backend', 'cpu')}"
                    f" pmin={cfar_stats.get('power_min_db', cfar_min_power_db):.1f}dB"
                    f" invalid={int(cfar_stats.get('invalid_cells', 0))}"
                    f" nonpos={int(cfar_stats.get('nonpositive_cells', 0))}"
                    f" nonfinite={int(cfar_stats.get('nonfinite_cells', 0))}"
                    f" noise=[{cfar_stats.get('noise_min', 0.0):.2e},{cfar_stats.get('noise_max', 0.0):.2e}]"
                )
            cfar_status_short = f"CFAR(CH{ch.ch_id + 1}): {'on' if cfar_enabled else 'off'}"
            if cfar_enabled:
                cfar_stats = latest_disp.get('cfar_stats') or {}
                cfar_status_short += (
                    f" raw={int(latest_disp.get('cfar_hits', 0))}"
                    f" sh={int(latest_disp.get('cfar_shown_hits', 0))}"
                    f" p={cfar_stats.get('power_min_db', cfar_min_power_db):.0f}"
                    f" inv={int(cfar_stats.get('invalid_cells', 0))}"
                )
            self._set_status_label_text(self.lbl_cfar, cfar_status_short)
            self.lbl_cfar.setToolTip(cfar_status)
            if refresh_targets:
                self.update_top_targets_text(ch, latest_disp)
                self._last_top_targets_key = top_targets_key

        self._force_ui_refresh = False

        self.update_phase_probe_text()
        self.refresh_display_button_text()

        # Queue status
        queue_states = " ".join(
            [f"CH{i + 1}:disp={CHANNELS[i].display_queue.qsize()}" for i in range(len(CHANNELS))]
        )
        self._set_status_label_text(
            self.lbl_queue,
            f"Q(agg/disp) AGG:{aggregate_frame_queue.qsize()} {queue_states}"
        )

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
        self.targets_window.close()
        print("Closing application...")
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
