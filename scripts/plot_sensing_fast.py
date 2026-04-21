import argparse
import importlib
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
try:
    from scipy.ndimage import rank_filter as scipy_rank_filter
    HAVE_SCIPY_NDIMAGE = True
except Exception:
    scipy_rank_filter = None
    HAVE_SCIPY_NDIMAGE = False

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
from sensing_detection import (
    CleanParams,
    build_detection_views,
    estimate_clean_padding,
    run_local_psf_clean,
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


def get_array_module(arr):
    module_root = type(arr).__module__.split(".", 1)[0]
    if module_root in {"cupy", "dpnp"}:
        if cp is not None:
            return cp
        return importlib.import_module(module_root)
    return np


def scalar_to_bool(value):
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def scalar_to_int(value):
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def scalar_to_float(value):
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def to_cpu_array(arr, dtype=None):
    host = to_numpy(arr)
    if dtype is None:
        return np.asarray(host)
    return np.asarray(host, dtype=dtype)


_MD_RESULT_UNSET = object()


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
DEFAULT_DISPLAY_DB_MIN = -80.0
DEFAULT_DISPLAY_DB_MAX = 0.0
DISPLAY_RD_DB_MIN = DEFAULT_DISPLAY_DB_MIN
DISPLAY_RD_DB_MAX = DEFAULT_DISPLAY_DB_MAX
DISPLAY_MD_DB_MIN = DEFAULT_DISPLAY_DB_MIN
DISPLAY_MD_DB_MAX = DEFAULT_DISPLAY_DB_MAX

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
cfar_os_rank_percent = 75.0
cfar_os_suppress_doppler = 2
cfar_os_suppress_range = 2
clean_loop_gain = 1.0
clean_max_targets = 64
clean_min_power_db = 10.0
CLEAN_PSF_THRESHOLD_DB = -35.0
CLEAN_MIN_HALF_WIDTH = 3
LOCAL_DETECTOR_CLEAN = "clean"
LOCAL_DETECTOR_OS_CFAR = "os_cfar"
local_detector_mode = LOCAL_DETECTOR_OS_CFAR
DELAY_ESTIMATOR_FFT = "fft"
DELAY_ESTIMATOR_MUSIC = "music"
DELAY_ESTIMATOR_CAPON = "capon"
DELAY_ESTIMATOR_ROOTMUSIC = "rootmusic"
DELAY_ESTIMATOR_ESPRIT = "esprit"
DELAY_ESTIMATOR_CHOICES = (
    DELAY_ESTIMATOR_FFT,
    DELAY_ESTIMATOR_MUSIC,
    DELAY_ESTIMATOR_CAPON,
    DELAY_ESTIMATOR_ROOTMUSIC,
    DELAY_ESTIMATOR_ESPRIT,
)
delay_estimator_mode = DELAY_ESTIMATOR_FFT
superres_peak_rel_threshold_db = -12.0
SUPERRES_DIAGONAL_LOADING = 1e-3
SUPERRES_MIN_SUBARRAY_LEN = 8
SUPERRES_MAX_SUBARRAY_LEN = 128
SUPERRES_MAX_POINTS = 256
SUPERRES_POINT_MAX_ROWS = 96
SUPERRES_POINT_MAX_ROWS_PER_K = 32
target_dbscan_min_samples = 1
PHASE_CALIBRATION_TARGET_SAMPLES = 300
PHASE_CALIBRATION_PROGRESS_INTERVAL = 25
PHASE_CALIBRATION_MAD_SCALE = 3.5
PHASE_CALIBRATION_MIN_ERROR = 0.05
PHASE_CALIBRATION_MIN_INLIERS = 80
PHASE_CALIBRATION_AUTO_SAVE = True
TARGET_SECTOR_HALF_ANGLE_DEG = 90.0
TARGET_SECTOR_POINT_LIMIT = 5
TARGET_TEXT_POINT_LIMIT = 12
TARGET_SECTOR_RANGE_RINGS = 4
TARGET_SECTOR_DEFAULT_ZERO_RANGE_BIN = 0
TARGET_SECTOR_DEFAULT_MAX_RANGE_BINS = 100
DISPLAY_TIMER_INTERVAL_MS = 5

RAW_QUEUE_SIZE = 20
DISPLAY_QUEUE_SIZE = 5
DISPLAY_DOWNSAMPLE = 1
BUFFER_LENGTH = 5000
MICRO_DOPPLER_STFT_NPERSEG = 256
MICRO_DOPPLER_STFT_NOVERLAP = 192
MICRO_DOPPLER_STFT_NFFT = 256
C_LIGHT_MPS = 299792458.0
ANTENNA_SPACING_M = 42.83e-3
AGGREGATE_FRAME_QUEUE_SIZE = 20

selected_range_bin = 0
show_micro_doppler = True
display_channel = 0

running = True
processing_generation = 0


def get_processing_range_fft_size():
    return max(1, int(PROCESS_RANGE_FFT_SIZE))


def get_processing_doppler_fft_size():
    return max(1, int(PROCESS_DOPPLER_FFT_SIZE))


def get_display_range_bin_limit():
    return max(1, int(DISPLAY_RANGE_BIN_LIMIT))


def get_display_doppler_bin_limit():
    return max(1, int(DISPLAY_DOPPLER_BIN_LIMIT))


def sanitize_display_db_range(low_db, high_db):
    low = float(low_db)
    high = float(high_db)
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Display dB range must be finite")
    if high < low:
        low, high = high, low
    if np.isclose(high, low):
        high = low + 1.0
    return float(low), float(high)


def get_delay_doppler_display_db_range():
    return sanitize_display_db_range(DISPLAY_RD_DB_MIN, DISPLAY_RD_DB_MAX)


def get_micro_doppler_display_db_range():
    return sanitize_display_db_range(DISPLAY_MD_DB_MIN, DISPLAY_MD_DB_MAX)


def _hamming_coherent_gain(length):
    length = max(1, int(length))
    cached = _hamming_coherent_gain_cache.get(length)
    if cached is not None:
        return cached
    cached = float(np.mean(np.hamming(length).astype(np.float64, copy=False)))
    _hamming_coherent_gain_cache[length] = cached
    return cached


def get_rd_window_coherent_gain(length, enabled):
    if not enabled:
        return 1.0
    return _hamming_coherent_gain(length)


def get_rd_display_amplitude_norm(raw_rows, raw_cols, range_window_active, doppler_window_active):
    raw_rows = max(1, int(raw_rows))
    raw_cols = max(1, int(raw_cols))
    range_window_gain = get_rd_window_coherent_gain(raw_cols, range_window_active)
    doppler_window_gain = get_rd_window_coherent_gain(raw_rows, doppler_window_active)
    periodogram_norm = np.sqrt(float(raw_rows) * float(raw_cols))
    processing_gain_norm = np.sqrt(float(raw_rows) * float(raw_cols))
    window_gain_norm = range_window_gain * doppler_window_gain
    # Divide by sqrt(M*K): standard periodogram normalization so the noise power stays unchanged.
    # Divide by another sqrt(M*K): remove the coherent processing gain from integrating M symbols and K subcarriers.
    # Finally divide by G_r * G_d: compensate the coherent gain introduced by the range and Doppler windows.
    norm = periodogram_norm * processing_gain_norm * window_gain_norm
    return max(float(norm), 1e-12)


def get_micro_doppler_display_amplitude_norm(viewer_params, nperseg):
    raw_cols = max(1, int(viewer_params.active_cols))
    range_window_active = enable_range_window and not local_clean_disables_windows()
    md_window_gain = _hamming_coherent_gain(nperseg)
    rd_range_window_gain = get_rd_window_coherent_gain(raw_cols, range_window_active)

    # Match the MD STFT amplitude scale to the calibrated RD convention by
    # removing the range coherent gain before the slow-time STFT coherent gain.
    norm = (
        float(raw_cols)
        * rd_range_window_gain
        * float(nperseg)
        * md_window_gain
    )
    return max(float(norm), 1e-12)


def get_local_detector_label():
    if local_detector_mode == LOCAL_DETECTOR_OS_CFAR:
        return "Local OS-CFAR"
    return "Local CLEAN"


def local_clean_disables_windows():
    return (
        delay_estimator_mode == DELAY_ESTIMATOR_FFT
        and local_detector_mode == LOCAL_DETECTOR_CLEAN
    )


def get_delay_estimator_label():
    labels = {
        DELAY_ESTIMATOR_FFT: "FFT",
        DELAY_ESTIMATOR_MUSIC: "MUSIC",
        DELAY_ESTIMATOR_CAPON: "CAPON",
        DELAY_ESTIMATOR_ROOTMUSIC: "ROOTMUSIC",
        DELAY_ESTIMATOR_ESPRIT: "ESPRIT",
    }
    return labels.get(delay_estimator_mode, str(delay_estimator_mode).upper())


def delay_estimator_uses_superres():
    return delay_estimator_mode != DELAY_ESTIMATOR_FFT


def delay_estimator_outputs_spectrum():
    return delay_estimator_mode in {DELAY_ESTIMATOR_MUSIC, DELAY_ESTIMATOR_CAPON}


def next_processing_generation():
    global processing_generation
    processing_generation += 1
    return int(processing_generation)


def get_processing_generation():
    return int(processing_generation)


def get_dbscan_eps_doppler():
    return max(0, int(cfar_os_suppress_doppler)) + 1


def get_dbscan_eps_range():
    return max(0, int(cfar_os_suppress_range)) + 1


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


def _compute_ranked_points_from_mask(mask, rd_db):
    hit_idx = np.argwhere(mask)
    if hit_idx.size == 0:
        return np.empty((0, 2), dtype=np.int32), 0, 0

    raw_hit_count = int(hit_idx.shape[0])
    values = rd_db[hit_idx[:, 0], hit_idx[:, 1]]
    order = np.argsort(values)[::-1]
    hit_idx = hit_idx[order]
    if hit_idx.shape[0] > cfar_max_points:
        hit_idx = hit_idx[:cfar_max_points]
    shown_hit_count = int(hit_idx.shape[0])
    return hit_idx.astype(np.int32, copy=False), raw_hit_count, shown_hit_count


def run_os_cfar_2d(
    rd_db,
    active_row_start=None,
    active_row_stop=None,
    active_col_start=None,
    active_col_stop=None,
    dc_center_row=None,
):
    rd_db = to_cpu_array(rd_db, dtype=np.float32)
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
            'power_min_db': min_power_db,
            'invalid_cells': 0,
            'nonfinite_cells': 0,
            'nonpositive_cells': 0,
            'os_rank_index': 0,
            'training_cells': 0,
            'rank_percent': float(cfar_os_rank_percent),
        }

    row_start = outer_h if active_row_start is None else max(outer_h, int(active_row_start))
    row_stop = (rows - outer_h) if active_row_stop is None else min(rows - outer_h, int(active_row_stop))
    col_start = outer_w if active_col_start is None else max(outer_w, int(active_col_start))
    col_stop = (cols - outer_w) if active_col_stop is None else min(cols - outer_w, int(active_col_stop))
    if row_start >= row_stop or col_start >= col_stop:
        return np.empty((0, 2), dtype=np.int32), 0, 0, "os-cfar", {
            'noise_min': 0.0,
            'noise_max': 0.0,
            'thresh_min': 0.0,
            'thresh_max': 0.0,
            'power_min_db': min_power_db,
            'invalid_cells': 0,
            'nonfinite_cells': 0,
            'nonpositive_cells': 0,
            'os_rank_index': 0,
            'training_cells': 0,
            'rank_percent': float(cfar_os_rank_percent),
        }

    power = np.power(np.float64(10.0), rd_db.astype(np.float64) / np.float64(10.0), dtype=np.float64)
    footprint = np.ones((2 * outer_h + 1, 2 * outer_w + 1), dtype=bool)
    footprint[outer_h - gd:outer_h + gd + 1, outer_w - gr:outer_w + gr + 1] = False
    training_cells = int(np.count_nonzero(footprint))
    if training_cells <= 0:
        return np.empty((0, 2), dtype=np.int32), 0, 0, "os-cfar", {
            'noise_min': 0.0,
            'noise_max': 0.0,
            'thresh_min': 0.0,
            'thresh_max': 0.0,
            'power_min_db': min_power_db,
            'invalid_cells': 0,
            'nonfinite_cells': 0,
            'nonpositive_cells': 0,
            'os_rank_index': 0,
            'training_cells': 0,
            'rank_percent': float(cfar_os_rank_percent),
        }

    rank_index = int(round((np.clip(float(cfar_os_rank_percent), 0.0, 100.0) / 100.0) * (training_cells - 1)))
    active_rows = row_stop - row_start
    active_cols = col_stop - col_start
    active_rd = rd_db[row_start:row_stop, col_start:col_stop]
    active_power = power[row_start:row_stop, col_start:col_stop]

    candidate_valid = np.ones((active_rows, active_cols), dtype=bool)
    if min_range > 0:
        candidate_valid[:, :min(active_cols, min_range)] = False

    center_row = (rows // 2) if dc_center_row is None else int(dc_center_row)
    center_row_local = center_row - row_start
    if dc_excl > 0 and 0 <= center_row_local < active_rows:
        lo = max(0, center_row_local - dc_excl)
        hi = min(active_rows, center_row_local + dc_excl + 1)
        candidate_valid[lo:hi, :] = False

    if not np.any(candidate_valid):
        return np.empty((0, 2), dtype=np.int32), 0, 0, "os-cfar", {
            'noise_min': 0.0,
            'noise_max': 0.0,
            'thresh_min': 0.0,
            'thresh_max': 0.0,
            'power_min_db': min_power_db,
            'invalid_cells': 0,
            'nonfinite_cells': 0,
            'nonpositive_cells': 0,
            'os_rank_index': int(rank_index),
            'training_cells': int(training_cells),
            'rank_percent': float(cfar_os_rank_percent),
            'evaluated_candidates': 0,
        }

    candidate_scores = np.where(candidate_valid, active_rd, np.float32(-np.inf)).astype(np.float32, copy=False)
    candidate_total = int(np.count_nonzero(np.isfinite(candidate_scores)))
    candidate_limit = min(candidate_total, max(int(cfar_max_points) * 32, 1024))
    if candidate_limit <= 0:
        return np.empty((0, 2), dtype=np.int32), 0, 0, "os-cfar", {
            'noise_min': 0.0,
            'noise_max': 0.0,
            'thresh_min': 0.0,
            'thresh_max': 0.0,
            'power_min_db': min_power_db,
            'invalid_cells': 0,
            'nonfinite_cells': 0,
            'nonpositive_cells': 0,
            'os_rank_index': int(rank_index),
            'training_cells': int(training_cells),
            'rank_percent': float(cfar_os_rank_percent),
            'evaluated_candidates': 0,
        }

    flat_scores = candidate_scores.reshape(-1)
    partition_k = max(0, flat_scores.size - candidate_limit)
    candidate_flat_idx = np.argpartition(flat_scores, partition_k)[partition_k:]
    candidate_flat_idx = candidate_flat_idx[np.argsort(flat_scores[candidate_flat_idx])[::-1]]

    accepted_points = []
    noise_vals = []
    thresh_vals = []
    invalid_cells = 0
    nonfinite_cells = 0
    nonpositive_cells = 0
    suppressed = np.zeros((active_rows, active_cols), dtype=bool)
    suppression_d = max(0, int(cfar_os_suppress_doppler))
    suppression_r = max(0, int(cfar_os_suppress_range))

    for flat_idx in candidate_flat_idx.tolist():
        local_row = int(flat_idx) // active_cols
        local_col = int(flat_idx) % active_cols
        if suppressed[local_row, local_col]:
            continue

        cut_db = float(active_rd[local_row, local_col])
        if not np.isfinite(cut_db) or cut_db < min_power_db:
            continue

        row = row_start + local_row
        col = col_start + local_col
        window = power[row - outer_h:row + outer_h + 1, col - outer_w:col + outer_w + 1]
        values = window[footprint]
        values = values[np.isfinite(values)]
        if values.size == 0:
            nonfinite_cells += 1
            invalid_cells += 1
            continue

        local_rank_index = min(rank_index, values.size - 1)
        noise_order = float(np.partition(values, local_rank_index)[local_rank_index])
        if not np.isfinite(noise_order):
            nonfinite_cells += 1
            invalid_cells += 1
            continue
        if noise_order <= eps:
            nonpositive_cells += 1
            invalid_cells += 1
            continue

        threshold = float(alpha * noise_order)
        noise_vals.append(noise_order)
        thresh_vals.append(threshold)
        if float(active_power[local_row, local_col]) <= threshold:
            continue

        accepted_points.append((row, col))
        lo_r = max(0, local_row - suppression_d)
        hi_r = min(active_rows, local_row + suppression_d + 1)
        lo_c = max(0, local_col - suppression_r)
        hi_c = min(active_cols, local_col + suppression_r + 1)
        suppressed[lo_r:hi_r, lo_c:hi_c] = True

    points = np.asarray(accepted_points, dtype=np.int32) if accepted_points else np.empty((0, 2), dtype=np.int32)
    raw_hits = int(points.shape[0])
    if points.shape[0] > int(cfar_max_points):
        values = rd_db[points[:, 0], points[:, 1]]
        order = np.argsort(values)[::-1][:int(cfar_max_points)]
        points = points[order]
    shown_hits = int(points.shape[0])

    stats = {
        'noise_min': float(np.min(noise_vals)) if noise_vals else 0.0,
        'noise_max': float(np.max(noise_vals)) if noise_vals else 0.0,
        'thresh_min': float(np.min(thresh_vals)) if thresh_vals else 0.0,
        'thresh_max': float(np.max(thresh_vals)) if thresh_vals else 0.0,
        'power_min_db': min_power_db,
        'invalid_cells': invalid_cells,
        'nonfinite_cells': nonfinite_cells,
        'nonpositive_cells': nonpositive_cells,
        'os_rank_index': int(rank_index),
        'training_cells': int(training_cells),
        'rank_percent': float(cfar_os_rank_percent),
        'suppress_d': int(cfar_os_suppress_doppler),
        'suppress_r': int(cfar_os_suppress_range),
        'evaluated_candidates': int(candidate_limit),
    }
    return points, raw_hits, shown_hits, "os-cfar", stats


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
    eps_d = get_dbscan_eps_doppler()
    eps_r = get_dbscan_eps_range()
    min_samples = max(1, int(target_dbscan_min_samples))

    diff_d = np.abs(points[:, None, 0] - points[None, :, 0]) <= eps_d
    diff_r = np.abs(points[:, None, 1] - points[None, :, 1]) <= eps_r
    neighbor_mask = diff_d & diff_r
    neighbor_lists = [np.flatnonzero(neighbor_mask[idx]) for idx in range(n_points)]

    labels = np.full(n_points, -2, dtype=np.int32)  # -2=unvisited, -1=noise
    cluster_id = 0

    for seed_idx in range(n_points):
        if labels[seed_idx] != -2:
            continue

        seed_neighbors = neighbor_lists[seed_idx]
        if seed_neighbors.size < min_samples:
            labels[seed_idx] = -1
            continue

        labels[seed_idx] = cluster_id
        queue = list(seed_neighbors.tolist())
        queue_pos = 0
        while queue_pos < len(queue):
            nbr_idx = int(queue[queue_pos])
            queue_pos += 1
            if labels[nbr_idx] == -1:
                labels[nbr_idx] = cluster_id
            if labels[nbr_idx] != -2:
                continue
            labels[nbr_idx] = cluster_id
            nbr_neighbors = neighbor_lists[nbr_idx]
            if nbr_neighbors.size >= min_samples:
                queue.extend(nbr_neighbors.tolist())
        cluster_id += 1

    clusters = []
    for cur_cluster_id in range(cluster_id):
        cluster_indices = np.flatnonzero(labels == cur_cluster_id)
        if cluster_indices.size == 0:
            continue

        cluster_points = points[cluster_indices]
        if point_strengths_db is not None:
            strengths_db = point_strengths_db[cluster_indices]
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


def build_direct_targets(points, rd_data, point_strengths_db=None):
    if points is None:
        return []

    points = np.asarray(points, dtype=np.int32)
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
    if points.size == 0:
        return []

    if point_strengths_db is not None:
        point_strengths_db = np.asarray(point_strengths_db, dtype=np.float32)
        if point_strengths_db.shape[0] == in_bounds.shape[0]:
            strengths_db = point_strengths_db[in_bounds]
        else:
            strengths_db = rd_data[points[:, 0], points[:, 1]]
    else:
        strengths_db = rd_data[points[:, 0], points[:, 1]]

    order = np.argsort(strengths_db)[::-1]
    clusters = []
    for rank_idx in order:
        point = points[int(rank_idx)]
        strength_db = float(strengths_db[int(rank_idx)])
        clusters.append({
            'peak_doppler_idx': int(point[0]),
            'peak_range_idx': int(point[1]),
            'peak_strength_db': strength_db,
            'cluster_size': 1,
            'centroid_doppler_idx': float(point[0]),
            'centroid_range_idx': float(point[1]),
        })
    return clusters


def reset_processing_state():
    global latest_phase_bundle_frame_id, latest_phase_bundle_rd_complex
    next_processing_generation()
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
    parser.add_argument(
        "--display-db-min",
        type=float,
        default=DEFAULT_DISPLAY_DB_MIN,
        help="Shared default display color scale minimum in dB",
    )
    parser.add_argument(
        "--display-db-max",
        type=float,
        default=DEFAULT_DISPLAY_DB_MAX,
        help="Shared default display color scale maximum in dB",
    )
    parser.add_argument(
        "--delay-doppler-db-min",
        type=float,
        default=None,
        help="Delay-Doppler display color scale minimum in dB",
    )
    parser.add_argument(
        "--delay-doppler-db-max",
        type=float,
        default=None,
        help="Delay-Doppler display color scale maximum in dB",
    )
    parser.add_argument(
        "--micro-doppler-db-min",
        type=float,
        default=None,
        help="Micro-Doppler display color scale minimum in dB",
    )
    parser.add_argument(
        "--micro-doppler-db-max",
        type=float,
        default=None,
        help="Micro-Doppler display color scale maximum in dB",
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
        self.metadata_frame_buffer = FrameBuffer()
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
shared_display_db_min, shared_display_db_max = sanitize_display_db_range(
    args.display_db_min,
    args.display_db_max,
)
DISPLAY_RD_DB_MIN, DISPLAY_RD_DB_MAX = sanitize_display_db_range(
    args.delay_doppler_db_min if args.delay_doppler_db_min is not None else shared_display_db_min,
    args.delay_doppler_db_max if args.delay_doppler_db_max is not None else shared_display_db_max,
)
DISPLAY_MD_DB_MIN, DISPLAY_MD_DB_MAX = sanitize_display_db_range(
    args.micro_doppler_db_min if args.micro_doppler_db_min is not None else shared_display_db_min,
    args.micro_doppler_db_max if args.micro_doppler_db_max is not None else shared_display_db_max,
)
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


def backend_stream_unsupported(viewer_params):
    return viewer_params.backend_processing() or viewer_params.metadata_sidecar()


def warn_unsupported_backend_params(prefix, viewer_params):
    print(
        f"{prefix} Warning: fast sensing viewer does not support backend sensing output; "
        f"ignoring backend stream until sender returns to local/raw mode. "
        f"Use plot_backend_sensing.py instead. Params: {viewer_params.describe()}"
    )


def warn_unsupported_backend_payload(prefix, kind, frame_id=None):
    frame_text = "" if frame_id is None else f" frame {int(frame_id)}"
    print(
        f"{prefix} Warning: fast sensing viewer does not support backend sensing output; "
        f"dropping backend {kind}{frame_text}. Use plot_backend_sensing.py instead."
    )


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
    time.sleep(0.05)
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
    if backend_stream_unsupported(params):
        warn_unsupported_backend_params(prefix, params)


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
            is_metadata = bool(total_chunks & 0x80000000)
            total_chunks &= 0x7FFFFFFF

            chunk_data = data[HEADER_SIZE:]
            frame_buffer = CHANNELS[0].metadata_frame_buffer if is_metadata else CHANNELS[0].frame_buffer
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
                if len(payload) >= 4 and payload[:4] == b"ASM1":
                    warn_unsupported_backend_payload("[AGG]", "metadata sidecar", frame_id_done)
                    continue
                if backend_stream_unsupported(viewer_params):
                    warn_unsupported_backend_payload("[AGG]", "aggregate sensing frame", frame_id_done)
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
_hamming_coherent_gain_cache = {}


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
    if backend_stream_unsupported(viewer_params):
        raise ValueError("Backend sensing output is not supported in plot_sensing_fast.py")

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
    range_window_active = enable_range_window and not local_clean_disables_windows()
    doppler_window_active = enable_doppler_window and not local_clean_disables_windows()
    rd_display_norm = get_rd_display_amplitude_norm(
        raw_rows,
        raw_cols,
        range_window_active,
        doppler_window_active,
    )

    if USE_NVIDIA_GPU or USE_INTEL_GPU:
        range_win = get_range_window(raw_cols) if range_window_active else cp.ones((1, raw_cols), dtype=cp.float32)
        doppler_win = get_doppler_window(raw_rows) if doppler_window_active else cp.ones((raw_rows, 1), dtype=cp.float32)
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

        magnitude = cp.abs(doppler_shifted) / rd_display_norm
        magnitude_db = 20.0 * cp.log10(magnitude + 1e-12)
        return (
            magnitude_db.astype(cp.float32, copy=False),
            range_time_view.astype(cp.complex64, copy=False),
            doppler_shifted.astype(cp.complex64, copy=False),
        )

    if USE_APPLE_GPU:
        range_win = get_range_window(raw_cols) if range_window_active else mx.ones((1, raw_cols), dtype=mx.float32)
        doppler_win = get_doppler_window(raw_rows) if doppler_window_active else mx.ones((raw_rows, 1), dtype=mx.float32)
        frame_data_gpu = mx.array(raw_frame, dtype=mx.complex64)
        shifted_data = mx.fft.fftshift(frame_data_gpu, axes=(1,))
        windowed_data = shifted_data * range_win
        range_time = mx.fft.ifft(windowed_data, n=range_fft_size, axis=1) * range_fft_size

        range_time_view = range_time[:, :max_view_range_bins] if max_view_range_bins < range_fft_size else range_time
        doppler_windowed = range_time_view * doppler_win
        doppler_fft = mx.fft.fft(doppler_windowed, n=doppler_fft_size, axis=0)
        doppler_shifted = mx.fft.fftshift(doppler_fft, axes=(0,))

        magnitude = mx.abs(doppler_shifted) / rd_display_norm
        magnitude_db = 20.0 * mx.log10(magnitude + 1e-12)
        return (
            np.array(magnitude_db, copy=False).astype(np.float32, copy=False),
            np.array(range_time_view, copy=False),
            np.array(doppler_shifted, copy=False).astype(np.complex64, copy=False),
        )

    range_win = get_range_window(raw_cols) if range_window_active else np.ones((1, raw_cols), dtype=np.float32)
    doppler_win = get_doppler_window(raw_rows) if doppler_window_active else np.ones((raw_rows, 1), dtype=np.float32)

    if HAVE_NUMBA:
        padded_data = cpu_prep_range_fft(raw_frame, range_win, raw_cols, range_fft_size)
        range_time = np.fft.ifft(padded_data, axis=1) * range_fft_size
        range_time_view = range_time[:, :max_view_range_bins] if max_view_range_bins < range_fft_size else range_time
        view_width = range_time_view.shape[1]

        padded_doppler = cpu_prep_doppler_fft(range_time_view, doppler_win, raw_rows, doppler_fft_size, view_width)
        doppler_fft = np.fft.fft(padded_doppler, axis=0)
        doppler_shifted = np.fft.fftshift(doppler_fft, axes=0)
        magnitude_db = 20.0 * np.log10(np.abs(doppler_shifted) / rd_display_norm + 1e-12)
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

    magnitude_db = 20.0 * np.log10(np.abs(doppler_shifted) / rd_display_norm + 1e-12)
    return magnitude_db.astype(np.float32, copy=False), range_time_view, doppler_shifted.astype(np.complex64, copy=False)


def process_range_doppler_batch(channel_frames, viewer_params, max_view_range_bins=None):
    if backend_stream_unsupported(viewer_params):
        raise ValueError("Backend sensing output is not supported in plot_sensing_fast.py")

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
    range_window_active = enable_range_window and not local_clean_disables_windows()
    doppler_window_active = enable_doppler_window and not local_clean_disables_windows()
    rd_display_norm = get_rd_display_amplitude_norm(
        raw_rows,
        raw_cols,
        range_window_active,
        doppler_window_active,
    )

    raw_batch = np.stack(
        [np.asarray(frame[:raw_rows, :raw_cols], dtype=np.complex64) for _, frame in active_items],
        axis=0,
    )

    if USE_NVIDIA_GPU or USE_INTEL_GPU:
        range_win = get_range_window(raw_cols) if range_window_active else cp.ones((1, raw_cols), dtype=cp.float32)
        doppler_win = get_doppler_window(raw_rows) if doppler_window_active else cp.ones((raw_rows, 1), dtype=cp.float32)
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

        magnitude_db = 20.0 * cp.log10(cp.abs(doppler_shifted) / rd_display_norm + 1e-12)
        for batch_idx, (ch_idx, _) in enumerate(active_items):
            results[ch_idx] = (
                magnitude_db[batch_idx].astype(cp.float32, copy=False),
                range_time_view[batch_idx].astype(cp.complex64, copy=False),
                doppler_shifted[batch_idx].astype(cp.complex64, copy=False),
            )
        return results

    if USE_APPLE_GPU:
        range_win = get_range_window(raw_cols) if range_window_active else mx.ones((1, raw_cols), dtype=mx.float32)
        doppler_win = get_doppler_window(raw_rows) if doppler_window_active else mx.ones((raw_rows, 1), dtype=mx.float32)
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
        magnitude_db = 20.0 * mx.log10(mx.abs(doppler_shifted) / rd_display_norm + 1e-12)
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

    range_win = get_range_window(raw_cols) if range_window_active else np.ones((1, raw_cols), dtype=np.float32)
    doppler_win = get_doppler_window(raw_rows) if doppler_window_active else np.ones((raw_rows, 1), dtype=np.float32)
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
    magnitude_db = 20.0 * np.log10(np.abs(doppler_shifted) / rd_display_norm + 1e-12)
    for batch_idx, (ch_idx, _) in enumerate(active_items):
        results[ch_idx] = (
            magnitude_db[batch_idx].astype(np.float32, copy=False),
            range_time_view[batch_idx].astype(np.complex64, copy=False),
            doppler_shifted[batch_idx].astype(np.complex64, copy=False),
        )
    return results


def _display_doppler_row_bounds(total_rows, display_doppler_bins, downsample=1):
    total_rows = max(0, int(total_rows))
    if total_rows <= 0:
        return 0, 0, max(1, int(downsample))
    ds = max(1, int(downsample))
    visible_rows = max(1, int(display_doppler_bins))
    center_idx = total_rows // 2
    row_start = max(0, center_idx - visible_rows // 2)
    row_stop = min(total_rows, row_start + visible_rows)
    return row_start, row_stop, ds


def process_doppler_frequency_input(frame_data, viewer_params):
    if backend_stream_unsupported(viewer_params):
        raise ValueError("Backend sensing output is not supported in plot_sensing_fast.py")
    if viewer_params.is_dense_range_doppler():
        return None
    if not viewer_params.raw_fft_locally_supported():
        raise ValueError(f"Unsupported sensing frame format: {viewer_params.describe()}")

    raw_rows = max(1, int(viewer_params.active_rows))
    raw_cols = max(1, int(viewer_params.active_cols))
    doppler_fft_size = max(raw_rows, get_processing_doppler_fft_size())
    raw_frame = np.asarray(frame_data[:raw_rows, :raw_cols], dtype=np.complex64)
    range_window_active = enable_range_window and not local_clean_disables_windows()
    doppler_window_active = enable_doppler_window and not local_clean_disables_windows()

    if USE_NVIDIA_GPU or USE_INTEL_GPU:
        range_win = get_range_window(raw_cols) if range_window_active else cp.ones((1, raw_cols), dtype=cp.float32)
        doppler_win = get_doppler_window(raw_rows) if doppler_window_active else cp.ones((raw_rows, 1), dtype=cp.float32)
        frame_gpu = cp.asarray(raw_frame, dtype=cp.complex64)
        shifted_data = cp.fft.fftshift(frame_gpu, axes=1)
        freq_windowed = shifted_data * range_win
        doppler_input = freq_windowed * doppler_win
        doppler_shifted = cp.fft.fftshift(cp.fft.fft(doppler_input, n=doppler_fft_size, axis=0), axes=0)
        if USE_NVIDIA_GPU:
            return doppler_shifted.astype(cp.complex64, copy=False)
        return to_cpu_array(doppler_shifted, dtype=np.complex64)

    if USE_APPLE_GPU:
        range_win = get_range_window(raw_cols) if range_window_active else mx.ones((1, raw_cols), dtype=mx.float32)
        doppler_win = get_doppler_window(raw_rows) if doppler_window_active else mx.ones((raw_rows, 1), dtype=mx.float32)
        frame_gpu = mx.array(raw_frame, dtype=mx.complex64)
        shifted_data = mx.fft.fftshift(frame_gpu, axes=(1,))
        freq_windowed = shifted_data * range_win
        doppler_input = freq_windowed * doppler_win
        doppler_shifted = mx.fft.fftshift(mx.fft.fft(doppler_input, n=doppler_fft_size, axis=0), axes=(0,))
        return np.asarray(doppler_shifted, dtype=np.complex64)

    range_win = get_range_window(raw_cols) if range_window_active else np.ones((1, raw_cols), dtype=np.float32)
    doppler_win = get_doppler_window(raw_rows) if doppler_window_active else np.ones((raw_rows, 1), dtype=np.float32)
    shifted_data = np.fft.fftshift(raw_frame, axes=1)
    freq_windowed = shifted_data * range_win
    doppler_input = freq_windowed * doppler_win
    doppler_shifted = np.fft.fftshift(np.fft.fft(doppler_input, n=doppler_fft_size, axis=0), axes=0)
    return doppler_shifted.astype(np.complex64, copy=False)


def _build_local_max_mask_2d(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        return np.zeros_like(values, dtype=bool)

    rows, cols = values.shape
    maxima = np.isfinite(values)
    strict = np.zeros((rows, cols), dtype=bool)
    neg_inf = float("-inf")
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            shifted = np.full((rows, cols), neg_inf, dtype=np.float64)
            src_r0 = max(0, -dr)
            src_r1 = min(rows, rows - dr) if dr >= 0 else rows
            src_c0 = max(0, -dc)
            src_c1 = min(cols, cols - dc) if dc >= 0 else cols
            dst_r0 = max(0, dr)
            dst_r1 = dst_r0 + (src_r1 - src_r0)
            dst_c0 = max(0, dc)
            dst_c1 = dst_c0 + (src_c1 - src_c0)
            if src_r1 > src_r0 and src_c1 > src_c0:
                shifted[dst_r0:dst_r1, dst_c0:dst_c1] = values[src_r0:src_r1, src_c0:src_c1]
            maxima &= values >= shifted
            strict |= values > shifted
    return maxima & strict


def _is_cupy_array(arr):
    if cp is None:
        return False
    return isinstance(arr, cp.ndarray)


def _select_superres_subarray_length(vector_len):
    vector_len = max(0, int(vector_len))
    if vector_len < SUPERRES_MIN_SUBARRAY_LEN:
        return 0
    upper = min(SUPERRES_MAX_SUBARRAY_LEN, vector_len - 1)
    if upper < SUPERRES_MIN_SUBARRAY_LEN:
        return 0
    return int(np.clip(vector_len // 2, SUPERRES_MIN_SUBARRAY_LEN, upper))


def _build_spatially_smoothed_covariance(row_vector):
    row_vector = np.asarray(row_vector, dtype=np.complex128).reshape(-1)
    vector_len = int(row_vector.size)
    subarray_len = _select_superres_subarray_length(vector_len)
    if subarray_len <= 0:
        return None

    subarrays = np.lib.stride_tricks.sliding_window_view(row_vector, subarray_len)
    snapshot_count = int(subarrays.shape[0])
    if snapshot_count < 2:
        return None

    covariance_f = (subarrays.T @ np.conjugate(subarrays)) / float(snapshot_count)
    if not np.all(np.isfinite(covariance_f)):
        return None

    reversal = np.eye(subarray_len, dtype=np.complex128)[:, ::-1]
    covariance_fb = 0.5 * (covariance_f + reversal @ np.conjugate(covariance_f) @ reversal)
    trace_scale = float(np.real(np.trace(covariance_fb))) / max(1, subarray_len)
    if not np.isfinite(trace_scale) or trace_scale <= 0.0:
        trace_scale = 1.0
    covariance = covariance_fb + (
        SUPERRES_DIAGONAL_LOADING * trace_scale * np.eye(subarray_len, dtype=np.complex128)
    )
    if not np.all(np.isfinite(covariance)):
        return None

    return covariance, subarray_len, snapshot_count


def _estimate_mdl_source_count(eigenvalues, snapshot_count):
    eigenvalues = np.asarray(np.real(eigenvalues), dtype=np.float64).reshape(-1)
    order = np.argsort(eigenvalues)[::-1]
    sorted_values = np.maximum(eigenvalues[order], 1e-12)
    dim = int(sorted_values.size)
    if dim <= 1:
        return 0, sorted_values, np.full((dim,), np.inf, dtype=np.float64)

    snapshot_count = max(1, int(snapshot_count))
    mdl_scores = np.full((dim,), np.inf, dtype=np.float64)
    log_snapshot_count = np.log(float(snapshot_count))
    for source_count in range(dim):
        noise_count = dim - source_count
        if noise_count <= 0:
            continue
        noise_eigs = sorted_values[source_count:]
        arithmetic_mean = float(np.mean(noise_eigs))
        geometric_mean = float(np.exp(np.mean(np.log(np.maximum(noise_eigs, 1e-12)))))
        if arithmetic_mean <= 0.0 or geometric_mean <= 0.0:
            continue
        ratio = geometric_mean / arithmetic_mean
        if not np.isfinite(ratio) or ratio <= 0.0:
            continue
        mdl_scores[source_count] = (
            -snapshot_count * noise_count * np.log(ratio)
            + 0.5 * source_count * (2 * dim - source_count) * log_snapshot_count
        )
    best_source_count = int(np.argmin(mdl_scores))
    return max(0, min(best_source_count, dim - 1)), sorted_values, mdl_scores


def _estimate_mdl_source_count_batch_gpu(eigenvalues_desc, snapshot_count):
    eigenvalues_desc = cp.asarray(cp.real(eigenvalues_desc), dtype=cp.float64)
    row_count, dim = eigenvalues_desc.shape
    if dim <= 1:
        return cp.zeros((row_count,), dtype=cp.int32), eigenvalues_desc, cp.full((row_count, dim), cp.inf)

    snapshot_count = max(1, int(snapshot_count))
    safe_eigs = cp.maximum(eigenvalues_desc, 1e-12)
    mdl_scores = cp.full((row_count, dim), cp.inf, dtype=cp.float64)
    log_snapshot_count = float(np.log(float(snapshot_count)))
    for source_count in range(dim):
        noise_count = dim - source_count
        if noise_count <= 0:
            continue
        noise_eigs = safe_eigs[:, source_count:]
        arithmetic_mean = cp.mean(noise_eigs, axis=1)
        geometric_mean = cp.exp(cp.mean(cp.log(cp.maximum(noise_eigs, 1e-12)), axis=1))
        ratio = geometric_mean / cp.maximum(arithmetic_mean, 1e-12)
        ratio = cp.maximum(ratio, 1e-12)
        mdl_scores[:, source_count] = (
            -snapshot_count * noise_count * cp.log(ratio)
            + 0.5 * source_count * (2 * dim - source_count) * log_snapshot_count
        )
    best_source_counts = cp.argmin(mdl_scores, axis=1).astype(cp.int32, copy=False)
    return cp.clip(best_source_counts, 0, dim - 1), safe_eigs, mdl_scores


def _quadratic_form_fft_grid(matrix, fft_size):
    matrix = np.asarray(matrix, dtype=np.complex128)
    rows, cols = matrix.shape
    if rows == 0 or cols == 0 or rows != cols:
        return np.empty((0,), dtype=np.float64)
    fft_size = max(1, int(fft_size))
    coeffs = np.zeros((fft_size,), dtype=np.complex128)
    for offset in range(-(rows - 1), rows):
        coeffs[offset % fft_size] += np.trace(matrix, offset=offset)
    values = np.fft.fft(coeffs)
    return np.maximum(np.real(values), 1e-12)


def _quadratic_form_fft_grid_batch_gpu(matrices, fft_size):
    matrices = cp.asarray(matrices, dtype=cp.complex128)
    batch_size, rows, cols = matrices.shape
    if batch_size == 0 or rows == 0 or cols == 0 or rows != cols:
        return cp.empty((batch_size, 0), dtype=cp.float64)
    fft_size = max(1, int(fft_size))
    coeffs = cp.zeros((batch_size, fft_size), dtype=cp.complex128)
    for offset in range(-(rows - 1), rows):
        diagonal = cp.diagonal(matrices, offset=offset, axis1=1, axis2=2)
        coeffs[:, offset % fft_size] += cp.sum(diagonal, axis=1)
    values = cp.fft.fft(coeffs, axis=1)
    return cp.maximum(cp.real(values), 1e-12)


def _trace_coeffs_from_matrix_batch_gpu(matrices):
    matrices = cp.asarray(matrices, dtype=cp.complex128)
    batch_size, rows, cols = matrices.shape
    if batch_size == 0 or rows == 0 or cols == 0 or rows != cols:
        return cp.empty((batch_size, 0), dtype=cp.complex128)
    coeffs = cp.zeros((batch_size, 2 * rows - 1), dtype=cp.complex128)
    base_idx = rows - 1
    for offset in range(-(rows - 1), rows):
        diagonal = cp.diagonal(matrices, offset=offset, axis1=1, axis2=2)
        coeffs[:, base_idx + offset] = cp.sum(diagonal, axis=1)
    return coeffs


def _limit_point_algorithm_rows_by_power(row_powers, candidate_indices, max_rows):
    candidate_indices = np.asarray(candidate_indices, dtype=np.int32).reshape(-1)
    if candidate_indices.size == 0:
        return candidate_indices
    max_rows = max(1, int(max_rows))
    if candidate_indices.size <= max_rows:
        return candidate_indices
    row_powers = np.asarray(row_powers, dtype=np.float64).reshape(-1)
    valid_indices = candidate_indices[
        (candidate_indices >= 0) & (candidate_indices < row_powers.shape[0])
    ]
    if valid_indices.size == 0:
        return np.empty((0,), dtype=np.int32)
    order = np.argsort(row_powers[valid_indices])[::-1][:max_rows]
    return valid_indices[order].astype(np.int32, copy=False)


def _prepare_delay_superresolution_gpu_common(
    doppler_frequency_matrix,
    display_doppler_bins,
):
    if not USE_NVIDIA_GPU or cp is None or not _is_cupy_array(doppler_frequency_matrix):
        return None

    try:
        total_rows = int(doppler_frequency_matrix.shape[0])
        vector_len = int(doppler_frequency_matrix.shape[1])
        row_start, row_stop, row_step = _display_doppler_row_bounds(
            total_rows,
            display_doppler_bins,
            DISPLAY_DOWNSAMPLE,
        )
        display_rows = doppler_frequency_matrix[row_start:row_stop:row_step, :]
        row_count = int(display_rows.shape[0])
        if row_count <= 0:
            return None

        row_powers = cp.sum(cp.abs(display_rows) ** 2, axis=1)
        subarray_len = _select_superres_subarray_length(vector_len)
        if subarray_len <= 0:
            return None

        subarrays = cp.lib.stride_tricks.sliding_window_view(display_rows, subarray_len, axis=1)
        snapshot_count = int(subarrays.shape[1])
        if snapshot_count < 2:
            return None

        subarrays = cp.asarray(subarrays, dtype=cp.complex64)
        covariance_f = cp.einsum("rml,rmk->rlk", subarrays, cp.conjugate(subarrays)) / float(snapshot_count)
        reversal = cp.eye(subarray_len, dtype=cp.complex64)[:, ::-1]
        covariance_fb = 0.5 * (
            covariance_f + reversal[None, :, :] @ cp.conjugate(covariance_f) @ reversal[None, :, :]
        )
        trace_scale = cp.real(cp.trace(covariance_fb, axis1=1, axis2=2)) / max(1, subarray_len)
        trace_scale = cp.where(cp.isfinite(trace_scale) & (trace_scale > 0.0), trace_scale, 1.0)
        eye = cp.eye(subarray_len, dtype=cp.complex64)[None, :, :]
        covariance = covariance_fb + (SUPERRES_DIAGONAL_LOADING * trace_scale[:, None, None] * eye)
        finite_covariance = cp.all(cp.isfinite(covariance.reshape(row_count, -1)), axis=1)

        eigenvalues, eigenvectors = cp.linalg.eigh(covariance)
        order = cp.argsort(cp.real(eigenvalues), axis=1)[:, ::-1]
        eigenvalues_desc = cp.take_along_axis(cp.real(eigenvalues), order, axis=1)
        eigenvectors_desc = cp.take_along_axis(eigenvectors, order[:, None, :], axis=2)
        source_counts, _, _ = _estimate_mdl_source_count_batch_gpu(eigenvalues_desc, snapshot_count)
        valid_rows = finite_covariance & (source_counts > 0) & (source_counts < subarray_len)
        source_counts_cpu = cp.asnumpy(source_counts).astype(np.int32, copy=False)
        valid_rows_cpu = cp.asnumpy(valid_rows).astype(bool, copy=False)
        return {
            'display_rows': display_rows,
            'row_count': row_count,
            'vector_len': vector_len,
            'row_powers_cpu': cp.asnumpy(row_powers).astype(np.float64, copy=False),
            'subarray_len': subarray_len,
            'snapshot_count': snapshot_count,
            'covariance': covariance,
            'eigenvectors_desc': eigenvectors_desc,
            'source_counts_cpu': source_counts_cpu,
            'valid_rows_cpu': valid_rows_cpu,
        }
    except Exception:
        return None


def _delays_from_rootmusic_coeffs(coeffs, source_count, fft_size):
    coeffs = np.asarray(coeffs, dtype=np.complex128).reshape(-1)
    dim = (coeffs.size + 1) // 2
    if source_count <= 0 or dim <= 1 or coeffs.size == 0:
        return np.empty((0,), dtype=np.float64)
    if not np.any(np.abs(coeffs) > 1e-12):
        return np.empty((0,), dtype=np.float64)
    try:
        roots = np.roots(coeffs)
    except Exception:
        return np.empty((0,), dtype=np.float64)
    if roots.size == 0:
        return np.empty((0,), dtype=np.float64)

    inside_roots = roots[np.abs(roots) < 1.0 + 1e-6]
    if inside_roots.size == 0:
        inside_roots = roots
    order = np.argsort(np.abs(np.abs(inside_roots) - 1.0))
    chosen_delays = []
    for root in inside_roots[order]:
        delay_bin = float((-np.angle(root) * float(fft_size) / (2.0 * np.pi)) % float(fft_size))
        if any(abs(delay_bin - prev) < 0.5 for prev in chosen_delays):
            continue
        chosen_delays.append(delay_bin)
        if len(chosen_delays) >= int(source_count):
            break
    return np.asarray(chosen_delays, dtype=np.float64)


def _run_delay_superresolution_gpu_spectrum(
    precomputed,
    range_fft_size,
    display_range_bins,
    peak_threshold_db,
    processing_generation_snapshot=None,
):
    if precomputed is None:
        return None
    if delay_estimator_mode not in {DELAY_ESTIMATOR_MUSIC, DELAY_ESTIMATOR_CAPON}:
        return None

    try:
        if (
            processing_generation_snapshot is not None
            and get_processing_generation() != int(processing_generation_snapshot)
        ):
            return None
        row_count = int(precomputed['row_count'])
        vector_len = int(precomputed['vector_len'])
        subarray_len = int(precomputed['subarray_len'])
        covariance = precomputed['covariance']
        eigenvectors_desc = precomputed['eigenvectors_desc']
        source_counts_cpu = precomputed['source_counts_cpu']
        valid_rows_cpu = precomputed['valid_rows_cpu']

        spectrum_db_gpu = cp.full((row_count, display_range_bins), -cp.inf, dtype=cp.float32)
        for source_count in sorted(set(source_counts_cpu[valid_rows_cpu].tolist())):
            if int(source_count) <= 0 or int(source_count) >= subarray_len:
                continue
            if (
                processing_generation_snapshot is not None
                and get_processing_generation() != int(processing_generation_snapshot)
            ):
                return None
            group_indices_cpu = np.flatnonzero(valid_rows_cpu & (source_counts_cpu == int(source_count)))
            if group_indices_cpu.size == 0:
                continue
            group_indices_gpu = cp.asarray(group_indices_cpu, dtype=cp.int32)
            covariance_group = covariance[group_indices_gpu]
            if delay_estimator_mode == DELAY_ESTIMATOR_MUSIC:
                noise_subspace = eigenvectors_desc[group_indices_gpu][:, :, int(source_count):]
                projector = noise_subspace @ cp.conjugate(cp.transpose(noise_subspace, (0, 2, 1)))
                denominator = _quadratic_form_fft_grid_batch_gpu(projector, range_fft_size)[:, :display_range_bins]
            else:
                inverse_covariance = cp.linalg.inv(covariance_group)
                denominator = _quadratic_form_fft_grid_batch_gpu(inverse_covariance, range_fft_size)[:, :display_range_bins]
            spectrum_group = 20.0 * cp.log10(1.0 / cp.maximum(denominator, 1e-12))
            spectrum_db_gpu[group_indices_gpu] = spectrum_group.astype(cp.float32, copy=False)

        spectrum_db = to_cpu_array(spectrum_db_gpu, dtype=np.float32)
        finite_mask = np.isfinite(spectrum_db)
        if np.any(finite_mask):
            peak_db = float(np.max(spectrum_db[finite_mask]))
            relative_db = np.where(finite_mask, spectrum_db - peak_db, -np.inf).astype(np.float32, copy=False)
            detection_mask = _build_local_max_mask_2d(relative_db) & (relative_db >= float(peak_threshold_db))
            candidate_points = np.argwhere(detection_mask)
            candidate_strengths = (
                relative_db[candidate_points[:, 0], candidate_points[:, 1]].astype(np.float32, copy=False)
                if candidate_points.size > 0
                else np.empty((0,), dtype=np.float32)
            )
            detected_points, detected_strengths_db = _limit_ranked_points(
                candidate_points.astype(np.int32, copy=False),
                candidate_strengths,
                SUPERRES_MAX_POINTS,
            )
            display_floor = min(-60.0, float(peak_threshold_db) - 24.0)
            display_image = np.where(finite_mask, np.maximum(relative_db, display_floor), display_floor)
            rd_display = display_image.astype(np.float32, copy=False)
            rd_levels = (float(display_floor), 0.0)
        else:
            detected_points = np.empty((0, 2), dtype=np.int32)
            detected_strengths_db = np.empty((0,), dtype=np.float32)
            rd_display = np.full((row_count, display_range_bins), -60.0, dtype=np.float32)
            rd_levels = (-60.0, 0.0)

        rows_detected = int(len(np.unique(detected_points[:, 0]))) if detected_points.size > 0 else 0
        source_counts_valid = source_counts_cpu[valid_rows_cpu]
        summary = {
            'mode': get_delay_estimator_label(),
            'rows_total': int(row_count),
            'rows_processed': int(np.count_nonzero(valid_rows_cpu)),
            'rows_skipped': int(row_count - np.count_nonzero(valid_rows_cpu)),
            'rows_detected': int(rows_detected),
            'source_count_max': int(np.max(source_counts_valid)) if source_counts_valid.size > 0 else 0,
            'source_count_mean': float(np.mean(source_counts_valid)) if source_counts_valid.size > 0 else 0.0,
            'threshold_db': float(peak_threshold_db),
            'subcarrier_count': int(vector_len),
            'compute_backend': 'gpu-cupy',
        }
        return {
            'rd': rd_display,
            'rd_levels': rd_levels,
            'superres_rd': rd_display,
            'points': detected_points,
            'strengths_db': detected_strengths_db.astype(np.float32, copy=False),
            'summary': summary,
        }
    except Exception:
        return None


def _run_delay_superresolution_gpu_points(
    precomputed,
    range_fft_size,
    display_range_bins,
    peak_threshold_db,
    processing_generation_snapshot=None,
):
    if precomputed is None:
        return None
    if delay_estimator_mode not in {DELAY_ESTIMATOR_ROOTMUSIC, DELAY_ESTIMATOR_ESPRIT}:
        return None

    try:
        if (
            processing_generation_snapshot is not None
            and get_processing_generation() != int(processing_generation_snapshot)
        ):
            return None
        display_rows = precomputed['display_rows']
        row_count = int(precomputed['row_count'])
        vector_len = int(precomputed['vector_len'])
        row_powers_cpu = precomputed['row_powers_cpu']
        subarray_len = int(precomputed['subarray_len'])
        eigenvectors_desc = precomputed['eigenvectors_desc']
        source_counts_cpu = precomputed['source_counts_cpu']
        valid_rows_cpu = precomputed['valid_rows_cpu']

        point_candidates = []
        point_strength_candidates = []
        total_cpu_rows_fetched = 0
        rows_capped = 0
        for source_count in sorted(set(source_counts_cpu[valid_rows_cpu].tolist())):
            if int(source_count) <= 0 or int(source_count) >= subarray_len:
                continue
            group_indices_cpu = np.flatnonzero(valid_rows_cpu & (source_counts_cpu == int(source_count)))
            if group_indices_cpu.size == 0:
                continue
            limited_group_indices_cpu = _limit_point_algorithm_rows_by_power(
                row_powers_cpu,
                group_indices_cpu,
                SUPERRES_POINT_MAX_ROWS_PER_K,
            )
            rows_capped += max(0, int(group_indices_cpu.size - limited_group_indices_cpu.size))
            group_indices_cpu = limited_group_indices_cpu
            if group_indices_cpu.size == 0:
                continue
            if (
                processing_generation_snapshot is not None
                and get_processing_generation() != int(processing_generation_snapshot)
            ):
                return None
            group_indices_gpu = cp.asarray(group_indices_cpu, dtype=cp.int32)
            candidate_row_indices = []
            candidate_delay_sets = []
            if delay_estimator_mode == DELAY_ESTIMATOR_ROOTMUSIC:
                noise_subspace = eigenvectors_desc[group_indices_gpu][:, :, int(source_count):]
                coeffs_cpu = cp.asnumpy(
                    _trace_coeffs_from_matrix_batch_gpu(
                        noise_subspace @ cp.conjugate(cp.transpose(noise_subspace, (0, 2, 1)))
                    )
                ).astype(np.complex128, copy=False)
                for local_idx, coeffs in enumerate(coeffs_cpu):
                    delays = _delays_from_rootmusic_coeffs(coeffs, int(source_count), range_fft_size)
                    valid_delays = [
                        float(delay_value)
                        for delay_value in delays.tolist()
                        if np.isfinite(delay_value) and 0.0 <= delay_value < float(display_range_bins)
                    ]
                    if not valid_delays:
                        continue
                    candidate_row_indices.append(int(group_indices_cpu[local_idx]))
                    candidate_delay_sets.append(valid_delays)
            else:
                signal_subspaces_cpu = cp.asnumpy(
                    eigenvectors_desc[group_indices_gpu][:, :, :int(source_count)]
                ).astype(np.complex128, copy=False)
                for local_idx, signal_subspace in enumerate(signal_subspaces_cpu):
                    delays = _delays_from_esprit(signal_subspace, int(source_count), range_fft_size)
                    valid_delays = [
                        float(delay_value)
                        for delay_value in delays.tolist()
                        if np.isfinite(delay_value) and 0.0 <= delay_value < float(display_range_bins)
                    ]
                    if not valid_delays:
                        continue
                    candidate_row_indices.append(int(group_indices_cpu[local_idx]))
                    candidate_delay_sets.append(valid_delays)

            if not candidate_row_indices:
                continue

            candidate_row_indices = _limit_point_algorithm_rows_by_power(
                row_powers_cpu,
                np.asarray(candidate_row_indices, dtype=np.int32),
                SUPERRES_POINT_MAX_ROWS_PER_K,
            ).tolist()
            if not candidate_row_indices:
                continue
            if (
                processing_generation_snapshot is not None
                and get_processing_generation() != int(processing_generation_snapshot)
            ):
                return None
            allowed_rows = set(int(row_idx) for row_idx in candidate_row_indices)
            filtered_delay_sets = []
            for row_idx, delays in zip(
                [int(idx) for idx in np.asarray(group_indices_cpu, dtype=np.int32).tolist()],
                candidate_delay_sets,
            ):
                if row_idx in allowed_rows:
                    filtered_delay_sets.append((row_idx, delays))
            candidate_row_indices = [row_idx for row_idx, _ in filtered_delay_sets]
            candidate_delay_sets = [delays for _, delays in filtered_delay_sets]
            if not candidate_row_indices:
                continue
            needed_indices_gpu = cp.asarray(candidate_row_indices, dtype=cp.int32)
            row_vectors_cpu = cp.asnumpy(display_rows[needed_indices_gpu]).astype(np.complex64, copy=False)
            total_cpu_rows_fetched += int(len(candidate_row_indices))
            for row_idx_cpu, row_vector, delays in zip(candidate_row_indices, row_vectors_cpu, candidate_delay_sets):
                if (
                    processing_generation_snapshot is not None
                    and get_processing_generation() != int(processing_generation_snapshot)
                ):
                    return None
                strengths_db = _estimate_point_strengths(row_vector, delays, range_fft_size)
                for delay_value, strength_db in zip(delays, strengths_db.tolist()):
                    if not np.isfinite(strength_db):
                        continue
                    point_candidates.append((
                        int(row_idx_cpu),
                        int(np.clip(round(delay_value), 0, display_range_bins - 1)),
                    ))
                    point_strength_candidates.append(float(strength_db))

        if point_candidates:
            points_arr = np.asarray(point_candidates, dtype=np.int32)
            strengths_arr = np.asarray(point_strength_candidates, dtype=np.float64)
            peak_strength_db = float(np.max(strengths_arr))
            relative_strengths = (strengths_arr - peak_strength_db).astype(np.float32, copy=False)
            keep_mask = relative_strengths >= float(peak_threshold_db)
            points_arr = points_arr[keep_mask]
            relative_strengths = relative_strengths[keep_mask]
            if points_arr.size > 0:
                unique_strengths = {}
                for point, strength_db in zip(points_arr.tolist(), relative_strengths.tolist()):
                    point_key = (int(point[0]), int(point[1]))
                    prev = unique_strengths.get(point_key)
                    if prev is None or float(strength_db) > prev:
                        unique_strengths[point_key] = float(strength_db)
                sorted_items = sorted(unique_strengths.items(), key=lambda item: item[1], reverse=True)
                points_arr = np.asarray([item[0] for item in sorted_items], dtype=np.int32)
                relative_strengths = np.asarray([item[1] for item in sorted_items], dtype=np.float32)
            detected_points, detected_strengths_db = _limit_ranked_points(
                points_arr,
                relative_strengths,
                SUPERRES_MAX_POINTS,
            )
        else:
            detected_points = np.empty((0, 2), dtype=np.int32)
            detected_strengths_db = np.empty((0,), dtype=np.float32)

        rows_detected = int(len(np.unique(detected_points[:, 0]))) if detected_points.size > 0 else 0
        source_counts_valid = source_counts_cpu[valid_rows_cpu]
        summary = {
            'mode': get_delay_estimator_label(),
            'rows_total': int(row_count),
            'rows_processed': int(np.count_nonzero(valid_rows_cpu)),
            'rows_skipped': int(row_count - np.count_nonzero(valid_rows_cpu)),
            'rows_detected': int(rows_detected),
            'source_count_max': int(np.max(source_counts_valid)) if source_counts_valid.size > 0 else 0,
            'source_count_mean': float(np.mean(source_counts_valid)) if source_counts_valid.size > 0 else 0.0,
            'threshold_db': float(peak_threshold_db),
            'subcarrier_count': int(vector_len),
            'compute_backend': 'gpu-cupy-partial',
            'cpu_rows_fetched': int(total_cpu_rows_fetched),
            'rows_capped': int(rows_capped),
        }
        return {
            'rd': np.zeros((row_count, display_range_bins), dtype=np.float32),
            'rd_levels': (0.0, 1.0),
            'superres_rd': None,
            'points': detected_points,
            'strengths_db': detected_strengths_db.astype(np.float32, copy=False),
            'summary': summary,
        }
    except Exception:
        return None


def _delays_from_rootmusic(noise_projector, source_count, fft_size):
    noise_projector = np.asarray(noise_projector, dtype=np.complex128)
    dim = noise_projector.shape[0]
    if source_count <= 0 or dim <= 1:
        return np.empty((0,), dtype=np.float64)

    coeffs = np.asarray(
        [np.trace(noise_projector, offset=offset) for offset in range(-(dim - 1), dim)],
        dtype=np.complex128,
    )
    if not np.any(np.abs(coeffs) > 1e-12):
        return np.empty((0,), dtype=np.float64)

    try:
        roots = np.roots(coeffs)
    except Exception:
        return np.empty((0,), dtype=np.float64)
    if roots.size == 0:
        return np.empty((0,), dtype=np.float64)

    inside_roots = roots[np.abs(roots) < 1.0 + 1e-6]
    if inside_roots.size == 0:
        inside_roots = roots
    order = np.argsort(np.abs(np.abs(inside_roots) - 1.0))
    chosen_delays = []
    for root in inside_roots[order]:
        delay_bin = float((-np.angle(root) * float(fft_size) / (2.0 * np.pi)) % float(fft_size))
        if any(abs(delay_bin - prev) < 0.5 for prev in chosen_delays):
            continue
        chosen_delays.append(delay_bin)
        if len(chosen_delays) >= int(source_count):
            break
    return np.asarray(chosen_delays, dtype=np.float64)


def _delays_from_esprit(signal_subspace, source_count, fft_size):
    signal_subspace = np.asarray(signal_subspace, dtype=np.complex128)
    rows, cols = signal_subspace.shape
    if source_count <= 0 or rows < 2 or cols <= 0:
        return np.empty((0,), dtype=np.float64)

    upper = signal_subspace[:-1, :]
    lower = signal_subspace[1:, :]
    try:
        psi = np.linalg.pinv(upper) @ lower
        eigenvalues = np.linalg.eigvals(psi)
    except Exception:
        return np.empty((0,), dtype=np.float64)
    if eigenvalues.size == 0:
        return np.empty((0,), dtype=np.float64)

    order = np.argsort(np.abs(np.abs(eigenvalues) - 1.0))
    chosen_delays = []
    for eigenvalue in eigenvalues[order]:
        delay_bin = float((-np.angle(eigenvalue) * float(fft_size) / (2.0 * np.pi)) % float(fft_size))
        if any(abs(delay_bin - prev) < 0.5 for prev in chosen_delays):
            continue
        chosen_delays.append(delay_bin)
        if len(chosen_delays) >= int(source_count):
            break
    return np.asarray(chosen_delays, dtype=np.float64)


def _estimate_point_strengths(row_vector, delays, fft_size):
    row_vector = np.asarray(row_vector, dtype=np.complex128).reshape(-1)
    delays = np.asarray(delays, dtype=np.float64).reshape(-1)
    if row_vector.size == 0 or delays.size == 0:
        return np.empty((0,), dtype=np.float64)

    sample_idx = np.arange(row_vector.size, dtype=np.float64)
    steering = np.exp(
        -2j * np.pi * np.outer(sample_idx, delays.astype(np.float64)) / float(max(1, int(fft_size)))
    )
    try:
        coeffs, _, _, _ = np.linalg.lstsq(steering, row_vector, rcond=None)
        strengths = np.abs(coeffs)
    except Exception:
        strengths = np.abs(np.conjugate(steering).T @ row_vector) / max(1.0, float(row_vector.size))
    return 20.0 * np.log10(np.maximum(strengths, 1e-12))


def _limit_ranked_points(points, strengths_db, max_points):
    points = np.asarray(points, dtype=np.int32)
    strengths_db = np.asarray(strengths_db, dtype=np.float32)
    if points.size == 0 or strengths_db.size == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)
    if points.shape[0] != strengths_db.shape[0]:
        limit = min(points.shape[0], strengths_db.shape[0])
        points = points[:limit]
        strengths_db = strengths_db[:limit]
    order = np.argsort(strengths_db)[::-1]
    if points.shape[0] > int(max_points):
        order = order[:int(max_points)]
    return points[order], strengths_db[order]


def run_delay_superresolution(
    doppler_frequency_matrix,
    range_fft_size,
    display_range_bins,
    display_doppler_bins,
    processing_generation_snapshot=None,
):
    display_range_bins = max(1, int(display_range_bins))
    display_doppler_bins = max(1, int(display_doppler_bins))
    range_fft_size = max(1, int(range_fft_size))
    peak_threshold_db = float(superres_peak_rel_threshold_db)

    if doppler_frequency_matrix is None:
        blank = np.zeros((display_doppler_bins, display_range_bins), dtype=np.float32)
        return {
            'rd': blank,
            'rd_levels': (0.0, 1.0),
            'superres_rd': None,
            'points': np.empty((0, 2), dtype=np.int32),
            'strengths_db': np.empty((0,), dtype=np.float32),
            'summary': {
                'mode': get_delay_estimator_label(),
                'rows_total': 0,
                'rows_processed': 0,
                'rows_skipped': 0,
                'rows_detected': 0,
                'source_count_max': 0,
                'source_count_mean': 0.0,
                'threshold_db': peak_threshold_db,
                'reason': 'unsupported_frame_format',
            },
        }

    gpu_precomputed = _prepare_delay_superresolution_gpu_common(
        doppler_frequency_matrix,
        display_doppler_bins,
    )
    if gpu_precomputed is not None:
        gpu_result = _run_delay_superresolution_gpu_spectrum(
            gpu_precomputed,
            range_fft_size,
            display_range_bins,
            peak_threshold_db,
            processing_generation_snapshot=processing_generation_snapshot,
        )
        if gpu_result is not None:
            return gpu_result
        gpu_result = _run_delay_superresolution_gpu_points(
            gpu_precomputed,
            range_fft_size,
            display_range_bins,
            peak_threshold_db,
            processing_generation_snapshot=processing_generation_snapshot,
        )
        if gpu_result is not None:
            return gpu_result

    doppler_frequency_matrix = to_cpu_array(doppler_frequency_matrix, dtype=np.complex64)
    total_rows, vector_len = doppler_frequency_matrix.shape
    row_start, row_stop, row_step = _display_doppler_row_bounds(
        total_rows,
        display_doppler_bins,
        DISPLAY_DOWNSAMPLE,
    )
    display_rows = doppler_frequency_matrix[row_start:row_stop:row_step, :]
    if display_rows.size == 0:
        blank = np.zeros((0, display_range_bins), dtype=np.float32)
        return {
            'rd': blank,
            'rd_levels': (0.0, 1.0),
            'superres_rd': None,
            'points': np.empty((0, 2), dtype=np.int32),
            'strengths_db': np.empty((0,), dtype=np.float32),
            'summary': {
                'mode': get_delay_estimator_label(),
                'rows_total': 0,
                'rows_processed': 0,
                'rows_skipped': 0,
                'rows_detected': 0,
                'source_count_max': 0,
                'source_count_mean': 0.0,
                'threshold_db': peak_threshold_db,
                'reason': 'empty_display_window',
            },
        }

    estimator = delay_estimator_mode
    uses_spectrum = delay_estimator_outputs_spectrum()
    spectrum_db = np.full((display_rows.shape[0], display_range_bins), -np.inf, dtype=np.float64)
    point_candidates = []
    point_strength_candidates = []
    source_counts = []
    row_powers = np.sum(np.abs(display_rows) ** 2, axis=1).astype(np.float64, copy=False)
    rows_processed = 0
    rows_skipped = 0
    rows_capped = 0

    point_mode_row_limit = min(int(display_rows.shape[0]), int(SUPERRES_POINT_MAX_ROWS))
    if not uses_spectrum and point_mode_row_limit > 0:
        row_priority = _limit_point_algorithm_rows_by_power(
            row_powers,
            np.arange(display_rows.shape[0], dtype=np.int32),
            point_mode_row_limit,
        )
        row_priority_set = set(int(row_idx) for row_idx in row_priority.tolist())
        rows_capped = max(0, int(display_rows.shape[0] - len(row_priority_set)))
    else:
        row_priority_set = None

    for row_idx, row_vector in enumerate(display_rows):
        if (
            processing_generation_snapshot is not None
            and get_processing_generation() != int(processing_generation_snapshot)
        ):
            return {
                'rd': np.zeros((0, display_range_bins), dtype=np.float32),
                'rd_levels': (0.0, 1.0),
                'superres_rd': None,
                'points': np.empty((0, 2), dtype=np.int32),
                'strengths_db': np.empty((0,), dtype=np.float32),
                'summary': {
                    'mode': get_delay_estimator_label(),
                    'rows_total': int(display_rows.shape[0]),
                    'rows_processed': 0,
                    'rows_skipped': int(display_rows.shape[0]),
                    'rows_detected': 0,
                    'source_count_max': 0,
                    'source_count_mean': 0.0,
                    'threshold_db': float(peak_threshold_db),
                    'subcarrier_count': int(vector_len),
                    'compute_backend': 'cpu-cancelled',
                },
            }
        if row_priority_set is not None and int(row_idx) not in row_priority_set:
            rows_skipped += 1
            continue
        smoothed = _build_spatially_smoothed_covariance(row_vector)
        if smoothed is None:
            rows_skipped += 1
            continue

        covariance, subarray_len, snapshot_count = smoothed
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        except Exception:
            rows_skipped += 1
            continue

        order = np.argsort(np.real(eigenvalues))[::-1]
        eigenvalues = np.real(eigenvalues[order])
        eigenvectors = eigenvectors[:, order]
        source_count, _, _ = _estimate_mdl_source_count(eigenvalues, snapshot_count)
        if source_count <= 0 or source_count >= subarray_len:
            rows_skipped += 1
            continue

        signal_subspace = eigenvectors[:, :source_count]
        noise_subspace = eigenvectors[:, source_count:]
        rows_processed += 1
        source_counts.append(int(source_count))

        if estimator == DELAY_ESTIMATOR_MUSIC:
            noise_projector = noise_subspace @ np.conjugate(noise_subspace).T
            denominator = _quadratic_form_fft_grid(noise_projector, range_fft_size)[:display_range_bins]
            spectrum_db[row_idx, :] = 20.0 * np.log10(1.0 / np.maximum(denominator, 1e-12))
        elif estimator == DELAY_ESTIMATOR_CAPON:
            try:
                inverse_covariance = np.linalg.pinv(covariance)
            except Exception:
                rows_skipped += 1
                rows_processed = max(0, rows_processed - 1)
                if source_counts:
                    source_counts.pop()
                continue
            denominator = _quadratic_form_fft_grid(inverse_covariance, range_fft_size)[:display_range_bins]
            spectrum_db[row_idx, :] = 20.0 * np.log10(1.0 / np.maximum(denominator, 1e-12))
        elif estimator == DELAY_ESTIMATOR_ROOTMUSIC:
            noise_projector = noise_subspace @ np.conjugate(noise_subspace).T
            delays = _delays_from_rootmusic(noise_projector, source_count, range_fft_size)
            strengths_db = _estimate_point_strengths(row_vector, delays, range_fft_size)
            for delay_value, strength_db in zip(delays.tolist(), strengths_db.tolist()):
                if not np.isfinite(delay_value) or not np.isfinite(strength_db):
                    continue
                if delay_value < 0.0 or delay_value >= float(display_range_bins):
                    continue
                point_candidates.append((int(row_idx), int(np.clip(round(delay_value), 0, display_range_bins - 1))))
                point_strength_candidates.append(float(strength_db))
        elif estimator == DELAY_ESTIMATOR_ESPRIT:
            delays = _delays_from_esprit(signal_subspace, source_count, range_fft_size)
            strengths_db = _estimate_point_strengths(row_vector, delays, range_fft_size)
            for delay_value, strength_db in zip(delays.tolist(), strengths_db.tolist()):
                if not np.isfinite(delay_value) or not np.isfinite(strength_db):
                    continue
                if delay_value < 0.0 or delay_value >= float(display_range_bins):
                    continue
                point_candidates.append((int(row_idx), int(np.clip(round(delay_value), 0, display_range_bins - 1))))
                point_strength_candidates.append(float(strength_db))

    detected_points = np.empty((0, 2), dtype=np.int32)
    detected_strengths_db = np.empty((0,), dtype=np.float32)
    rd_levels = (0.0, 1.0)
    superres_rd = None

    if uses_spectrum:
        finite_mask = np.isfinite(spectrum_db)
        if np.any(finite_mask):
            peak_db = float(np.max(spectrum_db[finite_mask]))
            relative_db = np.where(finite_mask, spectrum_db - peak_db, -np.inf)
            detection_mask = _build_local_max_mask_2d(relative_db) & (relative_db >= peak_threshold_db)
            candidate_points = np.argwhere(detection_mask)
            candidate_strengths = (
                relative_db[candidate_points[:, 0], candidate_points[:, 1]].astype(np.float32, copy=False)
                if candidate_points.size > 0
                else np.empty((0,), dtype=np.float32)
            )
            detected_points, detected_strengths_db = _limit_ranked_points(
                candidate_points.astype(np.int32, copy=False),
                candidate_strengths,
                SUPERRES_MAX_POINTS,
            )
            display_floor = min(-60.0, peak_threshold_db - 24.0)
            display_image = np.where(finite_mask, np.maximum(relative_db, display_floor), display_floor)
            superres_rd = display_image.astype(np.float32, copy=False)
            rd_levels = (float(display_floor), 0.0)
        else:
            superres_rd = np.full((display_rows.shape[0], display_range_bins), -60.0, dtype=np.float32)
            rd_levels = (-60.0, 0.0)
        rd_display = np.asarray(superres_rd, dtype=np.float32)
    else:
        if point_candidates:
            points_arr = np.asarray(point_candidates, dtype=np.int32)
            strengths_arr = np.asarray(point_strength_candidates, dtype=np.float64)
            peak_strength_db = float(np.max(strengths_arr))
            relative_strengths = (strengths_arr - peak_strength_db).astype(np.float32, copy=False)
            keep_mask = relative_strengths >= float(peak_threshold_db)
            points_arr = points_arr[keep_mask]
            relative_strengths = relative_strengths[keep_mask]
            if points_arr.size > 0:
                unique_strengths = {}
                for point, strength_db in zip(points_arr.tolist(), relative_strengths.tolist()):
                    point_key = (int(point[0]), int(point[1]))
                    prev = unique_strengths.get(point_key)
                    if prev is None or float(strength_db) > prev:
                        unique_strengths[point_key] = float(strength_db)
                sorted_items = sorted(unique_strengths.items(), key=lambda item: item[1], reverse=True)
                points_arr = np.asarray([item[0] for item in sorted_items], dtype=np.int32)
                relative_strengths = np.asarray([item[1] for item in sorted_items], dtype=np.float32)
            detected_points, detected_strengths_db = _limit_ranked_points(
                points_arr,
                relative_strengths,
                SUPERRES_MAX_POINTS,
            )
        rd_display = np.zeros((display_rows.shape[0], display_range_bins), dtype=np.float32)
        rd_levels = (0.0, 1.0)

    rows_detected = int(len(np.unique(detected_points[:, 0]))) if detected_points.size > 0 else 0
    summary = {
        'mode': get_delay_estimator_label(),
        'rows_total': int(display_rows.shape[0]),
        'rows_processed': int(rows_processed),
        'rows_skipped': int(rows_skipped),
        'rows_detected': int(rows_detected),
        'source_count_max': int(max(source_counts)) if source_counts else 0,
        'source_count_mean': float(np.mean(source_counts)) if source_counts else 0.0,
        'threshold_db': float(peak_threshold_db),
        'subcarrier_count': int(vector_len),
        'compute_backend': 'cpu',
        'rows_capped': int(rows_capped),
    }
    return {
        'rd': rd_display.astype(np.float32, copy=False),
        'rd_levels': rd_levels,
        'superres_rd': None if superres_rd is None else np.asarray(superres_rd, dtype=np.float32),
        'points': detected_points,
        'strengths_db': detected_strengths_db.astype(np.float32, copy=False),
        'summary': summary,
    }


def accumulate_range_time_data_batch(channel_range_time_views):
    updated_channels = []
    for ch_idx, range_time_view in channel_range_time_views:
        if range_time_view is None or getattr(range_time_view, "size", 0) == 0:
            continue
        range_idx = min(selected_range_bin, range_time_view.shape[1] - 1)
        signal_slice = to_cpu_array(range_time_view[:, range_idx], dtype=np.complex64)
        with CHANNELS[ch_idx].micro_lock:
            CHANNELS[ch_idx].micro_doppler_buffer.extend(signal_slice)
        updated_channels.append(ch_idx)
    return updated_channels


def _batched_micro_doppler_stft(
    signal_batch,
    fs=1.0,
    nperseg=MICRO_DOPPLER_STFT_NPERSEG,
    noverlap=MICRO_DOPPLER_STFT_NOVERLAP,
    nfft=MICRO_DOPPLER_STFT_NFFT,
    display_norms=None,
):
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
    magnitude = np.abs(Zxx).astype(np.float64, copy=False)
    if display_norms is None:
        norm_view = 1.0
    else:
        norm_arr = np.asarray(display_norms, dtype=np.float64)
        if norm_arr.ndim == 0:
            norm_view = max(float(norm_arr), 1e-12)
        else:
            if norm_arr.shape[0] != batch_size:
                raise ValueError("display_norms length must match the signal batch size")
            norm_view = np.maximum(norm_arr.reshape(batch_size, 1, 1), 1e-12)
    Pxx_db = 20.0 * np.log10(magnitude / norm_view + 1e-12)
    Pxx_db_shifted = np.fft.fftshift(Pxx_db, axes=1)
    f_shifted = np.fft.fftshift(f)
    f_idx = (f_shifted > -0.5) & (f_shifted < 0.5)
    return f_shifted[f_idx], t, Pxx_db_shifted[:, f_idx, :].astype(np.float32, copy=False)


def calculate_micro_doppler_batch(channel_indices):
    if not channel_indices:
        return {}

    min_len = None
    for ch_idx in channel_indices:
        with CHANNELS[ch_idx].micro_lock:
            buf_len = len(CHANNELS[ch_idx].micro_doppler_buffer)
        if buf_len < MICRO_DOPPLER_STFT_NPERSEG:
            continue
        min_len = buf_len if min_len is None else min(min_len, buf_len)

    if min_len is None or min_len < MICRO_DOPPLER_STFT_NPERSEG:
        return {}

    eligible_channels = []
    signal_batch = []
    display_norms = []
    for ch_idx in channel_indices:
        with CHANNELS[ch_idx].micro_lock:
            if len(CHANNELS[ch_idx].micro_doppler_buffer) < min_len:
                continue
            signal = np.asarray(list(CHANNELS[ch_idx].micro_doppler_buffer)[-min_len:], dtype=np.complex64)
        eligible_channels.append(ch_idx)
        signal_batch.append(signal)
        display_norms.append(
            get_micro_doppler_display_amplitude_norm(
                get_viewer_params(CHANNELS[ch_idx]),
                MICRO_DOPPLER_STFT_NPERSEG,
            )
        )

    if not eligible_channels:
        return {}

    f, t, Pxx_batch = _batched_micro_doppler_stft(
        np.stack(signal_batch, axis=0),
        nperseg=MICRO_DOPPLER_STFT_NPERSEG,
        noverlap=MICRO_DOPPLER_STFT_NOVERLAP,
        nfft=MICRO_DOPPLER_STFT_NFFT,
        display_norms=np.asarray(display_norms, dtype=np.float64),
    )
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
    md_result=_MD_RESULT_UNSET,
):
    display_range_bins = get_display_range_bin_limit()
    display_doppler_bins = get_display_doppler_bin_limit()
    raw_rows = max(1, int(viewer_params.active_rows))
    raw_cols = max(1, int(viewer_params.active_cols))
    range_fft_size = max(raw_cols, get_processing_range_fft_size())
    doppler_fft_size = max(raw_rows, get_processing_doppler_fft_size())
    detector_mode = 'local_clean'

    rd_complex_plot, _, _, _ = build_detection_views(
        rd_complex,
        display_range_bins,
        display_doppler_bins,
        DISPLAY_DOWNSAMPLE,
    )

    cfar_points = np.empty((0, 2), dtype=np.int32)
    cfar_hits = 0
    cfar_shown_hits = 0
    cfar_backend = "off"
    cfar_stats = None
    rd_spectrum_plot = rd_spectrum[:0, :0]
    rd_spectrum_plot_cpu = np.empty((0, 0), dtype=np.float32)
    target_clusters = []
    detected_points = np.empty((0, 2), dtype=np.int32)
    detected_strengths_db = np.empty((0,), dtype=np.float32)
    detected_targets = []
    detector_summary = None
    rd_levels = None
    superres_rd = None
    clean_component_points = np.empty((0, 2), dtype=np.int32)
    clean_component_strength_db = np.empty((0,), dtype=np.float32)
    if delay_estimator_uses_superres():
        detector_mode = f"delay_superres_{delay_estimator_mode}"
        processing_generation_snapshot = get_processing_generation()
        try:
            doppler_frequency_matrix = process_doppler_frequency_input(raw_frame, viewer_params)
            superres_result = run_delay_superresolution(
                doppler_frequency_matrix,
                range_fft_size,
                display_range_bins,
                display_doppler_bins,
                processing_generation_snapshot=processing_generation_snapshot,
            )
        except Exception as exc:
            print(f"[CH{ch.ch_id + 1}] Delay superresolution error: {exc}")
            superres_result = {
                'rd': np.zeros((min(display_doppler_bins, doppler_fft_size), display_range_bins), dtype=np.float32),
                'rd_levels': (0.0, 1.0),
                'superres_rd': None,
                'points': np.empty((0, 2), dtype=np.int32),
                'strengths_db': np.empty((0,), dtype=np.float32),
                'summary': {
                    'mode': get_delay_estimator_label(),
                    'rows_total': 0,
                    'rows_processed': 0,
                    'rows_skipped': 0,
                    'rows_detected': 0,
                    'source_count_max': 0,
                    'source_count_mean': 0.0,
                    'threshold_db': float(superres_peak_rel_threshold_db),
                    'reason': str(exc),
                },
            }

        rd_spectrum_plot_cpu = np.asarray(superres_result.get('rd'), dtype=np.float32)
        rd_levels = superres_result.get('rd_levels')
        superres_rd = superres_result.get('superres_rd')
        detected_points = np.asarray(superres_result.get('points'), dtype=np.int32)
        detected_strengths_db = np.asarray(superres_result.get('strengths_db'), dtype=np.float32)
        detector_summary = dict(superres_result.get('summary') or {})
        detector_summary['mode'] = get_delay_estimator_label()
        cfar_points = detected_points
        cfar_hits = int(detected_points.shape[0])
        cfar_shown_hits = int(detected_points.shape[0])
        cfar_backend = delay_estimator_mode
        cfar_stats = detector_summary
        target_clusters = build_direct_targets(
            detected_points,
            rd_spectrum_plot_cpu,
            point_strengths_db=detected_strengths_db,
        )
        detected_targets = list(target_clusters)
    elif local_detector_mode == LOCAL_DETECTOR_OS_CFAR:
        detector_mode = 'local_os_cfar'
        os_pad_rows = max(0, int(cfar_train_doppler)) + max(0, int(cfar_guard_doppler))
        os_pad_cols = max(0, int(cfar_train_range)) + max(0, int(cfar_guard_range))
        rd_spectrum_plot, rd_spectrum_os, cfar_row_offset, cfar_col_offset = build_detection_views(
            rd_spectrum,
            display_range_bins,
            display_doppler_bins,
            DISPLAY_DOWNSAMPLE,
            pad_doppler_bins=os_pad_rows,
            pad_range_bins=os_pad_cols,
        )
        rd_spectrum_plot_cpu = to_cpu_array(rd_spectrum_plot, dtype=np.float32)
        rd_spectrum_os_cpu = to_cpu_array(rd_spectrum_os, dtype=np.float32)
        if cfar_enabled and rd_spectrum_plot.size > 0:
            cfar_points, cfar_hits, cfar_shown_hits, cfar_backend, cfar_stats = run_os_cfar_2d(
                rd_spectrum_os_cpu,
                active_row_start=cfar_row_offset,
                active_row_stop=cfar_row_offset + rd_spectrum_plot_cpu.shape[0],
                active_col_start=cfar_col_offset,
                active_col_stop=cfar_col_offset + rd_spectrum_plot_cpu.shape[1],
                dc_center_row=cfar_row_offset + (rd_spectrum_plot_cpu.shape[0] // 2),
            )
            if cfar_points.size > 0:
                cfar_points = cfar_points - np.array([cfar_row_offset, cfar_col_offset], dtype=np.int32)
                in_bounds = (
                    (cfar_points[:, 0] >= 0)
                    & (cfar_points[:, 0] < rd_spectrum_plot_cpu.shape[0])
                    & (cfar_points[:, 1] >= 0)
                    & (cfar_points[:, 1] < rd_spectrum_plot_cpu.shape[1])
                )
                cfar_points = cfar_points[in_bounds]
                cfar_shown_hits = int(cfar_points.shape[0])
    else:
        clean_params = get_local_clean_params()
        clean_range_window = enable_range_window and not local_clean_disables_windows()
        clean_doppler_window = enable_doppler_window and not local_clean_disables_windows()
        clean_pad_rows, clean_pad_cols = estimate_clean_padding(
            raw_rows=raw_rows,
            raw_cols=raw_cols,
            range_fft_size=range_fft_size,
            doppler_fft_size=doppler_fft_size,
            downsample=DISPLAY_DOWNSAMPLE,
            enable_range_window=clean_range_window,
            enable_doppler_window=clean_doppler_window,
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
                enable_range_window=clean_range_window,
                enable_doppler_window=clean_doppler_window,
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
        cfar_points = clean_component_points
        cfar_shown_hits = int(cfar_points.shape[0])

    rd_complex_plot_cpu = to_cpu_array(rd_complex_plot, dtype=np.complex64)
    range_time_view_cpu = None
    if range_time_view is not None:
        range_time_view_cpu = to_cpu_array(range_time_view, dtype=np.complex64)

    with ch.range_time_lock:
        ch.range_time_data = range_time_view_cpu

    if detector_mode == 'local_os_cfar':
        if cfar_points.size > 0:
            target_clusters = cluster_detected_targets(cfar_points, rd_spectrum_plot_cpu)
    elif clean_component_points.size > 0:
        target_clusters = build_direct_targets(
            clean_component_points,
            rd_spectrum_plot_cpu,
            point_strengths_db=clean_component_strength_db,
        )
    elif delay_estimator_uses_superres():
        target_clusters = detected_targets

    md_spectrum = None
    md_extent = None
    if show_micro_doppler:
        if md_result is _MD_RESULT_UNSET:
            accumulate_range_time_data(ch)
            f, t, Pxx = calculate_micro_doppler(ch)
            if Pxx is not None:
                md_spectrum = Pxx
                x0 = float(t[0]) if len(t) > 0 else 0.0
                x1 = float(t[-1]) if len(t) > 1 else (x0 + 1.0)
                y0 = float(f[0]) if len(f) > 0 else -0.5
                y1 = float(f[-1]) if len(f) > 1 else (y0 + 1.0)
                md_extent = [x0, x1, y0, y1]
        elif md_result is not None:
            _, _, Pxx, md_extent = md_result
            md_spectrum = Pxx

    if not delay_estimator_uses_superres():
        detected_points = np.asarray(cfar_points, dtype=np.int32)
        detected_targets = list(target_clusters)
        detector_summary = {
            'mode': get_local_detector_label(),
            'enabled': bool(cfar_enabled),
            'backend': cfar_backend,
            'raw_hits': int(cfar_hits),
            'shown_hits': int(cfar_shown_hits),
        }
        if isinstance(cfar_stats, dict):
            detector_summary.update(cfar_stats)

    dsp_time = time.time() - t_start

    result = {
        'ch_id': ch.ch_id,
        'frame_id': int(frame_id),
        'raw': raw_frame,
        'rd': rd_spectrum_plot_cpu,
        'rd_levels': rd_levels,
        'rd_complex': rd_complex_plot_cpu,
        'superres_rd': superres_rd,
        'delay_estimator_mode': delay_estimator_mode,
        'detected_points': detected_points,
        'detected_strengths_db': detected_strengths_db,
        'detected_targets': detected_targets,
        'detector_summary': detector_summary,
        'cfar_points': cfar_points,
        'cfar_hits': cfar_hits,
        'cfar_shown_hits': cfar_shown_hits,
        'cfar_backend': cfar_backend,
        'cfar_stats': cfar_stats,
        'detector_mode': detector_mode,
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

        if backend_stream_unsupported(viewer_params):
            warn_unsupported_backend_payload("[DSP]", "aggregate sensing frame", frame_id)
            continue

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
                    md_result=None,
                )
                bundle_results[ch_idx] = result
                bundle_rd_complex[ch_idx] = result.get('rd_complex')
            except Exception as e:
                print(f"[CH{ch.ch_id + 1}] DSP Worker Error: {e}")
                import traceback
                traceback.print_exc()

        if show_micro_doppler and batch_outputs:
            updated_channels = accumulate_range_time_data_batch(
                [
                    (ch_idx, CHANNELS[ch_idx].range_time_data)
                    for ch_idx, _ in active_frames
                    if ch_idx in batch_outputs and bundle_results[ch_idx] is not None
                ]
            )
            md_outputs = calculate_micro_doppler_batch(updated_channels)
            for ch_idx, md_result in md_outputs.items():
                if ch_idx >= len(bundle_results) or bundle_results[ch_idx] is None:
                    continue
                _, _, Pxx, md_extent = md_result
                bundle_results[ch_idx]['md'] = Pxx
                bundle_results[ch_idx]['md_extent'] = md_extent

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
    # Select the logical sensing channel before applying alignment.
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
        self.targets_window.resize(820, 860)
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
        rd_default_db_levels = get_delay_doppler_display_db_range()
        self.rd_colorbar = pg.ColorBarItem(values=rd_default_db_levels, colorMap=pg.colormap.get('turbo'), interactive=False)
        self.rd_colorbar.setImageItem(self.rd_img, insert_in=self.rd_plot.plotItem)
        self.rd_img.setLevels(rd_default_db_levels)

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
        md_default_db_levels = get_micro_doppler_display_db_range()
        self.md_colorbar = pg.ColorBarItem(values=md_default_db_levels, colorMap=pg.colormap.get('turbo'), interactive=False)
        self.md_colorbar.setImageItem(self.md_img, insert_in=self.md_plot.plotItem)
        self.md_img.setLevels(md_default_db_levels)
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
        self.lbl_cfar = QtWidgets.QLabel("Detector: off")
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
        self.targets_text.setPlainText("Top Targets: waiting detector hits")
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

        delay_doppler_db_layout = QtWidgets.QHBoxLayout()
        delay_doppler_db_layout.addWidget(QtWidgets.QLabel("Delay Doppler dB:"))
        self.txt_delay_doppler_db_min = QtWidgets.QLineEdit(f"{DISPLAY_RD_DB_MIN:.1f}")
        self.txt_delay_doppler_db_min.setPlaceholderText("Min")
        self.txt_delay_doppler_db_max = QtWidgets.QLineEdit(f"{DISPLAY_RD_DB_MAX:.1f}")
        self.txt_delay_doppler_db_max.setPlaceholderText("Max")
        btn_delay_doppler_db = QtWidgets.QPushButton("Apply")
        btn_delay_doppler_db.clicked.connect(self.apply_delay_doppler_display_db_range)
        delay_doppler_db_layout.addWidget(self.txt_delay_doppler_db_min)
        delay_doppler_db_layout.addWidget(self.txt_delay_doppler_db_max)
        delay_doppler_db_layout.addWidget(btn_delay_doppler_db)
        control_layout.addLayout(delay_doppler_db_layout)

        micro_doppler_db_layout = QtWidgets.QHBoxLayout()
        micro_doppler_db_layout.addWidget(QtWidgets.QLabel("MicroDoppler dB:"))
        self.txt_micro_doppler_db_min = QtWidgets.QLineEdit(f"{DISPLAY_MD_DB_MIN:.1f}")
        self.txt_micro_doppler_db_min.setPlaceholderText("Min")
        self.txt_micro_doppler_db_max = QtWidgets.QLineEdit(f"{DISPLAY_MD_DB_MAX:.1f}")
        self.txt_micro_doppler_db_max.setPlaceholderText("Max")
        btn_micro_doppler_db = QtWidgets.QPushButton("Apply")
        btn_micro_doppler_db.clicked.connect(self.apply_micro_doppler_display_db_range)
        micro_doppler_db_layout.addWidget(self.txt_micro_doppler_db_min)
        micro_doppler_db_layout.addWidget(self.txt_micro_doppler_db_max)
        micro_doppler_db_layout.addWidget(btn_micro_doppler_db)
        control_layout.addLayout(micro_doppler_db_layout)
        self.superres_controls = QtWidgets.QWidget()
        superres_layout = QtWidgets.QVBoxLayout(self.superres_controls)
        superres_layout.setContentsMargins(0, 0, 0, 0)
        superres_layout.setSpacing(6)

        superres_mode_layout = QtWidgets.QHBoxLayout()
        superres_mode_layout.addWidget(QtWidgets.QLabel("Delay Estimator:"))
        self.combo_delay_estimator = QtWidgets.QComboBox()
        self.combo_delay_estimator.addItem("FFT", DELAY_ESTIMATOR_FFT)
        self.combo_delay_estimator.addItem("MUSIC", DELAY_ESTIMATOR_MUSIC)
        self.combo_delay_estimator.addItem("CAPON", DELAY_ESTIMATOR_CAPON)
        self.combo_delay_estimator.addItem("ROOTMUSIC", DELAY_ESTIMATOR_ROOTMUSIC)
        self.combo_delay_estimator.addItem("ESPRIT", DELAY_ESTIMATOR_ESPRIT)
        self.combo_delay_estimator.currentIndexChanged.connect(self.on_delay_estimator_changed)
        superres_mode_layout.addWidget(self.combo_delay_estimator)
        superres_layout.addLayout(superres_mode_layout)

        superres_threshold_layout = QtWidgets.QHBoxLayout()
        superres_threshold_layout.addWidget(QtWidgets.QLabel("Peak Rel(dB):"))
        self.txt_superres_peak_rel = QtWidgets.QLineEdit(f"{superres_peak_rel_threshold_db:.1f}")
        btn_superres_apply = QtWidgets.QPushButton("Apply")
        btn_superres_apply.clicked.connect(self.apply_superres_settings)
        superres_threshold_layout.addWidget(self.txt_superres_peak_rel)
        superres_threshold_layout.addWidget(btn_superres_apply)
        superres_layout.addLayout(superres_threshold_layout)
        control_layout.addWidget(self.superres_controls)
        # Micro-Doppler Toggle
        self.btn_md = QtWidgets.QPushButton("Micro-Doppler: ON")
        self.btn_md.setCheckable(True)
        self.btn_md.setChecked(True)
        self.btn_md.clicked.connect(self.toggle_micro_doppler)
        self.btn_md.setStyleSheet("QPushButton:checked { background-color: lightgreen; }")
        control_layout.addWidget(self.btn_md)
        control_layout.addSpacing(12)

        self.local_detector_selector = QtWidgets.QWidget()
        local_detector_select_layout = QtWidgets.QHBoxLayout(self.local_detector_selector)
        local_detector_select_layout.setContentsMargins(0, 0, 0, 0)
        local_detector_select_layout.addWidget(QtWidgets.QLabel("Local Detector:"))
        self.combo_local_detector = QtWidgets.QComboBox()
        self.combo_local_detector.addItem("OS-CFAR", LOCAL_DETECTOR_OS_CFAR)
        self.combo_local_detector.addItem("CLEAN", LOCAL_DETECTOR_CLEAN)
        self.combo_local_detector.currentIndexChanged.connect(self.on_local_detector_changed)
        local_detector_select_layout.addWidget(self.combo_local_detector)
        control_layout.addWidget(self.local_detector_selector)

        self.btn_cfar = QtWidgets.QPushButton("Local OS-CFAR: OFF")
        self.btn_cfar.setCheckable(True)
        self.btn_cfar.setChecked(False)
        self.btn_cfar.clicked.connect(self.toggle_cfar)
        control_layout.addWidget(self.btn_cfar)

        self.local_detector_controls = QtWidgets.QWidget()
        local_detector_layout = QtWidgets.QVBoxLayout(self.local_detector_controls)
        local_detector_layout.setContentsMargins(0, 0, 0, 0)
        local_detector_layout.setSpacing(6)

        self.local_os_cfar_controls = QtWidgets.QWidget()
        local_os_cfar_layout = QtWidgets.QVBoxLayout(self.local_os_cfar_controls)
        local_os_cfar_layout.setContentsMargins(0, 0, 0, 0)
        local_os_cfar_layout.setSpacing(6)

        local_os_rank_layout = QtWidgets.QHBoxLayout()
        local_os_rank_layout.addWidget(QtWidgets.QLabel("Rank(%):"))
        self.txt_local_cfar_rank = QtWidgets.QLineEdit(f"{cfar_os_rank_percent:.0f}")
        local_os_rank_layout.addWidget(self.txt_local_cfar_rank)
        local_os_cfar_layout.addLayout(local_os_rank_layout)

        local_os_train_layout = QtWidgets.QHBoxLayout()
        local_os_train_layout.addWidget(QtWidgets.QLabel("Train D:"))
        self.txt_local_cfar_train_d = QtWidgets.QLineEdit(str(cfar_train_doppler))
        local_os_train_layout.addWidget(self.txt_local_cfar_train_d)
        local_os_train_layout.addWidget(QtWidgets.QLabel("Train R:"))
        self.txt_local_cfar_train_r = QtWidgets.QLineEdit(str(cfar_train_range))
        local_os_train_layout.addWidget(self.txt_local_cfar_train_r)
        local_os_cfar_layout.addLayout(local_os_train_layout)

        local_os_guard_layout = QtWidgets.QHBoxLayout()
        local_os_guard_layout.addWidget(QtWidgets.QLabel("Guard D:"))
        self.txt_local_cfar_guard_d = QtWidgets.QLineEdit(str(cfar_guard_doppler))
        local_os_guard_layout.addWidget(self.txt_local_cfar_guard_d)
        local_os_guard_layout.addWidget(QtWidgets.QLabel("Guard R:"))
        self.txt_local_cfar_guard_r = QtWidgets.QLineEdit(str(cfar_guard_range))
        local_os_guard_layout.addWidget(self.txt_local_cfar_guard_r)
        local_os_cfar_layout.addLayout(local_os_guard_layout)

        local_os_suppress_layout = QtWidgets.QHBoxLayout()
        local_os_suppress_layout.addWidget(QtWidgets.QLabel("Supp D:"))
        self.txt_local_cfar_suppress_d = QtWidgets.QLineEdit(str(cfar_os_suppress_doppler))
        local_os_suppress_layout.addWidget(self.txt_local_cfar_suppress_d)
        local_os_suppress_layout.addWidget(QtWidgets.QLabel("Supp R:"))
        self.txt_local_cfar_suppress_r = QtWidgets.QLineEdit(str(cfar_os_suppress_range))
        local_os_suppress_layout.addWidget(self.txt_local_cfar_suppress_r)
        local_os_cfar_layout.addLayout(local_os_suppress_layout)

        local_os_misc_layout = QtWidgets.QHBoxLayout()
        local_os_misc_layout.addWidget(QtWidgets.QLabel("Alpha(dB):"))
        self.txt_local_cfar_alpha = QtWidgets.QLineEdit(f"{cfar_alpha_db:.2f}")
        local_os_misc_layout.addWidget(self.txt_local_cfar_alpha)
        local_os_misc_layout.addWidget(QtWidgets.QLabel("Min R:"))
        self.txt_local_cfar_min_range = QtWidgets.QLineEdit(str(cfar_min_range_bin))
        local_os_misc_layout.addWidget(self.txt_local_cfar_min_range)
        local_os_misc_layout.addWidget(QtWidgets.QLabel("Min P(dB):"))
        self.txt_local_cfar_min_power = QtWidgets.QLineEdit(f"{cfar_min_power_db:.1f}")
        local_os_misc_layout.addWidget(self.txt_local_cfar_min_power)
        local_os_cfar_layout.addLayout(local_os_misc_layout)

        local_os_dc_layout = QtWidgets.QHBoxLayout()
        local_os_dc_layout.addWidget(QtWidgets.QLabel("DC Excl:"))
        self.txt_local_cfar_dc_excl = QtWidgets.QLineEdit(str(cfar_dc_exclusion_bins))
        btn_local_cfar_apply = QtWidgets.QPushButton("Apply OS-CFAR")
        btn_local_cfar_apply.clicked.connect(self.apply_local_cfar_settings)
        local_os_dc_layout.addWidget(self.txt_local_cfar_dc_excl)
        local_os_dc_layout.addWidget(btn_local_cfar_apply)
        local_os_cfar_layout.addLayout(local_os_dc_layout)
        local_detector_layout.addWidget(self.local_os_cfar_controls)

        self.local_clean_controls = QtWidgets.QWidget()
        local_clean_layout = QtWidgets.QVBoxLayout(self.local_clean_controls)
        local_clean_layout.setContentsMargins(0, 0, 0, 0)
        local_clean_layout.setSpacing(6)

        clean_main_layout = QtWidgets.QHBoxLayout()
        clean_main_layout.addWidget(QtWidgets.QLabel("Loop Gain:"))
        self.txt_clean_loop_gain = QtWidgets.QLineEdit(f"{clean_loop_gain:.2f}")
        clean_main_layout.addWidget(self.txt_clean_loop_gain)
        clean_main_layout.addWidget(QtWidgets.QLabel("Max Targets:"))
        self.txt_clean_max_targets = QtWidgets.QLineEdit(str(clean_max_targets))
        clean_main_layout.addWidget(self.txt_clean_max_targets)
        local_clean_layout.addLayout(clean_main_layout)

        clean_misc_layout = QtWidgets.QHBoxLayout()
        clean_misc_layout.addWidget(QtWidgets.QLabel("Min R:"))
        self.txt_clean_min_range = QtWidgets.QLineEdit(str(cfar_min_range_bin))
        clean_misc_layout.addWidget(self.txt_clean_min_range)
        clean_misc_layout.addWidget(QtWidgets.QLabel("Min P(dB):"))
        self.txt_clean_min_power = QtWidgets.QLineEdit(f"{clean_min_power_db:.1f}")
        clean_misc_layout.addWidget(self.txt_clean_min_power)
        local_clean_layout.addLayout(clean_misc_layout)

        clean_dc_layout = QtWidgets.QHBoxLayout()
        clean_dc_layout.addWidget(QtWidgets.QLabel("DC Excl:"))
        self.txt_clean_dc_excl = QtWidgets.QLineEdit(str(cfar_dc_exclusion_bins))
        btn_clean_apply = QtWidgets.QPushButton("Apply CLEAN")
        btn_clean_apply.clicked.connect(self.apply_clean_settings)
        clean_dc_layout.addWidget(self.txt_clean_dc_excl)
        clean_dc_layout.addWidget(btn_clean_apply)
        local_clean_layout.addLayout(clean_dc_layout)
        local_detector_layout.addWidget(self.local_clean_controls)
        control_layout.addWidget(self.local_detector_controls)

        self.cluster_controls = QtWidgets.QWidget()
        cluster_gap_layout = QtWidgets.QHBoxLayout(self.cluster_controls)
        cluster_gap_layout.setContentsMargins(0, 0, 0, 0)
        cluster_gap_layout.addWidget(QtWidgets.QLabel("DBSCAN Eps:"))
        self.lbl_dbscan_eps = QtWidgets.QLabel("")
        cluster_gap_layout.addWidget(self.lbl_dbscan_eps)
        cluster_gap_layout.addWidget(QtWidgets.QLabel("MinPts:"))
        self.txt_cluster_min_samples = QtWidgets.QLineEdit(str(target_dbscan_min_samples))
        cluster_gap_layout.addWidget(self.txt_cluster_min_samples)
        btn_cluster_apply = QtWidgets.QPushButton("Apply DBSCAN")
        btn_cluster_apply.clicked.connect(self.apply_target_cluster_settings)
        cluster_gap_layout.addWidget(btn_cluster_apply)
        control_layout.addWidget(self.cluster_controls)

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
        self.combo_delay_estimator.blockSignals(True)
        self.combo_delay_estimator.setCurrentIndex(
            self.combo_delay_estimator.findData(delay_estimator_mode)
        )
        self.combo_delay_estimator.blockSignals(False)
        self.combo_local_detector.blockSignals(True)
        self.combo_local_detector.setCurrentIndex(
            self.combo_local_detector.findData(local_detector_mode)
        )
        self.combo_local_detector.blockSignals(False)
        self.local_detector_controls.setVisible(True)
        self.local_os_cfar_controls.setVisible(local_detector_mode == LOCAL_DETECTOR_OS_CFAR)
        self.local_clean_controls.setVisible(local_detector_mode == LOCAL_DETECTOR_CLEAN)
        self.refresh_detector_controls(force=True)

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

    def _enforce_clean_window_policy(self, announce=False):
        global enable_range_window, enable_doppler_window
        if not local_clean_disables_windows():
            self.btn_range_win.setEnabled(True)
            self.btn_doppler_win.setEnabled(True)
            self.btn_range_win.setToolTip("")
            self.btn_doppler_win.setToolTip("")
            self.update_toggle_style(self.btn_range_win, enable_range_window)
            self.update_toggle_style(self.btn_doppler_win, enable_doppler_window)
            return False

        changed = False
        if enable_range_window:
            enable_range_window = False
            changed = True
        if enable_doppler_window:
            enable_doppler_window = False
            changed = True
        self.btn_range_win.setEnabled(False)
        self.btn_doppler_win.setEnabled(False)
        self.btn_range_win.setToolTip("Disabled in Local CLEAN mode")
        self.btn_doppler_win.setToolTip("Disabled in Local CLEAN mode")
        self.update_toggle_style(self.btn_range_win, False)
        self.update_toggle_style(self.btn_doppler_win, False)
        if changed and announce:
            print("Local CLEAN forces Range Window and Doppler Window OFF")
        return changed

    def apply_superres_settings(self):
        global superres_peak_rel_threshold_db
        try:
            superres_peak_rel_threshold_db = float(self.txt_superres_peak_rel.text())
            self.txt_superres_peak_rel.setText(f"{superres_peak_rel_threshold_db:.1f}")
            self._force_ui_refresh = True
            self._last_rendered_display_key = None
            self._last_top_targets_key = None
            reset_processing_state()
            print(
                "Delay superresolution updated: "
                f"mode={get_delay_estimator_label()} "
                f"peak_rel_threshold_db={superres_peak_rel_threshold_db:.1f}"
            )
        except ValueError:
            print("Invalid Peak Rel Threshold")

    def on_delay_estimator_changed(self, _idx):
        global delay_estimator_mode
        selected = self.combo_delay_estimator.currentData()
        if selected not in DELAY_ESTIMATOR_CHOICES:
            selected = DELAY_ESTIMATOR_FFT
        if delay_estimator_mode == selected:
            return
        delay_estimator_mode = selected
        self.refresh_detector_controls(force=True)
        self._force_ui_refresh = True
        self._last_rendered_display_key = None
        self._last_top_targets_key = None
        reset_processing_state()
        print(f"Delay estimator switched to: {get_delay_estimator_label()}")

    def refresh_detector_controls(self, force=False):
        changed_windows = self._enforce_clean_window_policy(announce=force)
        using_superres = delay_estimator_uses_superres()
        if force:
            self.combo_local_detector.blockSignals(True)
            combo_idx = self.combo_local_detector.findData(local_detector_mode)
            if combo_idx >= 0:
                self.combo_local_detector.setCurrentIndex(combo_idx)
            self.combo_local_detector.blockSignals(False)
            self.combo_delay_estimator.blockSignals(True)
            estimator_idx = self.combo_delay_estimator.findData(delay_estimator_mode)
            if estimator_idx >= 0:
                self.combo_delay_estimator.setCurrentIndex(estimator_idx)
            self.combo_delay_estimator.blockSignals(False)
            self.txt_superres_peak_rel.setText(f"{superres_peak_rel_threshold_db:.1f}")
            self.txt_clean_loop_gain.setText(f"{clean_loop_gain:.2f}")
            self.txt_clean_max_targets.setText(str(clean_max_targets))
            self.txt_clean_min_range.setText(str(cfar_min_range_bin))
            self.txt_clean_min_power.setText(f"{clean_min_power_db:.1f}")
            self.txt_clean_dc_excl.setText(str(cfar_dc_exclusion_bins))
        self.lbl_dbscan_eps.setText(f"D={get_dbscan_eps_doppler()} R={get_dbscan_eps_range()}")
        if force:
            self.txt_cluster_min_samples.setText(str(target_dbscan_min_samples))
            self.txt_local_cfar_rank.setText(f"{cfar_os_rank_percent:.0f}")
            self.txt_local_cfar_train_d.setText(str(cfar_train_doppler))
            self.txt_local_cfar_train_r.setText(str(cfar_train_range))
            self.txt_local_cfar_guard_d.setText(str(cfar_guard_doppler))
            self.txt_local_cfar_guard_r.setText(str(cfar_guard_range))
            self.txt_local_cfar_suppress_d.setText(str(cfar_os_suppress_doppler))
            self.txt_local_cfar_suppress_r.setText(str(cfar_os_suppress_range))
            self.txt_local_cfar_alpha.setText(f"{cfar_alpha_db:.2f}")
            self.txt_local_cfar_min_range.setText(str(cfar_min_range_bin))
            self.txt_local_cfar_min_power.setText(f"{cfar_min_power_db:.1f}")
            self.txt_local_cfar_dc_excl.setText(str(cfar_dc_exclusion_bins))
        self.superres_controls.setVisible(True)
        self.local_detector_controls.setVisible(not using_superres)
        self.local_detector_selector.setVisible(not using_superres)
        self.cluster_controls.setVisible(not using_superres)
        self.btn_cfar.setVisible(not using_superres)
        self.local_clean_controls.setVisible((not using_superres) and local_detector_mode == LOCAL_DETECTOR_CLEAN)
        self.local_os_cfar_controls.setVisible((not using_superres) and local_detector_mode == LOCAL_DETECTOR_OS_CFAR)
        detector_label = get_local_detector_label() if not using_superres else get_delay_estimator_label()
        detector_state = cfar_enabled if not using_superres else True
        self.btn_cfar.setText(f"{detector_label}: {'ON' if detector_state else 'OFF'}")
        self.btn_cfar.setStyleSheet("QPushButton:checked { background-color: lightgreen; }")
        if changed_windows:
            reset_processing_state()

    def on_local_detector_changed(self, _idx):
        global local_detector_mode
        selected = self.combo_local_detector.currentData()
        if selected not in {LOCAL_DETECTOR_CLEAN, LOCAL_DETECTOR_OS_CFAR}:
            selected = LOCAL_DETECTOR_OS_CFAR
        if local_detector_mode == selected:
            return
        local_detector_mode = selected
        self.refresh_detector_controls(force=True)
        self._force_ui_refresh = True
        self._last_rendered_display_key = None
        self._last_top_targets_key = None
        reset_processing_state()
        print(f"Local detector switched to: {get_local_detector_label()}")

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

    def apply_delay_doppler_display_db_range(self):
        global DISPLAY_RD_DB_MIN, DISPLAY_RD_DB_MAX
        try:
            DISPLAY_RD_DB_MIN, DISPLAY_RD_DB_MAX = sanitize_display_db_range(
                self.txt_delay_doppler_db_min.text(),
                self.txt_delay_doppler_db_max.text(),
            )
            self.txt_delay_doppler_db_min.setText(f"{DISPLAY_RD_DB_MIN:.1f}")
            self.txt_delay_doppler_db_max.setText(f"{DISPLAY_RD_DB_MAX:.1f}")
            levels = get_delay_doppler_display_db_range()
            self.rd_img.setLevels(levels)
            self.rd_colorbar.setLevels(values=levels)
            self._force_ui_refresh = True
            self._last_rendered_display_key = None
            print(f"Delay Doppler dB range set to: {DISPLAY_RD_DB_MIN:.1f} to {DISPLAY_RD_DB_MAX:.1f}")
        except ValueError:
            print("Invalid Delay Doppler dB range")

    def apply_micro_doppler_display_db_range(self):
        global DISPLAY_MD_DB_MIN, DISPLAY_MD_DB_MAX
        try:
            DISPLAY_MD_DB_MIN, DISPLAY_MD_DB_MAX = sanitize_display_db_range(
                self.txt_micro_doppler_db_min.text(),
                self.txt_micro_doppler_db_max.text(),
            )
            self.txt_micro_doppler_db_min.setText(f"{DISPLAY_MD_DB_MIN:.1f}")
            self.txt_micro_doppler_db_max.setText(f"{DISPLAY_MD_DB_MAX:.1f}")
            levels = get_micro_doppler_display_db_range()
            self.md_img.setLevels(levels)
            self.md_colorbar.setLevels(values=levels)
            self._force_ui_refresh = True
            self._last_rendered_display_key = None
            print(f"MicroDoppler dB range set to: {DISPLAY_MD_DB_MIN:.1f} to {DISPLAY_MD_DB_MAX:.1f}")
        except ValueError:
            print("Invalid MicroDoppler dB range")

    def toggle_micro_doppler(self):
        global show_micro_doppler
        show_micro_doppler = self.btn_md.isChecked()
        self.btn_md.setText(f"Micro-Doppler: {'ON' if show_micro_doppler else 'OFF'}")

    def toggle_cfar(self):
        global cfar_enabled
        cfar_enabled = self.btn_cfar.isChecked()
        self.refresh_detector_controls(force=True)
        self._force_ui_refresh = True
        self._last_rendered_display_key = None
        self._last_top_targets_key = None
        if not cfar_enabled:
            for ch in CHANNELS:
                if isinstance(ch.current_display_data, dict):
                    ch.current_display_data['cfar_points'] = np.empty((0, 2), dtype=np.int32)
                    ch.current_display_data['detected_points'] = np.empty((0, 2), dtype=np.int32)
                    ch.current_display_data['target_clusters'] = []
                    ch.current_display_data['detected_targets'] = []
            self.rd_cfar_marker.setData([], [])

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

    def apply_local_cfar_settings(self):
        global cfar_train_doppler, cfar_train_range
        global cfar_guard_doppler, cfar_guard_range
        global cfar_alpha_db, cfar_min_range_bin, cfar_dc_exclusion_bins, cfar_min_power_db
        global cfar_os_rank_percent, cfar_os_suppress_doppler, cfar_os_suppress_range
        try:
            cfar_os_rank_percent = float(np.clip(float(self.txt_local_cfar_rank.text()), 0.0, 100.0))
            cfar_train_doppler = max(0, int(self.txt_local_cfar_train_d.text()))
            cfar_train_range = max(0, int(self.txt_local_cfar_train_r.text()))
            cfar_guard_doppler = max(0, int(self.txt_local_cfar_guard_d.text()))
            cfar_guard_range = max(0, int(self.txt_local_cfar_guard_r.text()))
            cfar_os_suppress_doppler = max(0, int(self.txt_local_cfar_suppress_d.text()))
            cfar_os_suppress_range = max(0, int(self.txt_local_cfar_suppress_r.text()))
            cfar_alpha_db = float(self.txt_local_cfar_alpha.text())
            cfar_min_range_bin = max(0, int(self.txt_local_cfar_min_range.text()))
            cfar_dc_exclusion_bins = max(0, int(self.txt_local_cfar_dc_excl.text()))
            cfar_min_power_db = float(self.txt_local_cfar_min_power.text())

            self.txt_local_cfar_rank.setText(f"{cfar_os_rank_percent:.0f}")
            self.txt_local_cfar_train_d.setText(str(cfar_train_doppler))
            self.txt_local_cfar_train_r.setText(str(cfar_train_range))
            self.txt_local_cfar_guard_d.setText(str(cfar_guard_doppler))
            self.txt_local_cfar_guard_r.setText(str(cfar_guard_range))
            self.txt_local_cfar_suppress_d.setText(str(cfar_os_suppress_doppler))
            self.txt_local_cfar_suppress_r.setText(str(cfar_os_suppress_range))
            self.lbl_dbscan_eps.setText(f"D={get_dbscan_eps_doppler()} R={get_dbscan_eps_range()}")
            self.txt_local_cfar_alpha.setText(f"{cfar_alpha_db:.2f}")
            self.txt_local_cfar_min_range.setText(str(cfar_min_range_bin))
            self.txt_local_cfar_dc_excl.setText(str(cfar_dc_exclusion_bins))
            self.txt_local_cfar_min_power.setText(f"{cfar_min_power_db:.1f}")
            reset_processing_state()
            print(
                "Local OS-CFAR updated: "
                f"rank={cfar_os_rank_percent:.0f}% "
                f"train=({cfar_train_doppler},{cfar_train_range}) "
                f"guard=({cfar_guard_doppler},{cfar_guard_range}) "
                f"suppress=({cfar_os_suppress_doppler},{cfar_os_suppress_range}) "
                f"alpha_db={cfar_alpha_db:.2f} "
                f"min_range={cfar_min_range_bin} min_power_db={cfar_min_power_db:.1f} "
                f"dc_excl={cfar_dc_exclusion_bins}"
            )
        except ValueError:
            print("Invalid local OS-CFAR settings")

    def apply_target_cluster_settings(self):
        global target_dbscan_min_samples
        try:
            target_dbscan_min_samples = max(1, int(self.txt_cluster_min_samples.text()))
            self.lbl_dbscan_eps.setText(f"D={get_dbscan_eps_doppler()} R={get_dbscan_eps_range()}")
            self.txt_cluster_min_samples.setText(str(target_dbscan_min_samples))

            for ch in CHANNELS:
                if isinstance(ch.current_display_data, dict):
                    ch.current_display_data.pop('target_clusters', None)
            self._last_top_targets_key = None
            self._force_ui_refresh = True
            reset_processing_state()
            print(
                "DBSCAN settings updated: "
                f"eps_doppler={get_dbscan_eps_doppler()} "
                f"eps_range={get_dbscan_eps_range()} "
                f"min_samples={target_dbscan_min_samples}"
            )
        except ValueError:
            print("Invalid DBSCAN settings")

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
        detected_targets = latest_disp.get('detected_targets')
        if detected_targets is not None:
            return detected_targets
        clusters = latest_disp.get('target_clusters')
        if clusters is not None:
            return clusters

        rd_data = latest_disp.get('rd')
        detected_points = latest_disp.get('detected_points')
        if detected_points is None:
            detected_points = latest_disp.get('cfar_points')
        if rd_data is None or detected_points is None or len(detected_points) == 0:
            return []

        detector_mode = latest_disp.get('detector_mode', 'local_clean')
        if detector_mode == 'local_clean' or str(detector_mode).startswith("delay_superres_"):
            clusters = build_direct_targets(
                detected_points,
                rd_data,
                point_strengths_db=latest_disp.get('detected_strengths_db'),
            )
        else:
            clusters = cluster_detected_targets(detected_points, rd_data)
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

        valid_entries = valid_entries[:TARGET_SECTOR_POINT_LIMIT]
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
        detector_mode = latest_disp.get('detector_mode', 'local_clean')
        detector_summary = latest_disp.get('detector_summary') or {}
        detector_label = {
            'local_os_cfar': 'local OS-CFAR',
            'local_clean': 'local CLEAN',
        }.get(detector_mode, detector_summary.get('mode', detector_mode))
        have_auto_aoa = (
            len(CHANNELS) >= 2
            and self.phase_ready
            and self.synced_frame_id is not None
            and int(frame_id) == int(self.synced_frame_id)
        )

        header = f"Top Targets CH{ch.ch_id + 1} F{frame_id} [{scale_text}]"
        if detector_mode == 'local_clean':
            detector_config_text = (
                f"Detector={detector_label} | "
                f"Mode=direct targets | "
                f"List={TARGET_TEXT_POINT_LIMIT} Plot={TARGET_SECTOR_POINT_LIMIT}"
            )
        elif str(detector_mode).startswith("delay_superres_"):
            detector_config_text = (
                f"Detector={detector_label} | "
                f"Rows={int(detector_summary.get('rows_processed', 0))}/{int(detector_summary.get('rows_total', 0))} | "
                f"RowsDetected={int(detector_summary.get('rows_detected', 0))} | "
                f"Kmax={int(detector_summary.get('source_count_max', 0))} | "
                f"Thr={float(detector_summary.get('threshold_db', superres_peak_rel_threshold_db)):.1f}dB"
            )
        else:
            detector_config_text = (
                f"Detector={detector_label} | "
                f"OS-Suppress(D,R)=({cfar_os_suppress_doppler},{cfar_os_suppress_range}) | "
                f"DBSCAN(EpsD,EpsR,MinPts)=({get_dbscan_eps_doppler()},{get_dbscan_eps_range()},{target_dbscan_min_samples}) | "
                f"List={TARGET_TEXT_POINT_LIMIT} Plot={TARGET_SECTOR_POINT_LIMIT}"
            )
        lines = [
            header,
            detector_config_text,
        ]
        entries = []
        for rank, cluster in enumerate(clusters[:TARGET_TEXT_POINT_LIMIT], start=1):
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
                return d_idx, r_idx, "detector strongest target"

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
        if local_clean_disables_windows():
            self._enforce_clean_window_policy(announce=True)
            return
        enable_range_window = not enable_range_window
        self.update_toggle_style(self.btn_range_win, enable_range_window)
        reset_processing_state()

    def toggle_doppler_win(self):
        global enable_doppler_window
        if local_clean_disables_windows():
            self._enforce_clean_window_policy(announce=True)
            return
        enable_doppler_window = not enable_doppler_window
        self.update_toggle_style(self.btn_doppler_win, enable_doppler_window)
        reset_processing_state()

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
        self.refresh_detector_controls()

        # Update currently displayed channel
        ch = self._get_display_channel_runtime()
        if ch is None:
            self._set_status_label_text(self.lbl_sender, "Sender: N/A")
            self._set_status_label_text(self.lbl_params, "Params: N/A")
            self._set_status_label_text(self.lbl_buffer, "MD Buffer: N/A")
            self._set_status_label_text(self.lbl_cfar, "Detector: N/A")
            self.rd_click_marker.setData([], [])
            self.rd_cfar_marker.setData([], [])
            self._last_rendered_display_key = None
            self._last_top_targets_key = None
            self.update_top_targets_text(None, None)
        else:
            latest_disp = ch.current_display_data
            rd_data = latest_disp.get('rd')
            md_data = latest_disp.get('md')
            overlay_points = latest_disp.get('detected_points')
            if overlay_points is None:
                overlay_points = latest_disp.get('cfar_points')
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
                int(cfar_os_suppress_doppler),
                int(cfar_os_suppress_range),
                int(get_dbscan_eps_doppler()),
                int(get_dbscan_eps_range()),
                int(target_dbscan_min_samples),
                str(local_detector_mode),
                str(delay_estimator_mode),
                float(superres_peak_rel_threshold_db),
            )
            refresh_targets = top_targets_key != self._last_top_targets_key

            if rd_data is not None and redraw_display:
                self.rd_img.setImage(rd_data, autoLevels=False)
                rd_levels = latest_disp.get('rd_levels')
                if isinstance(rd_levels, (tuple, list)) and len(rd_levels) == 2:
                    low_level = float(rd_levels[0])
                    high_level = float(rd_levels[1])
                else:
                    low_level, high_level = get_delay_doppler_display_db_range()
                self.rd_img.setLevels((low_level, high_level))
                self.rd_colorbar.setLevels(values=(low_level, high_level))
                rows, cols = rd_data.shape  # rows=doppler, cols=range
                self.rd_doppler_min = -0.5 * float(rows)
                self.rd_doppler_span = float(rows)
                self.rd_img.setRect(QtCore.QRectF(self.rd_doppler_min, 0.0, self.rd_doppler_span, float(cols)))
                if overlay_points is not None and len(overlay_points) > 0:
                    x_pts = self.rd_doppler_min + overlay_points[:, 0].astype(np.float32)
                    y_pts = overlay_points[:, 1].astype(np.float32)
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
                md_low_level, md_high_level = get_micro_doppler_display_db_range()
                self.md_img.setLevels((md_low_level, md_high_level))
                self.md_colorbar.setLevels(values=(md_low_level, md_high_level))
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
                f"view={get_display_range_bin_limit()}x{get_display_doppler_bin_limit()} | "
                f"delay_est={get_delay_estimator_label()} thr={superres_peak_rel_threshold_db:.1f}dB"
            )
            if ch.last_param_error:
                params_text += f" | last_error={ch.last_param_error}"
            self._set_status_label_text(self.lbl_params, params_text)

            with ch.micro_lock:
                self._set_status_label_text(
                    self.lbl_buffer,
                    f"MD Buffer(CH{ch.ch_id + 1}): {len(ch.micro_doppler_buffer)}/{BUFFER_LENGTH}"
                )
            detector_mode = latest_disp.get('detector_mode', 'local_clean')
            detector_summary = latest_disp.get('detector_summary') or {}
            cfar_stats = latest_disp.get('cfar_stats') or {}
            if str(detector_mode).startswith("delay_superres_"):
                mode_label = str(detector_summary.get('mode', get_delay_estimator_label()))
                rows_processed = int(detector_summary.get('rows_processed', 0))
                rows_total = int(detector_summary.get('rows_total', 0))
                rows_detected = int(detector_summary.get('rows_detected', 0))
                source_count_max = int(detector_summary.get('source_count_max', 0))
                threshold_db = float(detector_summary.get('threshold_db', superres_peak_rel_threshold_db))
                compute_backend = str(detector_summary.get('compute_backend', 'cpu'))
                cpu_rows_fetched = int(detector_summary.get('cpu_rows_fetched', 0))
                rows_capped = int(detector_summary.get('rows_capped', 0))
                hit_count = 0 if overlay_points is None else int(len(overlay_points))
                cfar_status = (
                    f"Detector(CH{ch.ch_id + 1}): {mode_label} hits={hit_count} "
                    f"rows={rows_processed}/{rows_total} rows_detected={rows_detected} "
                    f"kmax={source_count_max} thr={threshold_db:.1f}dB "
                    f"subc={int(detector_summary.get('subcarrier_count', 0))} "
                    f"capped={rows_capped} "
                    f"cpu_rows={cpu_rows_fetched} "
                    f"backend={compute_backend}"
                )
                cfar_status_short = (
                    f"Detector(CH{ch.ch_id + 1}): {mode_label} "
                    f"h={hit_count} r={rows_processed}/{rows_total} "
                    f"k={source_count_max} t={threshold_db:.1f} "
                    f"cap={rows_capped} "
                    f"cpu={cpu_rows_fetched} {compute_backend}"
                )
            elif detector_mode == 'local_os_cfar':
                cfar_status = f"Detector(CH{ch.ch_id + 1}): OS-CFAR {'on' if cfar_enabled else 'off'}"
                if cfar_enabled:
                    cfar_status += (
                        f" raw={int(latest_disp.get('cfar_hits', 0))}"
                        f" shown={int(latest_disp.get('cfar_shown_hits', 0))}"
                        f" pmin={cfar_stats.get('power_min_db', cfar_min_power_db):.1f}dB"
                        f" rank={int(cfar_stats.get('os_rank_index', 0)) + 1}/{int(cfar_stats.get('training_cells', 0))}"
                        f" pct={cfar_stats.get('rank_percent', cfar_os_rank_percent):.0f}%"
                        f" supp=({int(cfar_stats.get('suppress_d', cfar_os_suppress_doppler))},{int(cfar_stats.get('suppress_r', cfar_os_suppress_range))})"
                        f" invalid={int(cfar_stats.get('invalid_cells', 0))}"
                        f" noise=[{cfar_stats.get('noise_min', 0.0):.2e},{cfar_stats.get('noise_max', 0.0):.2e}]"
                    )
                cfar_status_short = f"Detector(CH{ch.ch_id + 1}): OS-CFAR {'on' if cfar_enabled else 'off'}"
                if cfar_enabled:
                    cfar_status_short += (
                        f" raw={int(latest_disp.get('cfar_hits', 0))}"
                        f" sh={int(latest_disp.get('cfar_shown_hits', 0))}"
                        f" k={int(cfar_stats.get('os_rank_index', 0)) + 1}"
                        f" s={int(cfar_stats.get('suppress_d', cfar_os_suppress_doppler))}/{int(cfar_stats.get('suppress_r', cfar_os_suppress_range))}"
                    )
            else:
                cfar_status = f"Detector(CH{ch.ch_id + 1}): CLEAN {'on' if cfar_enabled else 'off'}"
                if cfar_enabled:
                    cfar_status += (
                        f" raw={int(latest_disp.get('cfar_hits', 0))}"
                        f" shown={int(latest_disp.get('cfar_shown_hits', 0))}"
                        f" pmin={cfar_stats.get('power_min_db', clean_min_power_db):.1f}dB"
                        f" gain={cfar_stats.get('loop_gain', clean_loop_gain):.2f}"
                        f" psf={int(cfar_stats.get('psf_rows', 0))}x{int(cfar_stats.get('psf_cols', 0))}"
                        f" stop={cfar_stats.get('stop_reason', 'n/a')}"
                        f" resid={cfar_stats.get('residual_peak_db', 0.0):.1f}dB"
                    )
                cfar_status_short = f"Detector(CH{ch.ch_id + 1}): CLEAN {'on' if cfar_enabled else 'off'}"
                if cfar_enabled:
                    cfar_status_short += (
                        f" raw={int(latest_disp.get('cfar_hits', 0))}"
                        f" sh={int(latest_disp.get('cfar_shown_hits', 0))}"
                        f" p={cfar_stats.get('power_min_db', clean_min_power_db):.0f}"
                        f" g={cfar_stats.get('loop_gain', clean_loop_gain):.2f}"
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
