from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys
import threading
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BackendInfo:
    name: str
    use_nvidia_gpu: bool
    use_apple_gpu: bool
    use_intel_gpu: bool

    @property
    def use_gpu(self) -> bool:
        return self.use_nvidia_gpu or self.use_apple_gpu or self.use_intel_gpu


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


def _backend_allowed(name: str) -> bool:
    if name == "mlx":
        return FORCED_BACKEND in ("auto", "apple", "mlx")
    return FORCED_BACKEND in ("auto", name)


def _probe_mlx_backend(timeout_s: float = 5.0) -> tuple[bool, str]:
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
    detail = stderr[-1] if stderr else f"probe exited with code {result.returncode}"
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
        try:
            import dpctl
            import dpnp as _dpnp

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
njit = None
prange = range
try:
    from numba import njit as _njit
    from numba import prange as _prange

    HAVE_NUMBA = True
    njit = _njit
    prange = _prange
    print("Backend: Numba available for CPU acceleration")
except ImportError:
    print("Backend: Numba not available")


def to_numpy(arr):
    if hasattr(arr, "get"):
        return arr.get()
    if hasattr(arr, "asnumpy"):
        return arr.asnumpy()
    if hasattr(arr, "__array__"):
        return np.asarray(arr)
    return arr


def get_array_module(arr):
    module_root = type(arr).__module__.split(".", 1)[0]
    if module_root in {"cupy", "dpnp"}:
        if cp is not None:
            return cp
        return importlib.import_module(module_root)
    return np


def scalar_to_bool(value) -> bool:
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def scalar_to_int(value) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def scalar_to_float(value) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def to_cpu_array(arr, dtype=None):
    host = to_numpy(arr)
    if dtype is None:
        return np.asarray(host)
    return np.asarray(host, dtype=dtype)


try:
    from scipy.signal import stft

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

    def stft(x, fs=1.0, window="hamming", nperseg=256, noverlap=192, nfft=256, return_onesided=False):
        x = np.asarray(x)
        if x.ndim != 1:
            x = x.ravel()
        step = nperseg - noverlap
        if isinstance(window, str):
            w = np.hamming(nperseg).astype(np.float32) if window.lower() == "hamming" else np.ones(nperseg, dtype=np.float32)
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


BACKEND = BackendInfo(
    name=BACKEND_NAME,
    use_nvidia_gpu=USE_NVIDIA_GPU,
    use_apple_gpu=USE_APPLE_GPU,
    use_intel_gpu=USE_INTEL_GPU,
)
