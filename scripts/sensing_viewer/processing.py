from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from sensing_runtime_protocol import ViewerRuntimeParams

from .array_backend import stft, to_cpu_array
from .config import (
    BUFFER_LENGTH,
    DOPPLER_FFT_SIZE,
    MAX_DOPPLER_BINS,
    MAX_RANGE_BIN,
    MICRO_DOPPLER_STFT_NFFT,
    MICRO_DOPPLER_STFT_NOVERLAP,
    MICRO_DOPPLER_STFT_NPERSEG,
    RANGE_FFT_SIZE,
)


@dataclass(frozen=True)
class ProcessingOptions:
    range_fft_size: int = RANGE_FFT_SIZE
    doppler_fft_size: int = DOPPLER_FFT_SIZE
    display_range_bins: int = MAX_RANGE_BIN
    display_doppler_bins: int = MAX_DOPPLER_BINS
    enable_range_window: bool = True
    enable_doppler_window: bool = True

    def with_display_range(self, value: int) -> "ProcessingOptions":
        return replace(self, display_range_bins=max(1, int(value)))


@dataclass
class RangeDopplerResult:
    magnitude_db: np.ndarray
    range_time: np.ndarray | None
    rd_complex: np.ndarray


def backend_stream_unsupported(params: ViewerRuntimeParams) -> bool:
    return params.backend_processing() or params.metadata_sidecar()


def _hamming_window_row(length: int) -> np.ndarray:
    return np.hamming(max(1, int(length))).reshape(1, max(1, int(length))).astype(np.float32)


def _hamming_window_col(length: int) -> np.ndarray:
    return np.hamming(max(1, int(length))).reshape(max(1, int(length)), 1).astype(np.float32)


def _amplitude_norm(raw_rows: int, raw_cols: int, range_window_active: bool, doppler_window_active: bool) -> float:
    norm = float(np.sqrt(max(1, raw_rows) * max(1, raw_cols)))
    if range_window_active:
        norm *= float(np.mean(np.hamming(max(1, raw_cols))))
    if doppler_window_active:
        norm *= float(np.mean(np.hamming(max(1, raw_rows))))
    return max(norm, 1e-12)


def process_range_doppler(
    frame_data,
    viewer_params: ViewerRuntimeParams,
    options: ProcessingOptions | None = None,
    *,
    local_clean_disables_windows: bool = False,
) -> RangeDopplerResult:
    if backend_stream_unsupported(viewer_params):
        raise ValueError("Backend sensing output is not supported in fast sensing viewers")
    opts = options or ProcessingOptions()

    if viewer_params.is_dense_range_doppler():
        rd_complex = np.asarray(
            frame_data[:viewer_params.wire_rows, :viewer_params.wire_cols],
            dtype=np.complex64,
        )
        rd_shifted = np.fft.fftshift(rd_complex, axes=0).astype(np.complex64, copy=False)
        magnitude_db = 20.0 * np.log10(np.abs(rd_shifted) + 1e-12)
        return RangeDopplerResult(magnitude_db.astype(np.float32, copy=False), None, rd_shifted)

    if not viewer_params.raw_fft_locally_supported():
        raise ValueError(f"Unsupported sensing frame format: {viewer_params.describe()}")

    raw_rows = max(1, int(viewer_params.active_rows))
    raw_cols = max(1, int(viewer_params.active_cols))
    range_fft_size = max(raw_cols, int(opts.range_fft_size))
    doppler_fft_size = max(raw_rows, int(opts.doppler_fft_size))
    max_view_range_bins = min(max(1, int(opts.display_range_bins)), range_fft_size)
    raw_frame = np.asarray(frame_data[:raw_rows, :raw_cols], dtype=np.complex64)

    range_window_active = bool(opts.enable_range_window and not local_clean_disables_windows)
    doppler_window_active = bool(opts.enable_doppler_window and not local_clean_disables_windows)
    range_win = _hamming_window_row(raw_cols) if range_window_active else np.ones((1, raw_cols), dtype=np.float32)
    doppler_win = _hamming_window_col(raw_rows) if doppler_window_active else np.ones((raw_rows, 1), dtype=np.float32)

    windowed_data = raw_frame * range_win
    padded_data = np.zeros((raw_rows, range_fft_size), dtype=np.complex64)
    padded_data[:, :raw_cols] = windowed_data
    range_time = np.fft.ifft(padded_data, axis=1) * range_fft_size
    range_time_view = range_time[:, :max_view_range_bins] if max_view_range_bins < range_fft_size else range_time

    doppler_windowed = range_time_view * doppler_win
    padded_doppler = np.zeros((doppler_fft_size, range_time_view.shape[1]), dtype=np.complex64)
    padded_doppler[:raw_rows, :] = doppler_windowed
    doppler_shifted = np.fft.fftshift(np.fft.fft(padded_doppler, axis=0), axes=0)

    norm = _amplitude_norm(raw_rows, raw_cols, range_window_active, doppler_window_active)
    magnitude_db = 20.0 * np.log10(np.abs(doppler_shifted) / norm + 1e-12)
    return RangeDopplerResult(
        magnitude_db.astype(np.float32, copy=False),
        range_time_view.astype(np.complex64, copy=False),
        doppler_shifted.astype(np.complex64, copy=False),
    )


def process_range_doppler_batch(
    channel_frames: list[tuple[int, np.ndarray]],
    viewer_params: ViewerRuntimeParams,
    options: ProcessingOptions | None = None,
) -> dict[int, RangeDopplerResult]:
    return {
        int(ch_idx): process_range_doppler(frame, viewer_params, options)
        for ch_idx, frame in channel_frames
    }


class MicroDopplerBuffer:
    def __init__(self, maxlen: int = BUFFER_LENGTH) -> None:
        self._buffer: deque[np.complex64] = deque(maxlen=max(1, int(maxlen)))

    def __len__(self) -> int:
        return len(self._buffer)

    def extend_range_bin(self, range_time, range_bin: int) -> None:
        if range_time is None:
            return
        data = to_cpu_array(range_time, dtype=np.complex64)
        if data.size == 0:
            return
        idx = min(max(0, int(range_bin)), data.shape[1] - 1)
        self._buffer.extend(data[:, idx])

    def spectrum(self):
        if len(self._buffer) < MICRO_DOPPLER_STFT_NPERSEG:
            return None
        complex_signal = np.asarray(self._buffer, dtype=np.complex64)
        f, t, zxx = stft(
            complex_signal,
            fs=1.0,
            window="hamming",
            nperseg=MICRO_DOPPLER_STFT_NPERSEG,
            noverlap=MICRO_DOPPLER_STFT_NOVERLAP,
            nfft=MICRO_DOPPLER_STFT_NFFT,
            return_onesided=False,
        )
        power_db = 20.0 * np.log10(np.abs(zxx) + 1e-12)
        power_db_shifted = np.fft.fftshift(power_db, axes=0)
        f_shifted = np.fft.fftshift(f)
        f_idx = (f_shifted > -0.5) & (f_shifted < 0.5)
        return f_shifted[f_idx], t, power_db_shifted[f_idx, :]
