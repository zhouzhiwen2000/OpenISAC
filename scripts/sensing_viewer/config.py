from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sensing_runtime_protocol import ViewerRuntimeParams


FFT_SIZE = 1024
NUM_SYMBOLS = 100
MAX_RANGE_BIN = 1000
MAX_DOPPLER_BINS = 1000
RANGE_FFT_SIZE = 10240
DOPPLER_FFT_SIZE = 1000
DEFAULT_DISPLAY_DB_MIN = -80.0
DEFAULT_DISPLAY_DB_MAX = 0.0
BUFFER_LENGTH = 5000
MICRO_DOPPLER_STFT_NPERSEG = 256
MICRO_DOPPLER_STFT_NOVERLAP = 192
MICRO_DOPPLER_STFT_NFFT = 256


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


@dataclass(frozen=True)
class FastViewerConfig:
    mode: str
    settings_key: str
    title: str
    default_port: int
    default_control_port: int
    default_channels: int
    display_downsample: int
    supports_aggregate_stream: bool
    supports_phase_calibration: bool
    supports_gain_control: bool
    supports_superresolution: bool
    supports_os_cfar: bool


def default_config(mode: str) -> FastViewerConfig:
    normalized = (mode or "").strip().lower()
    if normalized == "bi":
        return FastViewerConfig(
            mode="bi",
            settings_key="plot_bi_sensing_fast",
            title="OpenISAC Bi-Sensing - PyQtGraph",
            default_port=8889,
            default_control_port=10001,
            default_channels=1,
            display_downsample=2,
            supports_aggregate_stream=False,
            supports_phase_calibration=False,
            supports_gain_control=False,
            supports_superresolution=False,
            supports_os_cfar=True,
        )
    if normalized == "mono":
        return FastViewerConfig(
            mode="mono",
            settings_key="plot_sensing_fast",
            title="OpenISAC Sensing - PyQtGraph (Multi-Channel)",
            default_port=8888,
            default_control_port=9999,
            default_channels=2,
            display_downsample=1,
            supports_aggregate_stream=True,
            supports_phase_calibration=True,
            supports_gain_control=True,
            supports_superresolution=True,
            supports_os_cfar=True,
        )
    raise ValueError(f"Unsupported fast sensing viewer mode: {mode!r}")


def sanitize_display_db_range(low_db, high_db) -> tuple[float, float]:
    try:
        low = float(low_db)
        high = float(high_db)
    except (TypeError, ValueError):
        return DEFAULT_DISPLAY_DB_MIN, DEFAULT_DISPLAY_DB_MAX
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        return DEFAULT_DISPLAY_DB_MIN, DEFAULT_DISPLAY_DB_MAX
    return low, high
