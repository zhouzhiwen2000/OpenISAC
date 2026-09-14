"""Physical coordinates for uniformly sampled OFDM sensing spectra."""
from __future__ import annotations

from dataclasses import dataclass
import math

from sensing_runtime_protocol import FLAG_BISTATIC, PARAMS_VERSION, ViewerRuntimeParams

C_LIGHT_MPS = 299_792_458.0


@dataclass(frozen=True)
class PhysicalAxes:
    slow_time_rate_hz: float
    doppler_bin_hz: float
    delay_bin_ns: float
    bistatic: bool

    @property
    def range_unit(self) -> str:
        return "ns" if self.bistatic else "m"

    @property
    def range_label(self) -> str:
        return "Relative Delay (ns)" if self.bistatic else "Range (m)"

    @property
    def range_bin_scale(self) -> float:
        return self.delay_bin_ns if self.bistatic else self.delay_bin_ns * 1e-9 * C_LIGHT_MPS / 2.0


def physical_axes(params: ViewerRuntimeParams, range_fft_size: int, doppler_fft_size: int) -> PhysicalAxes:
    """Use transform lengths, never cropped image dimensions, for bin spacing.

    Delay is relative to the current RX alignment/synchronization reference.
    A nonuniform compact frequency grid has no single physical delay scale.
    """
    if params.version != PARAMS_VERSION:
        raise ValueError(f"Physical axes require sensing PARM V{PARAMS_VERSION}")
    if (not math.isfinite(params.sample_rate_hz) or params.sample_rate_hz <= 0
            or params.ofdm_fft_size <= 0 or params.cp_length < 0
            or params.sensing_symbol_stride <= 0 or range_fft_size <= 0 or doppler_fft_size <= 0):
        raise ValueError("Invalid sensing sampling geometry")
    slow_rate = params.sample_rate_hz / ((params.ofdm_fft_size + params.cp_length) * params.sensing_symbol_stride)
    delay_ns = (1e9 * params.ofdm_fft_size
                / (params.sample_rate_hz * params.sensing_subcarrier_stride * range_fft_size)
                if params.sensing_subcarrier_stride > 0 else math.nan)
    return PhysicalAxes(slow_rate, slow_rate / doppler_fft_size, delay_ns, bool(params.flags & FLAG_BISTATIC))


def display_doppler_mapping(source_rows: int, display_bins: int, displayed_rows: int,
                            downsample: int, bin_hz: float) -> tuple[float, float]:
    """Map the viewer's centered image coordinates to Hz after crop/decimation."""
    step = max(1, int(downsample))
    start = max(0, source_rows // 2 - max(1, display_bins) // 2)
    # Existing image/marker coordinates run from -displayed_rows/2 upward.
    offset_bins = start - source_rows // 2 + displayed_rows * step / 2.0
    return step * bin_hz, offset_bins * bin_hz


def image_bounds_from_centers(extent, rows: int, cols: int) -> tuple[float, float, float, float]:
    """Convert first/last sample coordinates to ImageItem pixel-edge bounds."""
    x0, x1, y0, y1 = (float(v) for v in extent)
    dx = (x1 - x0) / max(1, cols - 1)
    dy = (y1 - y0) / max(1, rows - 1)
    return x0 - dx / 2, y0 - dy / 2, max(dx, 1e-12) * cols, max(dy, 1e-12) * rows
