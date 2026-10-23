"""Axis labels preserve the viewers' image, detector and click coordinates."""
from __future__ import annotations

import math

import pyqtgraph as pg

from sensing_physical_axes import display_doppler_mapping, doppler_display_units, physical_axes


class ScaledAxisItem(pg.AxisItem):
    """Apply an affine unit conversion only to the displayed tick labels."""

    def __init__(self, orientation: str):
        super().__init__(orientation=orientation)
        self.enableAutoSIPrefix(False)
        self.factor = math.nan
        self.offset = 0.0

    def set_mapping(self, factor: float, offset: float = 0.0) -> None:
        if factor == self.factor and offset == self.offset:
            return
        self.factor, self.offset = factor, offset
        self.picture = None
        self.update()

    def tickStrings(self, values, scale, spacing):
        if not math.isfinite(self.factor):
            return ["" for _ in values]
        return [f"{value * self.factor + self.offset:.6g}" for value in values]


class SensingPlotAxes:
    def __init__(self):
        self.range = ScaledAxisItem("left")
        self.doppler = ScaledAxisItem("bottom")
        self.micro_doppler = ScaledAxisItem("left")
        self.time = ScaledAxisItem("bottom")

    def update(self, rd_plot, md_plot, params, *, range_fft_size, doppler_fft_size,
               source_rows, display_bins, displayed_rows, downsample, range_downsample=None, velocity_carrier_hz=None):
        axes = physical_axes(params, range_fft_size, doppler_fft_size)
        self.range.set_mapping(axes.range_bin_scale * (downsample if range_downsample is None else range_downsample))
        velocity_scale, unit = doppler_display_units(velocity_carrier_hz, bistatic=axes.bistatic)
        factor, offset = display_doppler_mapping(
            source_rows, display_bins, displayed_rows, downsample, axes.doppler_bin_hz)
        self.doppler.set_mapping(factor * velocity_scale, offset * velocity_scale)
        # Both STFT implementations use fs=1: frequency is cycles/sample and time is samples.
        self.micro_doppler.set_mapping(axes.slow_time_rate_hz * velocity_scale)
        self.time.set_mapping(1.0 / axes.slow_time_rate_hz)
        rd_plot.setLabel("left", axes.range_label if math.isfinite(axes.range_bin_scale)
                         else "Delay unavailable (nonuniform subcarriers)")
        label = "Radial Velocity (m/s)" if unit == "m/s" else "Doppler Frequency (Hz)"
        rd_plot.setLabel("bottom", label)
        md_plot.setLabel("left", label)
        md_plot.setLabel("bottom", "Time (s)")
        return axes
