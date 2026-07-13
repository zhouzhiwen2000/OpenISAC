"""Visualize the UE uplink self-ZC scan spectrum.

The UE publishes a full-frame matched-filter spectrum of its own uplink ZC
(one complex value per candidate ZC-symbol start position in the RX frame).
Unlike the fixed-window self-channel estimator, this locates the self-ZC no
matter how far the TX-vs-RX window has drifted, so a large post-underflow /
post-resync offset shows up as the correlation peak moving away from the
expected uplink-window position.

Enable it on the UE with:
    uplink.debug_self_channel: true
    uplink.debug_self_scan_spectrum: true
and (optionally) network_output.self_scan_ip / self_scan_port (default 12362).

Pass --expected <sample> to draw the expected ZC position marker; the gap
between the peak (solid red) and the expected marker (dashed green) is the
TX-vs-RX window offset in samples. Use --frame-samples to scale the x-axis to
the full RX frame. By default the main plot auto-zooms around the detected peak
and the top overview bar shows that moving slice, similar to an oscilloscope
zoom bar. Use --slice START:END to force a fixed sample range.
"""

from __future__ import annotations

import argparse

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore
from PyQt6 import QtWidgets

try:
    from qt_debug_plot_common import (
        PLOT_BACKGROUND,
        PLOT_FOREGROUND,
        DebugPlotWindow,
        ViewerArgs,
        configure_plot,
        power_db,
    )
except ImportError:
    from scripts.qt_debug_plot_common import (
        PLOT_BACKGROUND,
        PLOT_FOREGROUND,
        DebugPlotWindow,
        ViewerArgs,
        configure_plot,
        power_db,
    )


DEFAULT_PORT = 12362


def _parse_slice(text: str | None) -> tuple[int, int] | None:
    if text is None:
        return None
    parts = text.split(":", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("slice must be START:END")
    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("slice bounds must be integers") from exc
    if start < 0 or end <= start:
        raise argparse.ArgumentTypeError("slice must satisfy 0 <= START < END")
    return start, end


class SelfScanWindow(DebugPlotWindow):
    def __init__(
        self,
        host: str,
        port: int,
        interval_ms: int,
        expected: int | None,
        frame_samples: int | None,
        view_slice: tuple[int, int] | None,
        auto_slice: bool,
        auto_slice_samples: int,
    ) -> None:
        self.expected = expected
        self.frame_samples = frame_samples
        self.view_slice = view_slice
        self.auto_slice = auto_slice
        self.auto_slice_samples = max(1, auto_slice_samples)
        self.current_slice: tuple[int, int] | None = None
        super().__init__(
            title="Uplink Self-ZC Scan Spectrum",
            default_port=port,
            host=host,
            interval_ms=interval_ms,
        )
        self.overview = pg.PlotWidget()
        configure_plot(
            self.overview,
            "Frame Slice Overview",
            "",
            "Samples into RX frame",
        )
        self.overview.setFixedHeight(92)
        self.overview.setMouseEnabled(x=False, y=False)
        self.overview.hideAxis("left")
        self.overview.setYRange(0, 1, padding=0)
        self.overview_curve = self.overview.plot(
            [0, 1],
            [0.5, 0.5],
            pen=pg.mkPen("#64748b", width=6),
        )
        self.slice_region = pg.LinearRegionItem(
            values=(0, 1),
            orientation=pg.LinearRegionItem.Vertical,
            movable=False,
            brush=pg.mkBrush(37, 99, 235, 70),
            pen=pg.mkPen("#2563eb", width=1.0),
        )
        self.overview.addItem(self.slice_region)
        self.overview_peak_line = pg.InfiniteLine(
            pos=0, angle=90, pen=pg.mkPen("#dc2626", width=1.0)
        )
        self.overview.addItem(self.overview_peak_line)
        self.overview_expected_line = pg.InfiniteLine(
            pos=0,
            angle=90,
            pen=pg.mkPen("#16a34a", width=1.0, style=QtCore.Qt.PenStyle.DashLine),
        )
        if self.expected is not None:
            self.overview_expected_line.setPos(self.expected)
            self.overview.addItem(self.overview_expected_line)
        self.plot_layout.addWidget(self.overview)

        self.plot = pg.PlotWidget()
        configure_plot(
            self.plot,
            "Uplink Self-ZC Matched-Filter Spectrum",
            "Power (dB)",
            "ZC symbol start (samples into RX frame)",
        )
        self.curve = self.plot.plot([0], [0], pen=pg.mkPen("#2563eb", width=1.2))

        # Measured peak (solid red) and expected uplink-window position (dashed green).
        self.peak_line = pg.InfiniteLine(
            pos=0, angle=90, pen=pg.mkPen("#dc2626", width=1.2)
        )
        self.plot.addItem(self.peak_line)
        self.expected_line = pg.InfiniteLine(
            pos=0,
            angle=90,
            pen=pg.mkPen("#16a34a", width=1.2, style=QtCore.Qt.PenStyle.DashLine),
        )
        if self.expected is not None:
            self.expected_line.setPos(self.expected)
            self.plot.addItem(self.expected_line)

        self.peak_label = pg.TextItem(
            "",
            anchor=(0, 1),
            color=PLOT_FOREGROUND,
            fill=pg.mkBrush(PLOT_BACKGROUND),
            border=pg.mkPen("#64748b"),
        )
        self.plot.addItem(self.peak_label)
        self.plot_layout.addWidget(self.plot)

    def _frame_axis_len(self, n: int) -> int:
        if self.frame_samples is not None and self.frame_samples > 0:
            return max(self.frame_samples, n)
        return n

    def _visible_span(self, n: int, peak_index: int | None = None) -> tuple[int, int]:
        frame_len = self._frame_axis_len(n)
        if self.view_slice is None:
            if self.auto_slice and peak_index is not None:
                width = min(self.auto_slice_samples, frame_len)
                start = int(peak_index) - width // 2
                start = min(max(0, start), max(0, frame_len - width))
                return start, start + width
            return 0, frame_len
        start, end = self.view_slice
        start = min(max(0, start), max(0, frame_len - 1))
        end = min(max(end, start + 1), frame_len)
        return start, end

    def _update_overview(self, n: int, peak_index: int) -> None:
        frame_len = self._frame_axis_len(n)
        start, end = self._visible_span(n, peak_index)
        self.current_slice = (start, end)
        self.overview_curve.setData([0, frame_len], [0.5, 0.5])
        self.slice_region.setRegion((start, end))
        self.overview_peak_line.setPos(peak_index)
        self.overview.setXRange(0, max(1, frame_len), padding=0)

    def handle_message(self, data: bytes) -> None:
        itemsize = np.dtype(np.complex64).itemsize
        if len(data) < itemsize or (len(data) % itemsize) != 0:
            self.status_label.setText(f"Ignored {len(data)} B frame (not complex64)")
            return

        cdata = np.frombuffer(data, dtype=np.complex64)
        power = power_db(cdata)
        n = power.size
        x = np.arange(n)

        peak_index = int(np.argmax(power))
        peak_value = float(power[peak_index])
        avg_value = float(np.mean(power))
        peak_over_avg = peak_value - avg_value  # both already in dB

        self.curve.setData(x, power)
        self.peak_line.setPos(peak_index)
        self._update_overview(n, peak_index)

        label = f"Peak {peak_value:.1f} dB @ {peak_index}\npeak-avg {peak_over_avg:.1f} dB"
        if self.expected is not None:
            offset = peak_index - self.expected
            frame_len = self._frame_axis_len(n)
            if offset > frame_len // 2:
                offset -= frame_len
            elif offset < -(frame_len // 2):
                offset += frame_len
            label += f"\nexpected {self.expected}\noffset {offset:+d} samples"
        self.peak_label.setText(label)
        self.peak_label.setPos(peak_index, peak_value)
        x_start, x_end = self.current_slice or self._visible_span(n, peak_index)
        self.plot.setXRange(x_start, max(x_start + 1, x_end), padding=0)
        self.plot.setYRange(avg_value - 5.0, peak_value + 5.0, padding=0)
        view_text = f"view {x_start}:{x_end}"
        if self.frame_samples is not None:
            view_text += f" of frame {self._frame_axis_len(n)}"
        if self.view_slice is None and self.auto_slice:
            view_text += " auto"
        self.status_label.setText(
            f"{n} taps | {view_text} | peak {peak_value:.1f} dB @ {peak_index} | "
            f"peak-avg {peak_over_avg:.1f} dB"
        )


def _parse_args() -> tuple[
    ViewerArgs,
    int | None,
    int | None,
    tuple[int, int] | None,
    bool,
    int,
]:
    parser = argparse.ArgumentParser(description="Plot the uplink self-ZC scan spectrum.")
    parser.add_argument("--host", default="127.0.0.1", help="Backend host or tcp:// endpoint")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Backend ZMQ port")
    parser.add_argument("--interval-ms", type=int, default=50, help="Plot refresh interval")
    parser.add_argument(
        "--frame-samples",
        type=int,
        default=None,
        help="Full RX frame length in samples; extends the x-axis beyond published scan taps",
    )
    parser.add_argument(
        "--slice",
        type=_parse_slice,
        default=None,
        metavar="START:END",
        help="Only show this fixed sample range on the main plot; disables auto-slice",
    )
    parser.add_argument(
        "--auto-slice-samples",
        type=int,
        default=8192,
        help="Auto-zoom width in samples when --slice is not set",
    )
    parser.add_argument(
        "--no-auto-slice",
        action="store_true",
        help="Show the full frame/range instead of auto-following the detected peak",
    )
    parser.add_argument(
        "--expected",
        type=int,
        default=None,
        help="Expected uplink-window ZC start (samples) to mark for offset readout",
    )
    args = parser.parse_args()
    return (
        ViewerArgs(args.host, args.port, max(1, args.interval_ms)),
        args.expected,
        args.frame_samples,
        args.slice,
        args.slice is None and not args.no_auto_slice,
        max(1, args.auto_slice_samples),
    )


def main() -> None:
    args, expected, frame_samples, view_slice, auto_slice, auto_slice_samples = _parse_args()
    app = QtWidgets.QApplication([])
    window = SelfScanWindow(
        args.host,
        args.port,
        args.interval_ms,
        expected,
        frame_samples,
        view_slice,
        auto_slice,
        auto_slice_samples,
    )
    window.resize(1160, 720)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
