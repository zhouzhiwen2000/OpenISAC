"""Visualize the UE uplink self-ZC scan spectrum (slice + metadata).

The UE runs a full-frame matched-filter search for its own uplink ZC, then
publishes only a peak-centered correlation slice plus a fixed metadata header.
This viewer is pure display: it does not choose slices or recompute peaks.

Enable it on the UE with:
    uplink.debug_self_channel: true
    uplink.debug_self_scan_spectrum: true
and (optionally) network_output.self_scan_ip / self_scan_port (default 12352).

Wire format (single ZMQ message, little-endian):
    [64-byte header magic "ULSCSCAN"][complex64 * slice_len]
"""

from __future__ import annotations

import argparse
import struct

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
        normalize_endpoint,
        power_db,
    )
    from viewer_endpoint_store import default_settings_key, load_endpoint
except ImportError:
    from scripts.qt_debug_plot_common import (
        PLOT_BACKGROUND,
        PLOT_FOREGROUND,
        DebugPlotWindow,
        ViewerArgs,
        configure_plot,
        normalize_endpoint,
        power_db,
    )
    from scripts.viewer_endpoint_store import default_settings_key, load_endpoint


DEFAULT_PORT = 12352
MAGIC = b"ULSCSCAN"
HEADER_SIZE = 64
# magic(8) + version/u32*7 + i32 + f32*2 + i32*2 + u64 = 8+28+4+8+8+8 = 64
HEADER_STRUCT = struct.Struct("<8s7Ii2f2iQ")
assert HEADER_STRUCT.size == HEADER_SIZE


def parse_self_scan_frame(data: bytes) -> tuple[dict, np.ndarray]:
    if len(data) < HEADER_SIZE:
        raise ValueError(f"frame too short for header: {len(data)} B")
    (
        magic,
        version,
        frame_samples,
        corr_len,
        slice_start,
        slice_len,
        peak_index,
        expected_zc_pos,
        offset_samples,
        peak_power,
        avg_power,
        ul_tx_rx_shift,
        timing_advance,
        generation,
    ) = HEADER_STRUCT.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError(f"bad self-scan magic: {magic!r}")
    if version != 1:
        raise ValueError(f"unsupported self-scan version: {version}")
    if slice_len == 0:
        raise ValueError("empty self-scan slice")

    payload = memoryview(data)[HEADER_SIZE:]
    expected_bytes = slice_len * np.dtype(np.complex64).itemsize
    if len(payload) < expected_bytes:
        raise ValueError(
            f"payload too short: {len(payload)} B, need {expected_bytes} B for slice_len={slice_len}"
        )
    if len(payload) != expected_bytes:
        # Tolerate trailing padding, but require at least the declared samples.
        payload = payload[:expected_bytes]
    spectrum = np.frombuffer(payload, dtype=np.complex64).copy()
    if spectrum.size != slice_len:
        raise ValueError(f"slice sample count mismatch: {spectrum.size} vs {slice_len}")

    meta = {
        "version": int(version),
        "frame_samples": int(frame_samples),
        "corr_len": int(corr_len),
        "slice_start": int(slice_start),
        "slice_len": int(slice_len),
        "peak_index": int(peak_index),
        "expected_zc_pos": int(expected_zc_pos),
        "offset_samples": int(offset_samples),
        "peak_power": float(peak_power),
        "avg_power": float(avg_power),
        "ul_tx_rx_shift": int(ul_tx_rx_shift),
        "timing_advance": int(timing_advance),
        "generation": int(generation),
    }
    return meta, spectrum


class SelfScanWindow(DebugPlotWindow):
    def __init__(self, host: str, port: int, interval_ms: int) -> None:
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
        self.overview.addItem(self.overview_expected_line)
        self.plot_layout.addWidget(self.overview)

        self.plot = pg.PlotWidget()
        configure_plot(
            self.plot,
            "Uplink Self-ZC Matched-Filter Spectrum (UE slice)",
            "Power (dB)",
            "ZC symbol start (samples into RX frame)",
        )
        self.curve = self.plot.plot([0], [0], pen=pg.mkPen("#2563eb", width=1.2))
        self.peak_line = pg.InfiniteLine(
            pos=0, angle=90, pen=pg.mkPen("#dc2626", width=1.2)
        )
        self.plot.addItem(self.peak_line)
        self.expected_line = pg.InfiniteLine(
            pos=0,
            angle=90,
            pen=pg.mkPen("#16a34a", width=1.2, style=QtCore.Qt.PenStyle.DashLine),
        )
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

    def handle_message(self, data: bytes) -> None:
        try:
            meta, spectrum = parse_self_scan_frame(data)
        except Exception as exc:
            self.status_label.setText(f"Ignored frame ({len(data)} B): {exc}")
            return

        power = power_db(spectrum)
        start = meta["slice_start"]
        n = power.size
        x = np.arange(start, start + n, dtype=np.float64)

        peak_index = meta["peak_index"]
        expected = meta["expected_zc_pos"]
        offset = meta["offset_samples"]
        # Prefer wire-side peak/avg (full-search stats) when valid.
        if meta["peak_power"] > 0.0 and meta["avg_power"] > 0.0:
            peak_value = 10.0 * np.log10(meta["peak_power"])
            avg_value = 10.0 * np.log10(meta["avg_power"])
        else:
            local_peak = int(np.argmax(power))
            peak_value = float(power[local_peak])
            avg_value = float(np.mean(power))
        peak_over_avg = peak_value - avg_value
        finite_power = power[np.isfinite(power)]
        if finite_power.size:
            local_floor = float(np.percentile(finite_power, 5.0))
        else:
            local_floor = avg_value

        frame_len = max(meta["frame_samples"], meta["corr_len"], start + n, 1)
        slice_end = start + n

        self.curve.setData(x, power)
        self.peak_line.setPos(peak_index)
        self.expected_line.setPos(expected)
        self.overview_curve.setData([0, frame_len], [0.5, 0.5])
        self.slice_region.setRegion((start, slice_end))
        self.overview_peak_line.setPos(peak_index)
        self.overview_expected_line.setPos(expected)
        self.overview.setXRange(0, frame_len, padding=0)

        label = (
            f"Peak {peak_value:.1f} dB @ {peak_index}\n"
            f"peak-avg {peak_over_avg:.1f} dB\n"
            f"expected {expected}\n"
            f"offset {offset:+d} samples\n"
            f"shift/TADV {meta['ul_tx_rx_shift']}/{meta['timing_advance']}"
        )
        self.peak_label.setText(label)
        self.peak_label.setPos(peak_index, peak_value)
        self.plot.setXRange(start, max(start + 1, slice_end), padding=0)
        y_min = min(local_floor - 5.0, avg_value - 20.0)
        y_max = max(peak_value + 15.0, y_min + 20.0)
        self.plot.setYRange(y_min, y_max, padding=0)
        self.status_label.setText(
            f"slice {start}:{slice_end} of corr {meta['corr_len']} "
            f"(frame {meta['frame_samples']}) | peak {peak_value:.1f} dB @ {peak_index} | "
            f"floor {local_floor:.1f} dB | offset {offset:+d} | gen {meta['generation']}"
        )


def _parse_args() -> ViewerArgs:
    parser = argparse.ArgumentParser(
        description="Plot the UE-published uplink self-ZC scan slice (pure display)."
    )
    parser.add_argument("--host", default=None, help="Backend host or tcp:// endpoint")
    parser.add_argument("--port", type=int, default=None, help="Backend ZMQ port")
    parser.add_argument("--interval-ms", type=int, default=50, help="Plot refresh interval")
    args = parser.parse_args()
    saved_host, saved_port = load_endpoint(default_settings_key(), "127.0.0.1", DEFAULT_PORT)
    host = args.host if args.host is not None else saved_host
    port = int(args.port if args.port is not None else saved_port)
    if args.host is not None:
        _, parsed_host, parsed_port = normalize_endpoint(args.host, port)
        host = parsed_host
        if args.port is None:
            port = parsed_port
    return ViewerArgs(
        host,
        port,
        max(1, args.interval_ms),
    )


def main() -> None:
    args = _parse_args()
    app = QtWidgets.QApplication([])
    window = SelfScanWindow(args.host, args.port, args.interval_ms)
    window.resize(1160, 720)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
