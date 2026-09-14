from __future__ import annotations

import argparse
import csv
import logging
import re
import struct
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import zmq
from PyQt6 import QtCore, QtWidgets

try:
    from qt_debug_plot_common import DebugPlotWindow, configure_plot
    from viewer_endpoint_store import default_settings_key, load_endpoint, load_settings, save_settings
except ImportError:
    from scripts.qt_debug_plot_common import DebugPlotWindow, configure_plot
    from scripts.viewer_endpoint_store import default_settings_key, load_endpoint, load_settings, save_settings


DEFAULT_PORT = 12347
MAGIC = b"OCXODBG3"
VERSION = 3
LEGACY_MAGIC = b"OCXODBG2"
LEGACY_VERSION = 2
CHANNEL_COUNT = 16
HEADER_STRUCT = struct.Struct("<8sIIQQ")
PACKET_V2_STRUCT = struct.Struct("<8sIIQQ16f")
PACKET_STRUCT = struct.Struct("<8sIIQQ16fddqqqdd")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChannelSpec:
    index: int
    name: str
    unit: str
    group: str
    color: str
    default_visible: bool = False


CHANNELS = (
    ChannelSpec(0, "cfo_mod_hz", "Hz", "CFO", "#2563eb"),
    ChannelSpec(1, "cfo_dealias_hz", "Hz", "CFO", "#dc2626"),
    ChannelSpec(2, "cfo_raw_ppm", "ppm", "Frequency / command", "#ea580c", True),
    ChannelSpec(3, "cfo_control_ppm", "ppm", "Frequency / command", "#16a34a", True),
    ChannelSpec(4, "ofdm_sfo_ppm", "ppm", "Frequency / command", "#7c3aed"),
    ChannelSpec(5, "to_raw", "samples", "Timing", "#db2777"),
    ChannelSpec(6, "to_clock", "samples", "Timing", "#0891b2", True),
    ChannelSpec(7, "phase_error", "samples", "Timing", "#65a30d", True),
    ChannelSpec(8, "clock_sfo_to_slope_ppm", "ppm", "Frequency / command", "#ca8a04", True),
    ChannelSpec(9, "alignment_samples", "samples", "Timing", "#9333ea"),
    ChannelSpec(10, "alignment_ff", "samples", "Timing", "#0d9488"),
    ChannelSpec(11, "desired_delta_ppm", "ppm", "Frequency / command", "#e11d48", True),
    ChannelSpec(12, "applied_delta_ppm", "ppm", "Frequency / command", "#0284c7", True),
    ChannelSpec(13, "command_ppm", "ppm", "Frequency / command", "#4f46e5", True),
    ChannelSpec(14, "coarse_outstanding_ppm", "ppm", "Frequency / command", "#b45309", True),
    ChannelSpec(15, "stage", "state", "Stage", "#f8fafc", True),
)

GROUPS = (
    ("CFO", "Hz"),
    ("Frequency / command", "ppm"),
    ("Timing", "samples"),
    ("Stage", "state"),
)

PPS_DIAGNOSTIC_NAMES = (
    "r_bs_samples",
    "r_ue_samples",
    "bs_last_pps_ticks",
    "bs_t0_to_pps_ticks",
    "ue_last_pps_ns",
    "bs_tick_rate",
    "sample_rate",
)


@dataclass(frozen=True)
class OcxoDebugFrame:
    sequence: int
    timestamp_ns: int
    channels: np.ndarray
    r_bs_samples: float
    r_ue_samples: float
    bs_last_pps_ticks: int
    bs_t0_to_pps_ticks: int
    ue_last_pps_ns: int
    bs_tick_rate: float
    sample_rate: float


@dataclass(frozen=True)
class ViewerArgs:
    host: str
    port: int
    interval_ms: int
    history_seconds: float
    channels: tuple[int, ...] | None
    record_path: Path | None


def parse_packet(data: bytes) -> OcxoDebugFrame:
    if len(data) < HEADER_STRUCT.size:
        raise ValueError(
            f"bad OCXO debug packet size: {len(data)} < {HEADER_STRUCT.size}"
        )
    magic, version, channel_count, sequence, timestamp_ns = HEADER_STRUCT.unpack_from(data)
    if magic not in (MAGIC, LEGACY_MAGIC):
        raise ValueError(f"bad OCXO debug magic: {magic!r}")
    if (magic, version) not in (
        (MAGIC, VERSION),
        (LEGACY_MAGIC, LEGACY_VERSION),
    ):
        raise ValueError(f"unsupported OCXO debug version: {version}")
    if channel_count != CHANNEL_COUNT:
        raise ValueError(f"bad OCXO debug channel count: {channel_count}")
    packet_struct = PACKET_STRUCT if version == VERSION else PACKET_V2_STRUCT
    if len(data) != packet_struct.size:
        raise ValueError(
            f"bad OCXO debug packet size: {len(data)} != {packet_struct.size}"
        )
    unpacked = packet_struct.unpack(data)
    values = np.asarray(unpacked[5:5 + CHANNEL_COUNT], dtype=np.float32)
    if version == VERSION:
        diagnostics = unpacked[5 + CHANNEL_COUNT:]
    else:
        diagnostics = (float("nan"), float("nan"), -1, -1, -1,
                       float("nan"), float("nan"))
    return OcxoDebugFrame(
        sequence=sequence,
        timestamp_ns=timestamp_ns,
        channels=values,
        r_bs_samples=float(diagnostics[0]),
        r_ue_samples=float(diagnostics[1]),
        bs_last_pps_ticks=int(diagnostics[2]),
        bs_t0_to_pps_ticks=int(diagnostics[3]),
        ue_last_pps_ns=int(diagnostics[4]),
        bs_tick_rate=float(diagnostics[5]),
        sample_rate=float(diagnostics[6]),
    )


def _parse_channel_selection(text: str | None) -> tuple[int, ...] | None:
    if text is None:
        return None
    by_name = {spec.name: spec.index for spec in CHANNELS}
    selected: list[int] = []
    for token in text.split(","):
        value = token.strip()
        if not value:
            continue
        try:
            index = int(value)
        except ValueError:
            if value not in by_name:
                raise ValueError(f"unknown channel: {value}") from None
            index = by_name[value]
        if not 0 <= index < CHANNEL_COUNT:
            raise ValueError(f"channel index out of range: {index}")
        if index not in selected:
            selected.append(index)
    return tuple(selected)


def parse_args() -> ViewerArgs:
    parser = argparse.ArgumentParser(description="Plot and record the OCXO ZeroMQ debug stream.")
    parser.add_argument("--host", default=None, help="Backend host")
    parser.add_argument("--port", type=int, default=None, help="Backend ZMQ port")
    parser.add_argument("--interval-ms", type=int, default=50, help="Plot refresh interval")
    parser.add_argument("--history-seconds", type=float, default=None, help="Rolling history duration")
    parser.add_argument("--channels", default=None, help="Comma-separated channel names or indices")
    parser.add_argument("--record", type=Path, default=None, help="Start CSV recording at launch")
    parsed = parser.parse_args()

    settings_key = default_settings_key()
    saved_host, saved_port = load_endpoint(settings_key, "127.0.0.1", DEFAULT_PORT)
    saved = load_settings(settings_key)
    host = str(parsed.host if parsed.host is not None else saved_host)
    port = int(parsed.port if parsed.port is not None else saved_port)
    if not 1 <= port <= 65535:
        parser.error("port must be in 1..65535")
    history = float(
        parsed.history_seconds
        if parsed.history_seconds is not None
        else saved.get("history_seconds", 120.0)
    )
    try:
        selected = _parse_channel_selection(parsed.channels)
    except ValueError as exc:
        parser.error(str(exc))
    return ViewerArgs(
        host=host,
        port=port,
        interval_ms=max(10, int(parsed.interval_ms)),
        history_seconds=min(max(history, 10.0), 3600.0),
        channels=selected,
        record_path=parsed.record,
    )


def _utc_text(timestamp_ns: int) -> str:
    seconds, nanos = divmod(int(timestamp_ns), 1_000_000_000)
    stamp = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return f"{stamp:%Y-%m-%dT%H:%M:%S}.{nanos // 1_000_000:03d}Z"


def _unique_path(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.exists():
        return candidate
    daily_match = re.fullmatch(r"(.+_)(\d{2,})", candidate.stem)
    if daily_match:
        prefix, existing_index = daily_match.groups()
        start_index = int(existing_index) + 1
        width = len(existing_index)
    else:
        prefix = f"{candidate.stem}_"
        start_index = 1
        width = 3
    for index in range(start_index, 10_000):
        numbered = candidate.with_name(
            f"{prefix}{index:0{width}d}{candidate.suffix}"
        )
        if not numbered.exists():
            return numbered
    raise OSError(f"could not allocate a unique recording path near {candidate}")


def _daily_record_path(directory: Path, now: datetime | None = None) -> Path:
    stamp = now or datetime.now()
    return _unique_path(directory / f"ocxo_debug_{stamp:%Y%m%d}_01.csv")


def _csv_header() -> list[str]:
    return (
        ["source_time_ns", "source_time_utc", "receive_time_ns", "sequence"]
        + [spec.name for spec in CHANNELS]
        + list(PPS_DIAGNOSTIC_NAMES)
    )


def _csv_row(frame: OcxoDebugFrame, receive_time_ns: int) -> list[str | int]:
    return (
        [frame.timestamp_ns, _utc_text(frame.timestamp_ns), receive_time_ns, frame.sequence]
        + [f"{float(value):.9g}" for value in frame.channels]
        + [
            f"{frame.r_bs_samples:.17g}",
            f"{frame.r_ue_samples:.17g}",
            str(frame.bs_last_pps_ticks),
            str(frame.bs_t0_to_pps_ticks),
            str(frame.ue_last_pps_ns),
            f"{frame.bs_tick_rate:.17g}",
            f"{frame.sample_rate:.17g}",
        ]
    )


class OcxoDebugWindow(DebugPlotWindow):
    def __init__(self, args: ViewerArgs) -> None:
        self.args = args
        self.samples: deque[tuple[float, int, int, np.ndarray]] = deque()
        self.received_frames = 0
        self.invalid_frames = 0
        self.sequence_gaps = 0
        self.last_sequence: int | None = None
        self.csv_file = None
        self.csv_writer: csv.writer | None = None
        self.last_csv_flush = 0.0
        self.channel_checks: dict[int, QtWidgets.QCheckBox] = {}
        self.plots: dict[str, pg.PlotWidget] = {}
        self.legends: dict[str, pg.LegendItem] = {}
        self.curves: dict[int, pg.PlotDataItem] = {}

        super().__init__(
            title="OCXO Debug",
            default_port=args.port,
            host=args.host,
            interval_ms=args.interval_ms,
            conflate=False,
        )
        self._build_controls()
        self._build_plots()
        self._load_channel_selection(args.channels)
        self.resize(1280, 920)
        if args.record_path is not None:
            self.record_path_edit.setText(str(args.record_path))
            self.start_recording()

    def _build_controls(self) -> None:
        settings = load_settings(self.settings_key)
        panel = QtWidgets.QWidget()
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)

        history_row = QtWidgets.QHBoxLayout()
        history_row.addWidget(QtWidgets.QLabel("History"))
        self.history_spin = QtWidgets.QDoubleSpinBox()
        self.history_spin.setRange(10.0, 3600.0)
        self.history_spin.setDecimals(0)
        self.history_spin.setSuffix(" s")
        self.history_spin.setValue(self.args.history_seconds)
        self.history_spin.valueChanged.connect(self._history_changed)
        history_row.addWidget(self.history_spin)

        self.pause_button = QtWidgets.QPushButton("Pause plot")
        self.pause_button.setCheckable(True)
        self.pause_button.toggled.connect(self._pause_changed)
        history_row.addWidget(self.pause_button)
        clear_button = QtWidgets.QPushButton("Clear")
        clear_button.clicked.connect(self.clear_history)
        history_row.addWidget(clear_button)
        history_row.addStretch(1)
        self.stream_label = QtWidgets.QLabel("Waiting for OCXO debug frames")
        self.stream_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        history_row.addWidget(self.stream_label, 2)
        panel_layout.addLayout(history_row)

        channel_box = QtWidgets.QGroupBox("Channels")
        channel_layout = QtWidgets.QGridLayout(channel_box)
        for offset, spec in enumerate(CHANNELS):
            check = QtWidgets.QCheckBox(f"{spec.index}  {spec.name}")
            check_color = "#475569" if spec.index == 15 else spec.color
            check.setStyleSheet(f"QCheckBox {{ color: {check_color}; }}")
            check.toggled.connect(self._channel_visibility_changed)
            self.channel_checks[spec.index] = check
            channel_layout.addWidget(check, offset // 4, offset % 4)
        all_button = QtWidgets.QPushButton("All")
        all_button.clicked.connect(lambda: self._set_all_channels(True))
        none_button = QtWidgets.QPushButton("None")
        none_button.clicked.connect(lambda: self._set_all_channels(False))
        channel_layout.addWidget(all_button, 4, 2)
        channel_layout.addWidget(none_button, 4, 3)
        panel_layout.addWidget(channel_box)

        record_row = QtWidgets.QHBoxLayout()
        record_row.addWidget(QtWidgets.QLabel("CSV"))
        saved_record_path = settings.get("record_path")
        record_dir = (
            Path(str(saved_record_path)).expanduser().parent
            if saved_record_path
            else Path.cwd()
        )
        self.record_path_edit = QtWidgets.QLineEdit(
            str(_daily_record_path(record_dir))
        )
        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.clicked.connect(self.choose_record_path)
        self.record_button = QtWidgets.QPushButton("Start recording")
        self.record_button.clicked.connect(self.toggle_recording)
        record_row.addWidget(self.record_path_edit, 1)
        record_row.addWidget(browse_button)
        record_row.addWidget(self.record_button)
        panel_layout.addLayout(record_row)
        self.root_layout.insertWidget(1, panel)

    def _build_plots(self) -> None:
        first_plot: pg.PlotWidget | None = None
        for group, unit in GROUPS:
            plot = pg.PlotWidget()
            configure_plot(plot, group, unit, "Time from newest sample (s)")
            self.legends[group] = plot.addLegend(offset=(8, 8), colCount=2)
            if first_plot is None:
                first_plot = plot
            else:
                plot.setXLink(first_plot)
            self.plots[group] = plot
            # Equal stretch keeps a group from reopening at its minimum size
            # after Qt has redistributed the hidden group's former height.
            self.plot_layout.addWidget(plot, 1)

        stage_axis = self.plots["Stage"].getPlotItem().getAxis("left")
        stage_axis.setTicks([[(0, "Coarse"), (1, "Fast"), (2, "Slow"), (3, "Phase"), (4, "Freeze")]])
        self.plots["Stage"].setYRange(-0.25, 4.25, padding=0)

        for spec in CHANNELS:
            curve = self.plots[spec.group].plot(
                [], [], pen=pg.mkPen(spec.color, width=1.5)
            )
            self.curves[spec.index] = curve

    def _load_channel_selection(self, override: tuple[int, ...] | None) -> None:
        if override is not None:
            selected = set(override)
        else:
            saved = load_settings(self.settings_key).get("channels")
            if isinstance(saved, list):
                selected = {int(value) for value in saved if isinstance(value, int)}
            else:
                selected = {spec.index for spec in CHANNELS if spec.default_visible}
        for index, check in self.channel_checks.items():
            check.setChecked(index in selected)
        self._channel_visibility_changed()

    def _selected_channels(self) -> list[int]:
        return [index for index, check in self.channel_checks.items() if check.isChecked()]

    def _set_all_channels(self, checked: bool) -> None:
        for check in self.channel_checks.values():
            check.blockSignals(True)
            check.setChecked(checked)
            check.blockSignals(False)
        self._channel_visibility_changed()

    def _channel_visibility_changed(self) -> None:
        selected = set(self._selected_channels())
        for spec in CHANNELS:
            self.curves[spec.index].setVisible(spec.index in selected)
        for group, _unit in GROUPS:
            legend = self.legends[group]
            legend.clear()
            for spec in CHANNELS:
                if spec.group == group and spec.index in selected:
                    legend.addItem(self.curves[spec.index], f"{spec.index} {spec.name}")
            self.plots[group].setVisible(
                any(spec.group == group and spec.index in selected for spec in CHANNELS)
            )
        save_settings(self.settings_key, {"channels": sorted(selected)})
        if not self.pause_button.isChecked():
            self.update_curves()

    def _history_changed(self, seconds: float) -> None:
        save_settings(self.settings_key, {"history_seconds": float(seconds)})
        self._prune_history()
        if not self.pause_button.isChecked():
            self.update_curves()

    def _pause_changed(self, paused: bool) -> None:
        self.pause_button.setText("Resume plot" if paused else "Pause plot")
        if not paused:
            self.update_curves()

    def clear_history(self) -> None:
        self.samples.clear()
        for curve in self.curves.values():
            curve.setData([], [])

    def choose_record_path(self) -> None:
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "OCXO CSV recording",
            self.record_path_edit.text(),
            "CSV files (*.csv);;All files (*)",
        )
        if path:
            self.record_path_edit.setText(path)

    def toggle_recording(self) -> None:
        if self.csv_file is None:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self) -> None:
        if self.csv_file is not None:
            return
        try:
            raw_path = self.record_path_edit.text().strip()
            if not raw_path:
                raise OSError("recording path is empty")
            path = _unique_path(Path(raw_path))
            path.parent.mkdir(parents=True, exist_ok=True)
            self.csv_file = path.open("w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(_csv_header())
            self.csv_file.flush()
            self.last_csv_flush = time.monotonic()
            self.record_path_edit.setText(str(path))
            self.record_path_edit.setEnabled(False)
            self.record_button.setText("Stop recording")
            save_settings(self.settings_key, {"record_path": str(path)})
        except OSError as exc:
            self.csv_file = None
            self.csv_writer = None
            self.stream_label.setText(f"Record error: {exc}")

    def stop_recording(self) -> None:
        if self.csv_file is not None:
            self.csv_file.flush()
            self.csv_file.close()
        self.csv_file = None
        self.csv_writer = None
        self.record_path_edit.setEnabled(True)
        self.record_button.setText("Start recording")

    def _record_frame(self, frame: OcxoDebugFrame, receive_time_ns: int) -> None:
        if self.csv_writer is None or self.csv_file is None:
            return
        self.csv_writer.writerow(_csv_row(frame, receive_time_ns))
        now = time.monotonic()
        if now - self.last_csv_flush >= 1.0:
            self.csv_file.flush()
            self.last_csv_flush = now

    def _ingest_packet(self, data: bytes) -> None:
        frame = parse_packet(data)
        receive_time_ns = time.time_ns()
        if self.last_sequence is not None and frame.sequence > self.last_sequence + 1:
            self.sequence_gaps += frame.sequence - self.last_sequence - 1
        self.last_sequence = frame.sequence
        self.received_frames += 1
        self.samples.append((time.monotonic(), frame.sequence, frame.timestamp_ns, frame.channels))
        self._record_frame(frame, receive_time_ns)

    def _prune_history(self) -> None:
        cutoff = time.monotonic() - float(self.history_spin.value())
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def poll_socket(self) -> None:
        if self.sock is None:
            return
        received = 0
        while received < 4096:
            try:
                parts = self.sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception as exc:
                LOGGER.warning("OCXO debug receive failed: %s", exc)
                self.status_label.setText(f"Receive error: {exc}")
                return
            received += 1
            if len(parts) != 1:
                self.invalid_frames += 1
                continue
            try:
                self._ingest_packet(parts[0])
            except Exception as exc:
                self.invalid_frames += 1
                LOGGER.warning("Dropped invalid OCXO debug packet: %s", exc)

        if received == 0:
            return
        self._prune_history()
        if not self.pause_button.isChecked():
            self.update_curves()
        recording = " | REC" if self.csv_file is not None else ""
        self.stream_label.setText(
            f"frames={self.received_frames} | gaps={self.sequence_gaps} | "
            f"invalid={self.invalid_frames} | buffered={len(self.samples)}{recording}"
        )

    def update_curves(self) -> None:
        if not self.samples:
            return
        newest = self.samples[-1][0]
        x = np.fromiter((sample[0] - newest for sample in self.samples), dtype=np.float64)
        values = np.stack([sample[3] for sample in self.samples])
        for index in self._selected_channels():
            self.curves[index].setData(x, values[:, index])
        history = float(self.history_spin.value())
        for plot in self.plots.values():
            plot.setXRange(-history, 0.0, padding=0)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.stop_recording()
        super().closeEvent(event)


def main() -> None:
    args = parse_args()
    app = QtWidgets.QApplication([])
    window = OcxoDebugWindow(args)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
