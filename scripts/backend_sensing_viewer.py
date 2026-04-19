from __future__ import annotations

import argparse
import datetime
import os
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import pyqtgraph as pg
import scipy.io as sio
import yaml
from PyQt6 import QtCore, QtGui, QtWidgets

from sensing_detection import build_detection_views
from sensing_runtime_protocol import (
    AGGREGATE_MAGIC_VERSION,
    CTRL_HEADER,
    FLAG_ENABLE_MTI,
    PARAMS_COMMAND,
    READY_COMMAND,
    ViewerRuntimeParams,
    build_params_request,
    decode_aggregate_sensing_metadata_payload,
    decode_aggregate_sensing_payload,
    decode_sensing_metadata_payload,
    decode_sensing_payload,
    parse_params_packet,
)


HEADER_SIZE = 12
MAX_CHUNK_SIZE = 60000
SOCKET_BUFFER_SIZE = 8 * 1024 * 1024
DEFAULT_CONTROL_PORT = 9999
DEFAULT_PORT_MONO = 8888
DEFAULT_PORT_BI = 8889
DEFAULT_DISPLAY_RANGE_BINS = 256
DEFAULT_DISPLAY_DOPPLER_BINS = 128
PENDING_BUNDLE_LIMIT = 8
C_LIGHT_MPS = 299792458.0
ANTENNA_SPACING_M = 42.83e-3
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


LEGACY_VIEWER_PARAMS = ViewerRuntimeParams(
    version=0,
    flags=0,
    frame_format=0,
    wire_rows=100,
    wire_cols=1024,
    active_rows=100,
    active_cols=1024,
    frame_symbol_period=100,
    range_fft_size=1024,
    doppler_fft_size=100,
    compact_mask_hash=0,
)


@dataclass
class ViewerLaunchConfig:
    mode: str
    port: int
    control_port: int
    channels: int
    title: str
    expect_backend_processing: bool
    display_range_bins: int
    display_doppler_bins: int
    downsample: int


@dataclass
class ChannelState:
    ch_id: int
    last_frame_id: int = -1
    latest_frame: DisplayFrame | None = None


@dataclass
class PendingBundle:
    raw_frames: dict[int, np.ndarray] | None = None
    metadata_frames: dict[int, object] | None = None
    created_at: float = field(default_factory=time.time)


@dataclass
class DisplayFrame:
    ch_id: int
    frame_id: int
    rd: np.ndarray
    rd_complex: np.ndarray | None
    cfar_points: np.ndarray
    cfar_hits: int
    cfar_shown_hits: int
    cfar_stats: dict | None
    target_clusters: list[dict]
    md: np.ndarray | None
    md_extent: list[float] | None
    viewer_params: ViewerRuntimeParams


class FrameBuffer:
    def __init__(self) -> None:
        self.frame_id = -1
        self.total_chunks = 0
        self.buffer: list[bytes | None] = []
        self.received_chunks = 0

    def init(self, frame_id: int, total_chunks: int) -> None:
        self.frame_id = int(frame_id)
        self.total_chunks = int(total_chunks)
        self.buffer = [None] * self.total_chunks
        self.received_chunks = 0

    def add_chunk(self, chunk_id: int, data: bytes) -> bool:
        if chunk_id < 0 or chunk_id >= self.total_chunks:
            return False
        if self.buffer[chunk_id] is not None:
            return False
        self.buffer[chunk_id] = data
        self.received_chunks += 1
        return self.received_chunks == self.total_chunks

    def assemble_payload(self) -> tuple[int, bytes]:
        if self.received_chunks != self.total_chunks or self.total_chunks <= 0:
            raise ValueError("Frame buffer is incomplete")
        return self.frame_id, b"".join(self.buffer)  # type: ignore[arg-type]


def load_launch_defaults(
    mode_hint: str | None,
    default_mode: str | None,
    default_title: str | None,
) -> ViewerLaunchConfig:
    inferred_mode = mode_hint or default_mode
    if inferred_mode in {None, "auto"}:
        inferred_mode = "mono"

    if inferred_mode == "bi":
        port = DEFAULT_PORT_BI
        control_port = DEFAULT_CONTROL_PORT
        channel_count = 1
        title = default_title or "OpenISAC Backend Bi-Sensing Viewer"
    else:
        port = DEFAULT_PORT_MONO
        control_port = DEFAULT_CONTROL_PORT
        channel_count = 2
        title = default_title or "OpenISAC Backend Sensing Viewer"

    return ViewerLaunchConfig(
        mode=inferred_mode,
        port=port,
        control_port=control_port,
        channels=max(1, channel_count),
        title=title,
        expect_backend_processing=True,
        display_range_bins=DEFAULT_DISPLAY_RANGE_BINS,
        display_doppler_bins=DEFAULT_DISPLAY_DOPPLER_BINS,
        downsample=1,
    )


class BackendViewerRuntime:
    def __init__(self, launch_cfg: ViewerLaunchConfig) -> None:
        self.launch_cfg = launch_cfg
        self._running = True
        self._viewer_params = LEGACY_VIEWER_PARAMS
        self._params_lock = threading.Lock()
        self._channel_lock = threading.Lock()
        self._buffer_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._display_queue: Queue[DisplayFrame] = Queue(maxsize=32)
        self._raw_buffer = FrameBuffer()
        self._metadata_buffer = FrameBuffer()
        self._pending_bundles: dict[int, PendingBundle] = {}
        self._sender_ip: str | None = None
        self._control_port = int(launch_cfg.control_port)
        self._last_error: str | None = None
        self._warned_non_backend = False
        self._channels: list[ChannelState] = []
        self._phase_bundle_lock = threading.Lock()
        self._latest_phase_bundle_frame_id: int | None = None
        self._latest_phase_bundle_rd_complex: list[np.ndarray | None] | None = None
        self.ensure_channel_count(launch_cfg.channels)
        self._receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._receiver_thread.start()

    def stop(self) -> None:
        self._running = False

    def ensure_channel_count(self, count: int, reason: str = "auto-detect") -> bool:
        count = max(1, int(count))
        with self._channel_lock:
            current = len(self._channels)
            if current == count:
                return False

            new_channels: list[ChannelState] = []
            for ch_id in range(count):
                if ch_id < current:
                    ch = self._channels[ch_id]
                else:
                    ch = ChannelState(ch_id=ch_id)
                ch.ch_id = ch_id
                ch.last_frame_id = -1
                ch.latest_frame = None
                new_channels.append(ch)
            self._channels = new_channels
        with self._phase_bundle_lock:
            self._latest_phase_bundle_frame_id = None
            self._latest_phase_bundle_rd_complex = None

        print(f"Auto-adjusted logical channel count: {current} -> {count} ({reason})")
        return True

    def channel_count(self) -> int:
        with self._channel_lock:
            return len(self._channels)

    def channels(self) -> list[ChannelState]:
        with self._channel_lock:
            return list(self._channels)

    def get_channel(self, ch_id: int) -> ChannelState | None:
        with self._channel_lock:
            if 0 <= ch_id < len(self._channels):
                return self._channels[ch_id]
        return None

    def get_viewer_params(self) -> ViewerRuntimeParams:
        with self._params_lock:
            return self._viewer_params

    def last_error(self) -> str | None:
        return self._last_error

    def sender_endpoint(self) -> tuple[str | None, int]:
        return self._sender_ip, self._control_port

    def latest_phase_bundle(self) -> tuple[int | None, list[np.ndarray | None] | None]:
        with self._phase_bundle_lock:
            return self._latest_phase_bundle_frame_id, self._latest_phase_bundle_rd_complex

    def pending_bundle_count(self) -> int:
        with self._pending_lock:
            return len(self._pending_bundles)

    def display_queue_size(self) -> int:
        return self._display_queue.qsize()

    def request_viewer_params(self) -> None:
        if self._sender_ip is None:
            return
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(build_params_request(0), (self._sender_ip, self._control_port))
            sock.close()
        except Exception as exc:
            self._last_error = str(exc)
            print(f"Failed to request viewer params: {exc}")

    def send_control_command(self, cmd_id: bytes, value: int) -> bool:
        if self._sender_ip is None:
            print("Control sender not detected yet.")
            return False
        try:
            packet = struct.pack("!4s4si", b"CMD ", cmd_id, int(value))
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(packet, (self._sender_ip, self._control_port))
            sock.close()
            return True
        except Exception as exc:
            self._last_error = str(exc)
            print(f"Failed to send {cmd_id!r}: {exc}")
            return False

    def drain_display_queue(self) -> list[DisplayFrame]:
        items: list[DisplayFrame] = []
        while True:
            try:
                items.append(self._display_queue.get_nowait())
            except Empty:
                break
        return items

    def _set_viewer_params(self, params: ViewerRuntimeParams) -> None:
        with self._params_lock:
            self._viewer_params = params
        if params.aggregated_stream():
            self.ensure_channel_count(params.stream_channel_count, reason="viewer params")
        print(f"Viewer params: {params.describe()}")

    def _aggregate_logical_channel_count(self, payload: bytes) -> int | None:
        if len(payload) < 24:
            return None
        try:
            magic_version, channel_count, _, channel_mask, _ = struct.unpack_from("!IIIIQ", payload)
        except struct.error:
            return None
        if magic_version != AGGREGATE_MAGIC_VERSION:
            return None
        highest_channel = max(
            [bit for bit in range(32) if channel_mask & (1 << bit)],
            default=(int(channel_count) - 1),
        )
        return max(int(channel_count), highest_channel + 1)

    def _update_sender(self, addr: tuple[str, int], update_control_port: bool = False) -> None:
        sender_ip, sender_port = addr
        if self._sender_ip is None:
            self._sender_ip = sender_ip
            print(f"Detected sender IP: {sender_ip}")
        if update_control_port and self._control_port != sender_port:
            self._control_port = sender_port
            print(f"Control port updated to {sender_port}")

    def _prune_pending_bundles(self) -> None:
        with self._pending_lock:
            while len(self._pending_bundles) > PENDING_BUNDLE_LIMIT:
                oldest_frame_id = min(self._pending_bundles.keys())
                self._pending_bundles.pop(oldest_frame_id, None)

    def _store_raw_frames(self, frame_id: int, frames: dict[int, np.ndarray]) -> None:
        with self._pending_lock:
            pending = self._pending_bundles.setdefault(int(frame_id), PendingBundle())
            pending.raw_frames = frames
        self._prune_pending_bundles()
        self._try_dispatch_bundle(frame_id)

    def _store_metadata_frames(self, frame_id: int, frames: dict[int, object]) -> None:
        with self._pending_lock:
            pending = self._pending_bundles.setdefault(int(frame_id), PendingBundle())
            pending.metadata_frames = frames
        self._prune_pending_bundles()
        self._try_dispatch_bundle(frame_id)

    def _metadata_required(self) -> bool:
        params = self.get_viewer_params()
        return params.backend_processing() and params.metadata_sidecar()

    def _try_dispatch_bundle(self, frame_id: int) -> None:
        params = self.get_viewer_params()
        with self._pending_lock:
            pending = self._pending_bundles.get(int(frame_id))
            if pending is None or pending.raw_frames is None:
                return
            if self._metadata_required() and pending.metadata_frames is None:
                return
            raw_frames = pending.raw_frames
            metadata_frames = pending.metadata_frames or {}
            self._pending_bundles.pop(int(frame_id), None)

        bundle_rd_complex: list[np.ndarray | None] = [None] * self.channel_count()
        for ch_id, raw_frame in raw_frames.items():
            try:
                display_frame = self._build_display_frame(
                    ch_id=ch_id,
                    frame_id=int(frame_id),
                    raw_frame=raw_frame,
                    metadata=metadata_frames.get(ch_id),
                    viewer_params=params,
                )
            except Exception as exc:
                self._last_error = str(exc)
                print(f"Failed to process CH{ch_id + 1} frame {frame_id}: {exc}")
                continue

            channel = self.get_channel(ch_id)
            if channel is not None:
                channel.last_frame_id = int(frame_id)
                channel.latest_frame = display_frame
            if ch_id < len(bundle_rd_complex):
                bundle_rd_complex[ch_id] = display_frame.rd_complex

            if self._display_queue.full():
                try:
                    self._display_queue.get_nowait()
                except Exception:
                    pass
            self._display_queue.put(display_frame)

        with self._phase_bundle_lock:
            self._latest_phase_bundle_frame_id = int(frame_id)
            self._latest_phase_bundle_rd_complex = bundle_rd_complex

    def _map_cfar_points(
        self,
        metadata_points: np.ndarray | None,
        rows: int,
        cols: int,
        display_rows: int,
        display_cols: int,
        downsample: int,
    ) -> np.ndarray:
        if metadata_points is None or metadata_points.size == 0:
            return np.empty((0, 2), dtype=np.int32)

        ds = max(1, int(downsample))
        display_col_stop = min(max(1, int(display_cols)), cols)
        center_idx = rows // 2
        display_row_start = max(0, center_idx - max(1, int(display_rows)) // 2)
        display_row_stop = min(rows, display_row_start + max(1, int(display_rows)))
        display_row_indices = np.arange(display_row_start, display_row_stop, ds, dtype=np.int32)
        display_col_indices = np.arange(0, display_col_stop, ds, dtype=np.int32)
        row_lookup = {int(row): idx for idx, row in enumerate(display_row_indices.tolist())}
        col_lookup = {int(col): idx for idx, col in enumerate(display_col_indices.tolist())}

        mapped_points: list[tuple[int, int]] = []
        for doppler_idx, range_idx in np.asarray(metadata_points, dtype=np.int32):
            row_idx = row_lookup.get(int(doppler_idx))
            col_idx = col_lookup.get(int(range_idx))
            if row_idx is None or col_idx is None:
                continue
            mapped_points.append((row_idx, col_idx))

        if not mapped_points:
            return np.empty((0, 2), dtype=np.int32)
        return np.asarray(mapped_points, dtype=np.int32)

    def _build_display_frame(
        self,
        *,
        ch_id: int,
        frame_id: int,
        raw_frame: np.ndarray,
        metadata: object | None,
        viewer_params: ViewerRuntimeParams,
    ) -> DisplayFrame:
        if not viewer_params.is_dense_range_doppler():
            if not self._warned_non_backend:
                self._warned_non_backend = True
                print(
                    "Backend viewer is waiting for dense range-doppler output. "
                    f"Current params: {viewer_params.describe()}"
                )
            raise ValueError("Current sensing stream is not dense range-doppler")

        rd_complex = np.asarray(
            raw_frame[:viewer_params.wire_rows, :viewer_params.wire_cols],
            dtype=np.complex64,
        )
        rd_shifted = np.fft.fftshift(rd_complex, axes=0)
        rd_spectrum = (20.0 * np.log10(np.abs(rd_shifted) + 1e-12)).astype(np.float32, copy=False)

        display_range_bins = min(self.launch_cfg.display_range_bins, rd_spectrum.shape[1])
        display_doppler_bins = min(self.launch_cfg.display_doppler_bins, rd_spectrum.shape[0])
        rd_display, _, _, _ = build_detection_views(
            rd_spectrum,
            display_range_bins,
            display_doppler_bins,
            self.launch_cfg.downsample,
        )
        rd_complex_display, _, _, _ = build_detection_views(
            rd_shifted,
            display_range_bins,
            display_doppler_bins,
            self.launch_cfg.downsample,
        )

        cfar_points = np.empty((0, 2), dtype=np.int32)
        cfar_hits = 0
        cfar_shown_hits = 0
        cfar_stats = None
        target_clusters: list[dict] = []
        md_data = None
        md_extent = None

        if metadata is not None:
            cfar_points = self._map_cfar_points(
                getattr(metadata, "cfar_points", None),
                rows=rd_spectrum.shape[0],
                cols=rd_spectrum.shape[1],
                display_rows=display_doppler_bins,
                display_cols=display_range_bins,
                downsample=self.launch_cfg.downsample,
            )
            cfar_hits = int(getattr(metadata, "cfar_hits", 0))
            cfar_shown_hits = int(getattr(metadata, "cfar_shown_hits", 0))
            cfar_stats = getattr(metadata, "cfar_stats", None)
            target_clusters = list(getattr(metadata, "target_clusters", []) or [])
            md_data = getattr(metadata, "md_spectrum", None)
            md_extent = getattr(metadata, "md_extent", None)

        return DisplayFrame(
            ch_id=ch_id,
            frame_id=frame_id,
            rd=rd_display,
            rd_complex=np.asarray(rd_complex_display, dtype=np.complex64),
            cfar_points=cfar_points,
            cfar_hits=cfar_hits,
            cfar_shown_hits=cfar_shown_hits,
            cfar_stats=cfar_stats,
            target_clusters=target_clusters,
            md=md_data,
            md_extent=md_extent,
            viewer_params=viewer_params,
        )

    def _handle_completed_payload(self, frame_id_hint: int, payload: bytes, is_metadata: bool) -> None:
        params = self.get_viewer_params()

        if is_metadata:
            if len(payload) >= 4 and payload[:4] == b"ASM1":
                frame_id, metadata_frames = decode_aggregate_sensing_metadata_payload(payload)
                self.ensure_channel_count(
                    max([ch_id for ch_id, _ in metadata_frames], default=-1) + 1,
                    reason="aggregate metadata",
                )
                self._store_metadata_frames(frame_id, {ch_id: meta for ch_id, meta in metadata_frames})
                return
            if len(payload) >= 4 and payload[:4] == b"SMD1":
                metadata = decode_sensing_metadata_payload(payload)
                self.ensure_channel_count(1, reason="single-channel metadata")
                self._store_metadata_frames(metadata.frame_id, {0: metadata})
                return
            print(f"Ignoring unknown metadata payload for frame {frame_id_hint}")
            return

        if len(payload) >= 4 and struct.unpack("!I", payload[:4])[0] == AGGREGATE_MAGIC_VERSION:
            if params.version == 0:
                self.request_viewer_params()
                raise ValueError("Aggregate payload arrived before viewer params were known")
            detected_count = self._aggregate_logical_channel_count(payload)
            if detected_count is not None:
                self.ensure_channel_count(detected_count, reason="aggregate frame")
            frame_id, decoded_frames = decode_aggregate_sensing_payload(frame_id_hint, payload, params)
            channel_frames = {int(ch_id): decoded.matrix for ch_id, decoded in decoded_frames}
            if channel_frames:
                self.ensure_channel_count(max(channel_frames.keys()) + 1, reason="decoded aggregate frame")
                self._store_raw_frames(frame_id, channel_frames)
            return

        decoded = decode_sensing_payload(frame_id_hint, payload, params)
        self.ensure_channel_count(1, reason="single-channel frame")
        self._store_raw_frames(decoded.frame_id, {0: decoded.matrix})

    def _receiver_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_BUFFER_SIZE)
        except Exception:
            pass

        try:
            sock.bind(("0.0.0.0", self.launch_cfg.port))
            sock.settimeout(0.5)
            print(f"Listening on UDP port {self.launch_cfg.port}")
        except Exception as exc:
            self._last_error = str(exc)
            print(f"Socket bind error: {exc}")
            return

        while self._running:
            try:
                data, addr = sock.recvfrom(MAX_CHUNK_SIZE + HEADER_SIZE)
            except socket.timeout:
                continue
            except Exception as exc:
                self._last_error = str(exc)
                print(f"Receiver socket error: {exc}")
                continue

            try:
                self._update_sender(addr)

                if len(data) >= 8 and data[:4] == CTRL_HEADER:
                    command = data[4:8]
                    if command == PARAMS_COMMAND:
                        params = parse_params_packet(data)
                        if params is not None:
                            self._update_sender(addr, update_control_port=True)
                            self._set_viewer_params(params)
                    elif command == READY_COMMAND:
                        self._update_sender(addr, update_control_port=True)
                        if self.get_viewer_params().version == 0 or self._last_error is not None:
                            self.request_viewer_params()
                    continue

                if len(data) < HEADER_SIZE:
                    continue

                frame_id, total_chunks, chunk_id = struct.unpack("!III", data[:HEADER_SIZE])
                is_metadata = bool(total_chunks & 0x80000000)
                total_chunks &= 0x7FFFFFFF
                chunk_data = data[HEADER_SIZE:]

                with self._buffer_lock:
                    buffer = self._metadata_buffer if is_metadata else self._raw_buffer
                    if buffer.frame_id != frame_id:
                        buffer.init(frame_id, total_chunks)
                    if not buffer.add_chunk(chunk_id, chunk_data):
                        continue
                    frame_id_done, payload = buffer.assemble_payload()

                self._handle_completed_payload(frame_id_done, payload, is_metadata)
            except Exception as exc:
                self._last_error = str(exc)
                print(f"Receiver error: {exc}")
                self.request_viewer_params()

        sock.close()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, runtime: BackendViewerRuntime) -> None:
        super().__init__()
        self.runtime = runtime
        self._selected_channel = 0
        self._last_render_key: tuple[int, int] | None = None
        self._last_params_text = ""
        self._rd_colormap = pg.colormap.get("turbo")
        self._md_colormap = pg.colormap.get("turbo")
        self._frame_counter = 0
        self._last_fps_ts = time.time()
        self._control_panel_width = 520
        self._status_label_text_width = self._control_panel_width - 24
        self.phase_ready = False
        self.synced_frame_id = None
        self.synced_rd_complex = None
        self.phase_curve_raw = None
        self.phase_curve_comp = None
        self._force_ui_refresh = False
        self._last_known_channel_count = self.runtime.channel_count()
        self.channel_bias_vector = np.ones(self.runtime.channel_count(), dtype=np.complex128)
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
        self.target_sector_plot.setLabel("left", "Forward Range (bin)")
        self.target_sector_plot.setLabel("bottom", "Cross-Range (bin)")
        self.target_sector_plot.showGrid(x=True, y=True, alpha=0.2)
        self.target_sector_plot.setAspectLocked(True)
        self.target_sector_plot.setMouseEnabled(x=False, y=False)
        self.target_sector_plot.setMenuEnabled(False)
        self.target_sector_plot.setMinimumHeight(360)
        self.target_sector_sensor_item = pg.ScatterPlotItem(
            [0.0], [0.0], symbol="t", size=14,
            pen=pg.mkPen("#ffffff", width=1.5),
            brush=pg.mkBrush(255, 255, 255, 220),
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
            pen=pg.mkPen("#0b132b", width=1.2),
            brush=pg.mkBrush(255, 145, 77, 210),
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

        self.setWindowTitle(runtime.launch_cfg.title)
        self.resize(1500, 920)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root_layout = QtWidgets.QHBoxLayout(central)

        plot_panel = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_panel)
        root_layout.addWidget(plot_panel, stretch=4)

        self.rd_plot = pg.PlotWidget(title="Range-Doppler Spectrum")
        self.rd_plot.setLabel("left", "Range Bin")
        self.rd_plot.setLabel("bottom", "Doppler Bin")
        self.rd_plot.showGrid(x=True, y=True, alpha=0.2)
        self.rd_img = pg.ImageItem(axisOrder="col-major")
        self.rd_img.setLookupTable(self._rd_colormap.getLookupTable())
        self.rd_plot.addItem(self.rd_img)
        self.rd_colorbar = pg.ColorBarItem(values=(0, 60), colorMap=self._rd_colormap, interactive=False)
        self.rd_colorbar.setImageItem(self.rd_img, insert_in=self.rd_plot.plotItem)
        self.rd_click_marker = pg.ScatterPlotItem([], [], symbol="x", size=12, pen=pg.mkPen("w", width=2))
        self.rd_cfar_marker = pg.ScatterPlotItem(
            [], [], symbol="o", size=7,
            pen=pg.mkPen("#00e5ff", width=1.5),
            brush=pg.mkBrush(0, 229, 255, 70),
        )
        self.rd_plot.addItem(self.rd_cfar_marker)
        self.rd_plot.addItem(self.rd_click_marker)
        self.rd_plot.scene().sigMouseClicked.connect(self.on_rd_mouse_clicked)
        plot_layout.addWidget(self.rd_plot, stretch=3)

        self.md_plot = pg.PlotWidget(title="Micro-Doppler Spectrum")
        self.md_plot.setLabel("left", "Doppler")
        self.md_plot.setLabel("bottom", "Time")
        self.md_plot.showGrid(x=True, y=True, alpha=0.2)
        self.md_img = pg.ImageItem(axisOrder="row-major")
        self.md_img.setLookupTable(self._md_colormap.getLookupTable())
        self.md_plot.addItem(self.md_img)
        self.md_colorbar = pg.ColorBarItem(values=(0, 60), colorMap=self._md_colormap, interactive=False)
        self.md_colorbar.setImageItem(self.md_img, insert_in=self.md_plot.plotItem)
        plot_layout.addWidget(self.md_plot, stretch=2)

        self.phase_curve_plot = pg.PlotWidget(title="Phase-Channel Curve @ Top Target")
        self.phase_curve_plot.setLabel("left", "Phase (rad, unwrapped)")
        self.phase_curve_plot.setLabel("bottom", "Channel Index")
        self.phase_curve_plot.showGrid(x=True, y=True, alpha=0.3)
        self.phase_curve_plot.setYRange(-10.0, 10.0, padding=0.0)
        self.phase_curve_plot.addLegend(offset=(8, 8))
        self.phase_curve_raw_item = self.phase_curve_plot.plot(
            [], [], pen=pg.mkPen("#f7d154", width=2), symbol="o", symbolSize=6, name="Raw"
        )
        self.phase_curve_comp_item = self.phase_curve_plot.plot(
            [], [], pen=pg.mkPen("#4cc9f0", width=2), symbol="x", symbolSize=7, name="Calibrated"
        )
        plot_layout.addWidget(self.phase_curve_plot, stretch=1)

        control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        control_panel.setFixedWidth(self._control_panel_width)
        control_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        root_layout.addWidget(control_panel, stretch=2)

        self.lbl_display = QtWidgets.QLabel("Display: CH1")
        self.lbl_fps = QtWidgets.QLabel("FPS: 0.0")
        self.lbl_queue = QtWidgets.QLabel("Queue: 0")
        self.lbl_sender = QtWidgets.QLabel("Sender: Detecting...")
        self.lbl_params = QtWidgets.QLabel("Params: waiting...")
        self.lbl_buffer = QtWidgets.QLabel("MD Buffer: backend")
        self.lbl_cfar = QtWidgets.QLabel("Detector: off")
        self.lbl_phase_sync = QtWidgets.QLabel("Phase Sync: waiting synchronized aggregate frame")
        self.lbl_phase_clicked = QtWidgets.QLabel("Phase@Clicked: click RD map to query")
        self.lbl_aoa_status = QtWidgets.QLabel("AoA: waiting calibration/click")
        self.lbl_status = QtWidgets.QLabel("Status: Waiting for backend stream...")

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
            self.lbl_status,
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
            self.lbl_status,
        ]:
            lbl.setFont(fixed_font)
            lbl.setFixedWidth(self._status_label_text_width)
            self._set_status_label_text(lbl, lbl.text())
        self.targets_text.setFont(fixed_font)
        self.targets_text.setPlainText("Top Targets: waiting detector hits")

        control_layout.addSpacing(16)

        channel_row = QtWidgets.QHBoxLayout()
        channel_row.addWidget(QtWidgets.QLabel("Display CH:"))
        self.combo_channel = QtWidgets.QComboBox()
        self.combo_channel.currentIndexChanged.connect(self._on_channel_changed)
        channel_row.addWidget(self.combo_channel)
        control_layout.addLayout(channel_row)

        control_layout.addSpacing(16)

        range_bin_row = QtWidgets.QHBoxLayout()
        range_bin_row.addWidget(QtWidgets.QLabel("Range Bin:"))
        self.txt_range_bin = QtWidgets.QLineEdit("0")
        btn_set_range_bin = QtWidgets.QPushButton("Set")
        btn_set_range_bin.clicked.connect(self._set_range_bin)
        range_bin_row.addWidget(self.txt_range_bin)
        range_bin_row.addWidget(btn_set_range_bin)
        control_layout.addLayout(range_bin_row)

        delay_view_row = QtWidgets.QHBoxLayout()
        delay_view_row.addWidget(QtWidgets.QLabel("Delay View:"))
        self.txt_delay_view = QtWidgets.QLineEdit(str(runtime.launch_cfg.display_range_bins))
        btn_set_delay_view = QtWidgets.QPushButton("Set")
        btn_set_delay_view.clicked.connect(self._set_delay_view_bins)
        delay_view_row.addWidget(self.txt_delay_view)
        delay_view_row.addWidget(btn_set_delay_view)
        control_layout.addLayout(delay_view_row)

        doppler_view_row = QtWidgets.QHBoxLayout()
        doppler_view_row.addWidget(QtWidgets.QLabel("Doppler View:"))
        self.txt_doppler_view = QtWidgets.QLineEdit(str(runtime.launch_cfg.display_doppler_bins))
        btn_set_doppler_view = QtWidgets.QPushButton("Set")
        btn_set_doppler_view.clicked.connect(self._set_doppler_view_bins)
        doppler_view_row.addWidget(self.txt_doppler_view)
        doppler_view_row.addWidget(btn_set_doppler_view)
        control_layout.addLayout(doppler_view_row)

        center_freq_row = QtWidgets.QHBoxLayout()
        center_freq_row.addWidget(QtWidgets.QLabel("Center Freq(GHz):"))
        self.txt_center_freq_ghz = QtWidgets.QLineEdit("2.4")
        btn_center_freq = QtWidgets.QPushButton("Set")
        btn_center_freq.clicked.connect(self.set_center_freq)
        center_freq_row.addWidget(self.txt_center_freq_ghz)
        center_freq_row.addWidget(btn_center_freq)
        control_layout.addLayout(center_freq_row)

        sector_range_row = QtWidgets.QHBoxLayout()
        sector_range_row.addWidget(QtWidgets.QLabel("Sector Zero(bin):"))
        self.txt_sector_zero_bin = QtWidgets.QLineEdit(str(self.target_sector_zero_range_bin))
        sector_range_row.addWidget(self.txt_sector_zero_bin)
        sector_range_row.addWidget(QtWidgets.QLabel("Sector Max(bin):"))
        self.txt_sector_max_bin = QtWidgets.QLineEdit(str(self.target_sector_max_range_bins))
        sector_range_row.addWidget(self.txt_sector_max_bin)
        btn_sector_range = QtWidgets.QPushButton("Apply")
        btn_sector_range.clicked.connect(self.set_target_sector_range)
        sector_range_row.addWidget(btn_sector_range)
        control_layout.addLayout(sector_range_row)

        self.btn_calibrate = QtWidgets.QPushButton("Calibrate Phase")
        self.btn_calibrate.clicked.connect(self.calibrate_phase_bias)
        control_layout.addWidget(self.btn_calibrate)
        self._refresh_phase_calibration_button()

        control_layout.addSpacing(20)

        self.btn_md = QtWidgets.QPushButton("Micro-Doppler: ON")
        self.btn_md.setCheckable(True)
        self.btn_md.setChecked(True)
        self.btn_md.clicked.connect(self._toggle_md)
        self._update_toggle_style(self.btn_md, True)
        control_layout.addWidget(self.btn_md)

        self.btn_cfar = QtWidgets.QPushButton("Backend OS-CFAR: OFF")
        self.btn_cfar.setCheckable(True)
        self.btn_cfar.setChecked(False)
        self.btn_cfar.clicked.connect(self._toggle_cfar)
        self._update_toggle_style(self.btn_cfar, False)
        control_layout.addWidget(self.btn_cfar)

        self.btn_mti = QtWidgets.QPushButton("MTI")
        self.btn_mti.setCheckable(True)
        self.btn_mti.clicked.connect(self._toggle_mti)
        self._update_toggle_style(self.btn_mti, False)
        control_layout.addWidget(self.btn_mti)

        self.backend_cfar_controls = QtWidgets.QWidget()
        backend_cfar_layout = QtWidgets.QVBoxLayout(self.backend_cfar_controls)
        backend_cfar_layout.setContentsMargins(0, 0, 0, 0)
        backend_cfar_layout.setSpacing(6)

        cfar_train_layout = QtWidgets.QHBoxLayout()
        cfar_train_layout.addWidget(QtWidgets.QLabel("Train D:"))
        self.txt_cfar_train_d = QtWidgets.QLineEdit("20")
        cfar_train_layout.addWidget(self.txt_cfar_train_d)
        cfar_train_layout.addWidget(QtWidgets.QLabel("Train R:"))
        self.txt_cfar_train_r = QtWidgets.QLineEdit("20")
        cfar_train_layout.addWidget(self.txt_cfar_train_r)
        backend_cfar_layout.addLayout(cfar_train_layout)

        cfar_guard_layout = QtWidgets.QHBoxLayout()
        cfar_guard_layout.addWidget(QtWidgets.QLabel("Guard D:"))
        self.txt_cfar_guard_d = QtWidgets.QLineEdit("10")
        cfar_guard_layout.addWidget(self.txt_cfar_guard_d)
        cfar_guard_layout.addWidget(QtWidgets.QLabel("Guard R:"))
        self.txt_cfar_guard_r = QtWidgets.QLineEdit("10")
        cfar_guard_layout.addWidget(self.txt_cfar_guard_r)
        backend_cfar_layout.addLayout(cfar_guard_layout)

        cfar_misc_layout = QtWidgets.QHBoxLayout()
        cfar_misc_layout.addWidget(QtWidgets.QLabel("Alpha(dB):"))
        self.txt_cfar_alpha = QtWidgets.QLineEdit(f"{float(10.0 * np.log10(50.0)):.2f}")
        cfar_misc_layout.addWidget(self.txt_cfar_alpha)
        cfar_misc_layout.addWidget(QtWidgets.QLabel("Min R:"))
        self.txt_cfar_min_range = QtWidgets.QLineEdit("0")
        cfar_misc_layout.addWidget(self.txt_cfar_min_range)
        cfar_misc_layout.addWidget(QtWidgets.QLabel("Min P(dB):"))
        self.txt_cfar_min_power = QtWidgets.QLineEdit("0.0")
        cfar_misc_layout.addWidget(self.txt_cfar_min_power)
        backend_cfar_layout.addLayout(cfar_misc_layout)

        backend_os_rank_layout = QtWidgets.QHBoxLayout()
        backend_os_rank_layout.addWidget(QtWidgets.QLabel("Rank(%):"))
        self.txt_backend_cfar_rank = QtWidgets.QLineEdit("75")
        backend_os_rank_layout.addWidget(self.txt_backend_cfar_rank)
        backend_cfar_layout.addLayout(backend_os_rank_layout)

        backend_os_suppress_layout = QtWidgets.QHBoxLayout()
        backend_os_suppress_layout.addWidget(QtWidgets.QLabel("Supp D:"))
        self.txt_backend_cfar_suppress_d = QtWidgets.QLineEdit("2")
        backend_os_suppress_layout.addWidget(self.txt_backend_cfar_suppress_d)
        backend_os_suppress_layout.addWidget(QtWidgets.QLabel("Supp R:"))
        self.txt_backend_cfar_suppress_r = QtWidgets.QLineEdit("2")
        backend_os_suppress_layout.addWidget(self.txt_backend_cfar_suppress_r)
        backend_cfar_layout.addLayout(backend_os_suppress_layout)

        md_row = QtWidgets.QHBoxLayout()
        md_row.addWidget(QtWidgets.QLabel("MD Range Bin:"))
        self.txt_md_range = QtWidgets.QLineEdit("0")
        md_row.addWidget(self.txt_md_range)
        backend_cfar_layout.addLayout(md_row)

        cfar_dc_layout = QtWidgets.QHBoxLayout()
        cfar_dc_layout.addWidget(QtWidgets.QLabel("DC Excl:"))
        self.txt_cfar_dc_excl = QtWidgets.QLineEdit("0")
        cfar_dc_layout.addWidget(self.txt_cfar_dc_excl)
        btn_cfar_apply = QtWidgets.QPushButton("Apply Backend OS-CFAR")
        btn_cfar_apply.clicked.connect(self._apply_backend_settings)
        cfar_dc_layout.addWidget(btn_cfar_apply)
        backend_cfar_layout.addLayout(cfar_dc_layout)
        control_layout.addWidget(self.backend_cfar_controls)

        control_layout.addSpacing(20)

        align_layout = QtWidgets.QHBoxLayout()
        align_layout.addWidget(QtWidgets.QLabel("Delay:"))
        self.txt_delay = QtWidgets.QLineEdit("0")
        btn_apply_alignment = QtWidgets.QPushButton("Apply")
        btn_apply_alignment.clicked.connect(self.apply_alignment)
        align_layout.addWidget(self.txt_delay)
        align_layout.addWidget(btn_apply_alignment)
        control_layout.addLayout(align_layout)

        quick_layout = QtWidgets.QHBoxLayout()
        for label, val in [("+57600", 57600), ("-57600", -57600), ("+10", 10), ("-10", -10), ("+1", 1), ("-1", -1)]:
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(lambda _checked=False, v=val: self._send_alignment_command(v))
            quick_layout.addWidget(btn)
        control_layout.addLayout(quick_layout)

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

        control_layout.addSpacing(20)

        btn_refresh = QtWidgets.QPushButton("Refresh Viewer Params")
        btn_refresh.clicked.connect(self.runtime.request_viewer_params)
        control_layout.addWidget(btn_refresh)

        self.btn_range_win = QtWidgets.QPushButton("Range Window")
        self.btn_range_win.setEnabled(False)
        self.btn_range_win.setToolTip("Not used in backend-only viewer")
        self._update_toggle_style(self.btn_range_win, False)
        control_layout.addWidget(self.btn_range_win)

        self.btn_doppler_win = QtWidgets.QPushButton("Doppler Window")
        self.btn_doppler_win.setEnabled(False)
        self.btn_doppler_win.setToolTip("Not used in backend-only viewer")
        self._update_toggle_style(self.btn_doppler_win, False)
        control_layout.addWidget(self.btn_doppler_win)

        control_layout.addSpacing(20)

        save_layout = QtWidgets.QHBoxLayout()
        for name, fn in [("Save Raw", self.save_raw), ("Save RD", self.save_rd), ("Save MD", self.save_md)]:
            btn = QtWidgets.QPushButton(name)
            btn.clicked.connect(fn)
            save_layout.addWidget(btn)
        control_layout.addLayout(save_layout)

        control_layout.addStretch(1)
        self.refresh_display_button_text()
        self._load_range_scale_config()
        self._update_target_sector_background(
            max_range=float(self.target_sector_max_range_bins),
            axis_unit="bin",
            scale_text=self._format_target_sector_scale_text("range bin"),
        )
        self._auto_load_latest_phase_calibration()
        self.targets_window.show()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(50)

    def refresh_display_button_text(self) -> None:
        count = self.runtime.channel_count()
        if count <= 0:
            self._selected_channel = 0
            self.combo_channel.setEnabled(False)
            self._set_status_label_text(self.lbl_display, "Display: N/A")
            self.phase_curve_plot.setVisible(False)
            self._refresh_phase_calibration_button()
            return

        self._selected_channel = max(0, min(self._selected_channel, count - 1))
        if self.combo_channel.count() != count:
            self.combo_channel.blockSignals(True)
            self.combo_channel.clear()
            for ch_id in range(count):
                self.combo_channel.addItem(f"CH{ch_id + 1}", ch_id)
            self.combo_channel.blockSignals(False)
        if self.combo_channel.currentIndex() != self._selected_channel:
            self.combo_channel.blockSignals(True)
            self.combo_channel.setCurrentIndex(self._selected_channel)
            self.combo_channel.blockSignals(False)
        self._set_status_label_text(
            self.lbl_display,
            "Display: N/A" if count <= 0 else f"Display: CH{self._selected_channel + 1}",
        )
        self.combo_channel.setEnabled(count > 1)
        self.phase_curve_plot.setVisible(count >= 2)
        self._refresh_phase_calibration_button()

    def _on_channel_changed(self, index: int) -> None:
        self._selected_channel = max(0, index)
        self._last_render_key = None
        self._force_ui_refresh = True
        self.update_phase_probe_text()
        self.refresh_display_button_text()
        print(f"Display channel switched to CH{self._selected_channel + 1}")

    def _set_status_label_text(self, label: QtWidgets.QLabel, text: str) -> None:
        elided = label.fontMetrics().elidedText(
            text,
            QtCore.Qt.TextElideMode.ElideRight,
            self._status_label_text_width,
        )
        label.setText(elided)
        label.setToolTip(text)

    def _update_toggle_style(self, btn: QtWidgets.QPushButton, state: bool) -> None:
        btn.setChecked(bool(state))
        if state:
            btn.setStyleSheet("background-color: lightgreen;")
        else:
            btn.setStyleSheet("background-color: lightgray;")

    def _refresh_toggle_text(self) -> None:
        self.btn_md.setText(f"Micro-Doppler: {'ON' if self.btn_md.isChecked() else 'OFF'}")
        self.btn_cfar.setText(f"Backend OS-CFAR: {'ON' if self.btn_cfar.isChecked() else 'OFF'}")

    def _toggle_cfar(self, _checked: bool = False) -> None:
        self._update_toggle_style(self.btn_cfar, self.btn_cfar.isChecked())
        self._refresh_toggle_text()
        self.runtime.send_control_command(b"CFEN", 1 if self.btn_cfar.isChecked() else 0)

    def _toggle_md(self, _checked: bool = False) -> None:
        self._update_toggle_style(self.btn_md, self.btn_md.isChecked())
        self._refresh_toggle_text()
        self.runtime.send_control_command(b"MDEN", 1 if self.btn_md.isChecked() else 0)

    def _toggle_mti(self, _checked: bool = False) -> None:
        self._update_toggle_style(self.btn_mti, self.btn_mti.isChecked())
        self.runtime.send_control_command(b"MTI ", 1 if self.btn_mti.isChecked() else 0)

    def _send_alignment_command(self, value: int) -> None:
        if self.runtime.launch_cfg.mode == "mono":
            self.runtime.send_control_command(b"ALCH", int(self._selected_channel))
        self.runtime.send_control_command(b"ALGN", int(value))

    def _set_range_bin(self) -> None:
        try:
            value = max(0, int(self.txt_range_bin.text()))
        except ValueError:
            print("Invalid range bin")
            return
        self.txt_range_bin.setText(str(value))
        self.runtime.send_control_command(b"MDRB", value)

    def _set_delay_view_bins(self) -> None:
        try:
            value = max(1, int(self.txt_delay_view.text()))
        except ValueError:
            print("Invalid delay display bins")
            return
        self.runtime.launch_cfg.display_range_bins = value
        self.txt_delay_view.setText(str(value))
        self._last_render_key = None

    def _set_doppler_view_bins(self) -> None:
        try:
            value = max(1, int(self.txt_doppler_view.text()))
        except ValueError:
            print("Invalid doppler display bins")
            return
        self.runtime.launch_cfg.display_doppler_bins = value
        self.txt_doppler_view.setText(str(value))
        self._last_render_key = None

    def apply_alignment(self) -> None:
        try:
            self._send_alignment_command(int(self.txt_delay.text()))
        except ValueError:
            print("Invalid alignment value")

    def set_strd(self) -> None:
        try:
            value = max(1, int(self.txt_strd.text()))
        except ValueError:
            print("Invalid STRD value")
            return
        self.txt_strd.setText(str(value))
        self.runtime.send_control_command(b"STRD", value)
        self.runtime.request_viewer_params()

    def set_tx_gain(self) -> None:
        try:
            gain_db = float(self.txt_tx_gain.text())
        except ValueError:
            print("Invalid TX gain value")
            return
        gain_x10 = int(round(gain_db * 10.0))
        self.txt_tx_gain.setText(f"{gain_db:.1f}")
        self.runtime.send_control_command(b"TXGN", gain_x10)

    def set_rx_gain(self) -> None:
        try:
            gain_db = float(self.txt_rx_gain.text())
        except ValueError:
            print("Invalid RX gain value")
            return
        gain_x10 = int(round(gain_db * 10.0))
        self.txt_rx_gain.setText(f"{gain_db:.1f}")
        if self.runtime.launch_cfg.mode == "mono":
            self.runtime.send_control_command(b"ALCH", int(self._selected_channel))
        self.runtime.send_control_command(b"RXGN", gain_x10)

    def _apply_backend_settings(self) -> None:
        try:
            cfar_train_d = max(0, int(self.txt_cfar_train_d.text()))
            cfar_train_r = max(0, int(self.txt_cfar_train_r.text()))
            cfar_guard_d = max(0, int(self.txt_cfar_guard_d.text()))
            cfar_guard_r = max(0, int(self.txt_cfar_guard_r.text()))
            cfar_alpha_db = float(self.txt_cfar_alpha.text())
            cfar_min_range = max(0, int(self.txt_cfar_min_range.text()))
            cfar_min_power = float(self.txt_cfar_min_power.text())
            cfar_dc_excl = max(0, int(self.txt_cfar_dc_excl.text()))
            rank = int(round(float(self.txt_backend_cfar_rank.text()) * 100.0))
            suppress_d = max(0, int(self.txt_backend_cfar_suppress_d.text()))
            suppress_r = max(0, int(self.txt_backend_cfar_suppress_r.text()))
            md_range_bin = max(0, int(self.txt_md_range.text()))
        except ValueError:
            print("Invalid backend setting value")
            return

        self.runtime.send_control_command(b"CFTD", cfar_train_d)
        self.runtime.send_control_command(b"CFTR", cfar_train_r)
        self.runtime.send_control_command(b"CFGD", cfar_guard_d)
        self.runtime.send_control_command(b"CFGR", cfar_guard_r)
        self.runtime.send_control_command(b"CFAL", int(round(cfar_alpha_db * 100.0)))
        self.runtime.send_control_command(b"CFMR", cfar_min_range)
        self.runtime.send_control_command(b"CFDC", cfar_dc_excl)
        self.runtime.send_control_command(b"CFMP", int(round(cfar_min_power * 100.0)))
        self.runtime.send_control_command(b"CFRK", rank)
        self.runtime.send_control_command(b"CFSD", suppress_d)
        self.runtime.send_control_command(b"CFSR", suppress_r)
        self.runtime.send_control_command(b"MDRB", md_range_bin)
        self.runtime.request_viewer_params()

    def _get_display_channel_runtime(self) -> ChannelState | None:
        return self.runtime.get_channel(self._selected_channel)

    def set_center_freq(self) -> None:
        try:
            self.center_freq_hz = max(1.0, float(self.txt_center_freq_ghz.text()) * 1e9)
            self.txt_center_freq_ghz.setText(f"{self.center_freq_hz / 1e9:.6f}")
            ch = self._get_display_channel_runtime()
            self.update_top_targets_text(ch.latest_frame if ch is not None else None)
            self.update_phase_probe_text()
        except ValueError:
            print(f"Invalid center frequency: {self.txt_center_freq_ghz.text()}")
            self.txt_center_freq_ghz.setText(f"{self.center_freq_hz / 1e9:.6f}")

    def set_target_sector_range(self) -> None:
        try:
            zero_bin = max(0, int(self.txt_sector_zero_bin.text()))
            max_bin = max(1, int(self.txt_sector_max_bin.text()))
            self.target_sector_zero_range_bin = zero_bin
            self.target_sector_max_range_bins = max_bin
            self.txt_sector_zero_bin.setText(str(self.target_sector_zero_range_bin))
            self.txt_sector_max_bin.setText(str(self.target_sector_max_range_bins))
            ch = self._get_display_channel_runtime()
            self.update_top_targets_text(ch.latest_frame if ch is not None else None)
        except ValueError:
            self.txt_sector_zero_bin.setText(str(self.target_sector_zero_range_bin))
            self.txt_sector_max_bin.setText(str(self.target_sector_max_range_bins))

    def _refresh_phase_calibration_button(self) -> None:
        if len(self.runtime.channels()) < 2:
            self.btn_calibrate.setEnabled(False)
            self.btn_calibrate.setText("Calibrate Phase")
            self.btn_calibrate.setStyleSheet("")
            return
        self.btn_calibrate.setEnabled(True)
        if self.phase_calibration_active:
            count = len(self.phase_calibration_samples)
            self.btn_calibrate.setText(f"Stop Calibration ({count}/{self.phase_calibration_target_samples})")
            self.btn_calibrate.setStyleSheet("background-color: khaki;")
        else:
            self.btn_calibrate.setText("Start Phase Calibration")
            self.btn_calibrate.setStyleSheet("")

    def _normalize_channel_bias_vector(self, bias_vector: np.ndarray) -> np.ndarray:
        channels = self.runtime.channel_count()
        vec = np.asarray(bias_vector, dtype=np.complex128)
        if vec.ndim != 1 or vec.shape[0] != channels:
            return np.ones(channels, dtype=np.complex128)
        if not np.all(np.isfinite(vec.real)) or not np.all(np.isfinite(vec.imag)):
            return np.ones(channels, dtype=np.complex128)
        ref = vec[0]
        if np.abs(ref) <= 1e-12:
            return np.ones(channels, dtype=np.complex128)
        vec = vec / ref
        vec[0] = 1.0 + 0.0j
        return vec

    def _format_complex_bias_vector(self, bias_vector: np.ndarray) -> str:
        mags = np.abs(bias_vector)
        phases = np.unwrap(np.angle(bias_vector).astype(np.float64))
        return (
            f"mag={np.array2string(mags, precision=3, separator=',')}, "
            f"phase(rad, CH1-ref)={np.array2string(phases, precision=3, separator=',')}"
        )

    def _reset_phase_calibration_collection(self) -> None:
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
        status: str,
        samples: np.ndarray,
        frame_ids: np.ndarray,
        range_indices: np.ndarray,
        doppler_bins: np.ndarray,
        inlier_mask: np.ndarray,
        threshold: float,
        bias_vector: np.ndarray,
    ) -> None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fname = f"./capture/capture_phase_calibration_{status}_{ts}.mat"
        try:
            os.makedirs("./capture", exist_ok=True)
            bias_vec = self._normalize_channel_bias_vector(bias_vector)
            payload = {
                "status": status,
                "channel_bias_vector": np.asarray(bias_vec, dtype=np.complex128),
                "channel_bias_magnitude": np.abs(bias_vec).astype(np.float64, copy=False),
                "channel_bias_phase_rad": np.unwrap(np.angle(bias_vec).astype(np.float64)),
                "calibration_samples": np.asarray(samples, dtype=np.complex128),
                "calibration_frame_ids": np.asarray(frame_ids, dtype=np.int64),
                "calibration_range_indices": np.asarray(range_indices, dtype=np.int32),
                "calibration_doppler_bins": np.asarray(doppler_bins, dtype=np.float64),
                "calibration_inlier_mask": np.asarray(inlier_mask, dtype=bool),
                "calibration_threshold": np.array(float(threshold), dtype=np.float64),
                "calibration_sample_count": np.array(int(len(samples)), dtype=np.int32),
                "calibration_inlier_count": np.array(int(np.count_nonzero(inlier_mask)), dtype=np.int32),
                "center_freq_hz": np.array(float(self.center_freq_hz), dtype=np.float64),
                "display_channel": np.array(int(self._selected_channel), dtype=np.int32),
                "target_sector_zero_range_bin": np.array(int(self.target_sector_zero_range_bin), dtype=np.int32),
                "target_sector_max_range_bins": np.array(int(self.target_sector_max_range_bins), dtype=np.int32),
                "range_scale_sample_rate_hz": np.array(
                    float(self.range_scale_sample_rate_hz) if self.range_scale_sample_rate_hz is not None else np.nan,
                    dtype=np.float64,
                ),
            }
            sio.savemat(fname, payload)
            self.phase_calibration_last_saved_path = fname
            print(f"Saved phase calibration data to {fname}")
        except Exception as exc:
            print(f"Error saving phase calibration data: {exc}")

    def _load_phase_calibration_data_from_file(self, path: Path) -> bool:
        try:
            data = sio.loadmat(path)
        except Exception as exc:
            print(f"Phase calibration auto-load skipped for {path}: {exc}")
            return False
        bias_raw = data.get("channel_bias_vector")
        if bias_raw is None:
            return False
        bias_vec = np.asarray(bias_raw, dtype=np.complex128).ravel()
        if bias_vec.size != self.runtime.channel_count():
            return False
        if not np.all(np.isfinite(bias_vec.real)) or not np.all(np.isfinite(bias_vec.imag)):
            return False
        self.channel_bias_vector = self._normalize_channel_bias_vector(bias_vec)
        self.phase_calibrated = True
        self.phase_calibration_last_saved_path = str(path)
        center_freq_raw = data.get("center_freq_hz")
        if center_freq_raw is not None:
            center_freq_arr = np.asarray(center_freq_raw, dtype=np.float64).ravel()
            if center_freq_arr.size > 0 and np.isfinite(center_freq_arr[0]) and float(center_freq_arr[0]) > 0.0:
                self.center_freq_hz = float(center_freq_arr[0])
                self.txt_center_freq_ghz.setText(f"{self.center_freq_hz / 1e9:.6f}")
        print(f"Auto-loaded phase calibration from {path}: {self._format_complex_bias_vector(self.channel_bias_vector)}")
        return True

    def _auto_load_latest_phase_calibration(self) -> bool:
        repo_root = Path(__file__).resolve().parent.parent
        candidate_dirs = []
        for candidate in [Path.cwd() / "capture", repo_root / "capture"]:
            if candidate not in candidate_dirs:
                candidate_dirs.append(candidate)
        calibration_files: list[Path] = []
        for directory in candidate_dirs:
            if directory.is_dir():
                calibration_files.extend(directory.glob("capture_phase_calibration_success_*.mat"))
        for path in sorted(calibration_files, key=lambda item: item.stat().st_mtime, reverse=True):
            if self._load_phase_calibration_data_from_file(path):
                return True
        return False

    def _load_range_scale_config(self) -> None:
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
            except Exception:
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
            self.range_scale_source = str(path)
            return

    def _resolve_target_range_axis(self, frame: DisplayFrame | None) -> tuple[float, str, str]:
        if frame is None:
            return 1.0, "bin", "range bin"
        params = frame.viewer_params
        raw_cols = max(1, int(params.active_cols if params.active_cols > 0 else params.wire_cols))
        range_fft_size = max(1, int(params.range_fft_size if params.range_fft_size > 0 else params.wire_cols))
        if self.range_scale_sample_rate_hz is None or self.range_scale_sample_rate_hz <= 0.0:
            return 1.0, "bin", "range bin"
        delay_bin_s = float(raw_cols) / (float(range_fft_size) * float(self.range_scale_sample_rate_hz))
        range_bin_spacing_m = 0.5 * C_LIGHT_MPS * delay_bin_s
        if not np.isfinite(range_bin_spacing_m) or range_bin_spacing_m <= 0.0:
            return 1.0, "bin", "range bin"
        return float(range_bin_spacing_m), "m", f"approx m from {self.range_scale_source}"

    def _format_target_sector_scale_text(self, base_scale_text: str) -> str:
        return (
            f"{base_scale_text}; zero=R{self.target_sector_zero_range_bin}; "
            f"max={self.target_sector_max_range_bins}bin"
        )

    def _target_sector_max_display_range(self, range_scale: float) -> float:
        return max(1.0, float(self.target_sector_max_range_bins) * float(range_scale))

    def _clear_target_sector_labels(self) -> None:
        for item in self.target_sector_label_items:
            self.target_sector_plot.removeItem(item)
        self.target_sector_label_items = []

    def _update_target_sector_background(self, max_range: float, axis_unit: str, scale_text: str) -> None:
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
            item.setData([0.0, max_range * np.sin(angle_rad)], [0.0, max_range * np.cos(angle_rad)])
        axis_label = f"Range ({axis_unit})"
        self.target_sector_plot.setLabel("left", axis_label)
        self.target_sector_plot.setLabel("bottom", f"Cross-Range ({axis_unit})")
        self.target_sector_plot.setXRange(-1.08 * max_range, 1.08 * max_range, padding=0.0)
        self.target_sector_plot.setYRange(0.0, 1.08 * max_range, padding=0.0)
        self.target_sector_plot.setTitle(f"Target Sector View [{scale_text}]")

    def _update_target_sector_plot(self, entries: list[dict], axis_unit: str, scale_text: str, range_scale: float) -> None:
        self._clear_target_sector_labels()
        max_range = self._target_sector_max_display_range(range_scale)
        self._update_target_sector_background(max_range=max_range, axis_unit=axis_unit, scale_text=scale_text)
        if not entries:
            self.target_sector_points_item.setData([], [])
            return
        valid_entries = [
            entry for entry in entries
            if entry.get("aoa_deg") is not None and float(entry.get("range_value", -1.0)) >= 0.0
        ]
        if not valid_entries:
            self.target_sector_points_item.setData([], [])
            self.target_sector_plot.setTitle(f"Target Sector View [{scale_text}; AoA pending or target before zero bin]")
            return
        valid_entries = valid_entries[:TARGET_SECTOR_POINT_LIMIT]
        spots = []
        for entry in valid_entries:
            theta_rad = np.deg2rad(float(entry["aoa_deg"]))
            radius = float(entry["range_value"])
            x_pos = radius * np.sin(theta_rad)
            y_pos = max(0.0, radius * np.cos(theta_rad))
            color = "#ff7f50" if int(entry["rank"]) == 1 else "#4cc9f0"
            spots.append({
                "pos": (x_pos, y_pos),
                "size": 16 if int(entry["rank"]) == 1 else 13,
                "brush": pg.mkBrush(color),
                "pen": pg.mkPen("#0b132b", width=1.2),
            })
            label = pg.TextItem(text=f"{int(entry['rank'])}", color="w", anchor=(0.5, 1.0))
            label.setPos(x_pos, y_pos)
            self.target_sector_plot.addItem(label)
            self.target_sector_label_items.append(label)
        self.target_sector_points_item.setData(spots)

    def _build_top_target_entries(self, frame: DisplayFrame | None):
        if frame is None:
            return None, "Top Targets: N/A", "bin", "range bin", 1.0
        range_scale, axis_unit, base_scale_text = self._resolve_target_range_axis(frame)
        scale_text = self._format_target_sector_scale_text(base_scale_text)
        clusters = list(frame.target_clusters or [])
        if not clusters:
            return [], "Top Targets: none", axis_unit, scale_text, range_scale
        header = f"Top Targets CH{frame.ch_id + 1} F{frame.frame_id} [{scale_text}]"
        lines = [
            header,
            (
                "Detector=backend CFAR | "
                f"OS-Suppress(D,R)=({frame.viewer_params.backend_os_suppress_doppler},"
                f"{frame.viewer_params.backend_os_suppress_range}) | "
                f"List={TARGET_TEXT_POINT_LIMIT} Plot={TARGET_SECTOR_POINT_LIMIT}"
            ),
        ]
        entries = []
        have_auto_aoa = (
            self.runtime.channel_count() >= 2
            and self.phase_ready
            and self.synced_frame_id is not None
            and int(frame.frame_id) == int(self.synced_frame_id)
        )
        for rank, cluster in enumerate(clusters[:TARGET_TEXT_POINT_LIMIT], start=1):
            d_idx = int(cluster["peak_doppler_idx"])
            r_idx = int(cluster["peak_range_idx"])
            strength_db = float(cluster["peak_strength_db"])
            doppler_bin = self.rd_doppler_min + d_idx
            centroid_d = float(cluster["centroid_doppler_idx"])
            centroid_r = float(cluster["centroid_range_idx"])
            relative_range_bins = float(r_idx) - float(self.target_sector_zero_range_bin)
            range_value = relative_range_bins * float(range_scale)
            range_text = f"{range_value:+6.2f}m" if axis_unit == "m" else f"{relative_range_bins:+5.0f}bin"
            aoa_deg = None
            aoa_text = "AoA=wait" if self.runtime.channel_count() >= 2 else "AoA=n/a"
            if have_auto_aoa:
                phase_result = self._compute_phase_vectors_at(d_idx, r_idx)
                if phase_result is not None:
                    _, _, _, _, phase_comp = phase_result
                    aoa_deg, _ = self._estimate_aoa_from_phase(phase_comp)
                    if aoa_deg is not None:
                        aoa_text = f"AoA={aoa_deg:+6.1f}deg"
            lines.append(
                f"{rank}. R={r_idx:4d} (rel={range_text}) D={doppler_bin:+5.0f} "
                f"S={strength_db:6.1f}dB N={int(cluster['cluster_size']):2d} "
                f"C=({centroid_r:5.1f},{centroid_d:5.1f}) {aoa_text}"
            )
            entries.append({
                "rank": rank,
                "range_idx": r_idx,
                "relative_range_bins": relative_range_bins,
                "range_value": range_value,
                "aoa_deg": aoa_deg,
                "strength_db": strength_db,
            })
        return entries, "\n".join(lines), axis_unit, scale_text, range_scale

    def update_top_targets_text(self, frame: DisplayFrame | None) -> None:
        entries, text, axis_unit, scale_text, range_scale = self._build_top_target_entries(frame)
        self.targets_text.setPlainText(text)
        if entries is None:
            self._update_target_sector_plot([], "bin", self._format_target_sector_scale_text("range bin"), 1.0)
            return
        self._update_target_sector_plot(entries, axis_unit, scale_text, range_scale)

    def _format_phase_calibration_status(self) -> str:
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

    def _set_aoa_status_text(self, text: str) -> None:
        if self.phase_calibration_active:
            self._set_status_label_text(self.lbl_aoa_status, self._format_phase_calibration_status())
            return
        self._set_status_label_text(self.lbl_aoa_status, text)

    def _handle_runtime_channel_count_change(self) -> None:
        current_count = self.runtime.channel_count()
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
        self._last_render_key = None
        self._force_ui_refresh = True
        print(
            "Viewer detected channel-count change: "
            f"{previous_count} -> {current_count}; attempting to auto-load matching calibration"
        )
        self._auto_load_latest_phase_calibration()

    def _start_phase_calibration(self) -> None:
        self.phase_calibration_active = True
        self._reset_phase_calibration_collection()
        self._refresh_phase_calibration_button()
        self._set_aoa_status_text(self._format_phase_calibration_status())

    def _cancel_phase_calibration(self, reason: str) -> None:
        sample_count = len(self.phase_calibration_samples)
        self.phase_calibration_active = False
        self._reset_phase_calibration_collection()
        self._refresh_phase_calibration_button()
        print(f"Phase calibration cancelled: {reason} (collected {sample_count} valid samples)")

    def _estimate_phase_bias_from_samples(self, samples: np.ndarray):
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

    def _finish_phase_calibration(self) -> None:
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
                fail_bias = self.channel_bias_vector if self.phase_calibrated else np.ones(self.runtime.channel_count(), dtype=np.complex128)
                self._save_phase_calibration_data(
                    "failed",
                    samples,
                    frame_ids,
                    range_indices,
                    doppler_bins,
                    np.asarray(inlier_mask, dtype=bool) if inlier_mask is not None else np.zeros(sample_count, dtype=bool),
                    float(threshold) if threshold is not None else np.nan,
                    fail_bias,
                )
            self._reset_phase_calibration_collection()
            self.update_phase_probe_text()
            return
        self.channel_bias_vector = self._normalize_channel_bias_vector(bias)
        self.phase_calibrated = True
        if PHASE_CALIBRATION_AUTO_SAVE:
            self._save_phase_calibration_data(
                "success", samples, frame_ids, range_indices, doppler_bins, np.asarray(inlier_mask, dtype=bool), float(threshold), self.channel_bias_vector
            )
        self._reset_phase_calibration_collection()
        self.update_phase_probe_text()

    def _select_calibration_target(self):
        if not self.phase_ready or self.synced_rd_complex is None:
            return None
        disp_idx = max(0, min(self._selected_channel, self.runtime.channel_count() - 1))
        synced_rd = self.synced_rd_complex[disp_idx]
        if synced_rd is None or synced_rd.size == 0:
            return None
        ch = self._get_display_channel_runtime()
        latest_frame = ch.latest_frame if ch is not None else None
        if latest_frame is not None and int(latest_frame.frame_id) == int(self.synced_frame_id):
            clusters = list(latest_frame.target_clusters or [])
            if clusters:
                strongest = clusters[0]
                return int(strongest["peak_doppler_idx"]), int(strongest["peak_range_idx"]), "detector strongest target"
        metric = np.abs(np.asarray(synced_rd))
        safe_metric = np.where(np.isfinite(metric), metric, -np.inf)
        peak_flat = int(np.argmax(safe_metric))
        d_idx, r_idx = np.unravel_index(peak_flat, safe_metric.shape)
        return int(d_idx), int(r_idx), "global max target"

    def _collect_phase_calibration_sample(self) -> None:
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
        if z.shape[0] != self.runtime.channel_count():
            return
        if not np.all(np.isfinite(z.real)) or not np.all(np.isfinite(z.imag)) or np.abs(z[0]) <= 1e-12:
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
        if sample_count >= self.phase_calibration_target_samples:
            self._finish_phase_calibration()

    def calibrate_phase_bias(self) -> None:
        if self.runtime.channel_count() < 2:
            self._set_aoa_status_text("AoA: unavailable (need >=2 channels)")
            return
        if self.phase_calibration_active:
            self._cancel_phase_calibration("stopped by user")
            self.update_phase_probe_text()
            return
        self._start_phase_calibration()
        self.update_phase_probe_text()

    def _sync_phase_frame(self) -> None:
        self.phase_ready = False
        self.synced_frame_id = None
        self.synced_rd_complex = None
        if self.runtime.channel_count() < 2:
            return
        synced_frame_id, synced_rd_complex = self.runtime.latest_phase_bundle()
        if synced_frame_id is None or synced_rd_complex is None:
            return
        if len(synced_rd_complex) < self.runtime.channel_count():
            return
        rd_complex_list = []
        ref_shape = None
        for ch_id in range(self.runtime.channel_count()):
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

    def _compute_phase_vectors_at(self, doppler_idx: int, range_idx: int):
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
        if self.channel_bias_vector.shape[0] != self.runtime.channel_count():
            self.channel_bias_vector = np.ones(self.runtime.channel_count(), dtype=np.complex128)
            self.phase_calibrated = False
        self.channel_bias_vector = self._normalize_channel_bias_vector(self.channel_bias_vector)
        safe_bias = np.where(np.abs(self.channel_bias_vector) > 1e-12, self.channel_bias_vector, 1.0 + 0.0j)
        z_comp = z.astype(np.complex128, copy=False) / safe_bias
        phase_comp_rel = np.angle(z_comp * np.conj(z_comp[0]))
        phase_comp = np.unwrap(phase_comp_rel.astype(np.float64))
        phase_comp[0] = 0.0
        return d_idx, r_idx, doppler_bin, phase_raw, phase_comp

    def _update_phase_curve_plot(self) -> None:
        if self.runtime.channel_count() < 2:
            self.phase_curve_raw_item.setData([], [])
            self.phase_curve_comp_item.setData([], [])
            return
        if self.phase_curve_raw is None:
            self.phase_curve_raw_item.setData([], [])
            self.phase_curve_comp_item.setData([], [])
            self.phase_curve_plot.setYRange(-10.0, 10.0, padding=0.0)
            return
        x = np.arange(1, self.runtime.channel_count() + 1, dtype=np.float64)
        self.phase_curve_raw_item.setData(x, self.phase_curve_raw)
        if self.phase_calibrated and self.phase_curve_comp is not None:
            self.phase_curve_comp_item.setData(x, self.phase_curve_comp)
        else:
            self.phase_curve_comp_item.setData([], [])
        self.phase_curve_plot.setXRange(0.5, self.runtime.channel_count() + 0.5, padding=0.0)
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

    def on_rd_mouse_clicked(self, event) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        ch = self._get_display_channel_runtime()
        frame = ch.latest_frame if ch is not None else None
        rd_data = None if frame is None else frame.rd
        if rd_data is None or rd_data.size == 0:
            return
        scene_pos = event.scenePos()
        if not self.rd_plot.sceneBoundingRect().contains(scene_pos):
            return
        mouse_point = self.rd_plot.plotItem.vb.mapSceneToView(scene_pos)
        x_val = float(mouse_point.x())
        y_val = float(mouse_point.y())
        rows, cols = rd_data.shape
        doppler_idx = int(np.clip(round(x_val - self.rd_doppler_min), 0, rows - 1))
        range_idx = int(np.clip(round(y_val), 0, cols - 1))
        doppler_bin = self.rd_doppler_min + doppler_idx
        self.clicked_range_idx = range_idx
        self.clicked_doppler_idx = doppler_idx
        self.rd_click_marker.setData([doppler_bin], [range_idx])
        self.update_phase_probe_text()

    def _save_mat(self, key: str, data: np.ndarray, suffix: str) -> None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"./capture/capture_{suffix}_{ts}.mat"
        try:
            os.makedirs("./capture", exist_ok=True)
            sio.savemat(fname, {key: data})
            print(f"Saved {suffix} to {fname}")
        except Exception as exc:
            print(f"Error saving {suffix}: {exc}")

    def save_raw(self) -> None:
        ch = self._get_display_channel_runtime()
        frame = ch.latest_frame if ch is not None else None
        if frame is not None and frame.rd_complex is not None:
            self._save_mat("raw_frame", frame.rd_complex, "raw")

    def save_rd(self) -> None:
        ch = self._get_display_channel_runtime()
        frame = ch.latest_frame if ch is not None else None
        if frame is not None:
            self._save_mat("rd_map", frame.rd, "rd")

    def save_md(self) -> None:
        ch = self._get_display_channel_runtime()
        frame = ch.latest_frame if ch is not None else None
        if frame is not None and frame.md is not None:
            self._save_mat("md_map", frame.md, "md")

    def update_phase_status(self) -> None:
        if self.runtime.channel_count() < 2:
            if self.phase_calibration_active:
                self._cancel_phase_calibration("need >=2 channels")
            self.phase_ready = False
            self.synced_frame_id = None
            self.synced_rd_complex = None
            self._set_status_label_text(self.lbl_phase_sync, "Phase Sync: unavailable (need >=2 channels)")
            self._set_aoa_status_text("AoA: unavailable (need >=2 channels)")
            return
        frame_tags = ", ".join([f"CH{i + 1}={ch.last_frame_id}" for i, ch in enumerate(self.runtime.channels())])
        if not self.phase_ready or self.synced_frame_id is None or self.synced_rd_complex is None:
            self._set_status_label_text(
                self.lbl_phase_sync,
                f"Phase Sync: waiting synchronized aggregate frame across {self.runtime.channel_count()} channels ({frame_tags})",
            )
            self._set_aoa_status_text(
                "AoA: waiting phase sync (calibrated)" if self.phase_calibrated else "AoA: waiting phase sync (not calibrated)"
            )
            return
        self._set_status_label_text(
            self.lbl_phase_sync,
            f"Phase Sync: locked frame {self.synced_frame_id} across {self.runtime.channel_count()} channels",
        )

    def update_phase_probe_text(self) -> None:
        if self.runtime.channel_count() < 2:
            self.phase_curve_raw = None
            self.phase_curve_comp = None
            self.phase_target_range_idx = None
            self.phase_target_doppler_idx = None
            self._update_phase_curve_plot()
            self._set_status_label_text(self.lbl_phase_clicked, "Phase@Clicked: unavailable (need >=2 channels)")
            self._set_aoa_status_text("AoA: unavailable (need >=2 channels)")
            return
        ch = self._get_display_channel_runtime()
        frame = ch.latest_frame if ch is not None else None
        if frame is None or not frame.target_clusters or self.synced_frame_id is None or int(frame.frame_id) != int(self.synced_frame_id):
            self.phase_curve_raw = None
            self.phase_curve_comp = None
            self.phase_target_range_idx = None
            self.phase_target_doppler_idx = None
            self._update_phase_curve_plot()
            self._set_status_label_text(self.lbl_phase_clicked, "Phase@Clicked: click RD map to query")
            self._set_aoa_status_text("AoA: calibrated; waiting clustered top target" if self.phase_calibrated else "AoA: not calibrated; waiting clustered top target")
            return
        top_target = frame.target_clusters[0]
        phase_result = self._compute_phase_vectors_at(int(top_target["peak_doppler_idx"]), int(top_target["peak_range_idx"]))
        if phase_result is None:
            self.phase_curve_raw = None
            self.phase_curve_comp = None
            self.phase_target_range_idx = None
            self.phase_target_doppler_idx = None
            self._update_phase_curve_plot()
            self._set_status_label_text(self.lbl_phase_clicked, "Phase@Clicked: waiting synchronized aggregate frame data")
            self._set_aoa_status_text("AoA: waiting same-frame phase (calibrated)" if self.phase_calibrated else "AoA: waiting same-frame phase")
            return
        _, r_idx, doppler_bin, phase_raw, phase_comp = phase_result
        self.phase_curve_raw = phase_raw
        self.phase_curve_comp = phase_comp
        self.phase_target_range_idx = r_idx
        self.phase_target_doppler_idx = int(top_target["peak_doppler_idx"])
        self._update_phase_curve_plot()
        aoa_deg, slope = self._estimate_aoa_from_phase(phase_comp)
        clicked_text = "Phase@Clicked: click RD map to query"
        if self.clicked_range_idx is not None and self.clicked_doppler_idx is not None:
            clicked_phase = self._compute_phase_vectors()
            if clicked_phase is not None:
                _, clicked_r_idx, clicked_doppler_bin, clicked_phase_raw, clicked_phase_comp = clicked_phase
                disp_idx = max(0, min(self._selected_channel, self.runtime.channel_count() - 1))
                clicked_text = (
                    f"Phase@Clicked R={clicked_r_idx},D={clicked_doppler_bin:+.0f},F={self.synced_frame_id}: "
                    f"CH{disp_idx + 1} raw={float(clicked_phase_raw[disp_idx]):+.3f}, "
                    f"comp={float(clicked_phase_comp[disp_idx]):+.3f} rad"
                )
            else:
                clicked_text = "Phase@Clicked: waiting synchronized aggregate frame data"
        self._set_status_label_text(self.lbl_phase_clicked, clicked_text)
        if aoa_deg is None or slope is None:
            self._set_aoa_status_text(f"AoA@Strongest(clustered top target) R={r_idx},D={doppler_bin:+.0f}: estimation unavailable")
            return
        calib_tag = "calibrated" if self.phase_calibrated else "uncalibrated"
        self._set_aoa_status_text(
            f"AoA@Strongest(clustered top target) R={r_idx},D={doppler_bin:+.0f}: {aoa_deg:+.2f} deg, "
            f"slope={slope:+.4f} rad/ch, fc={self.center_freq_hz/1e9:.3f} GHz ({calib_tag})"
        )

    def _refresh(self) -> None:
        frames = self.runtime.drain_display_queue()
        for frame in frames:
            channel = self.runtime.get_channel(frame.ch_id)
            if channel is not None:
                channel.latest_frame = frame

        if self.runtime.channel_count() != self._last_known_channel_count:
            self._handle_runtime_channel_count_change()

        if self.combo_channel.count() != self.runtime.channel_count():
            self.refresh_display_button_text()

        self._sync_phase_frame()
        self._collect_phase_calibration_sample()
        self.update_phase_status()

        channel = self.runtime.get_channel(self._selected_channel)
        if channel is None:
            self._set_status_label_text(self.lbl_sender, "Sender: N/A")
            self._set_status_label_text(self.lbl_params, "Params: N/A")
            self._set_status_label_text(self.lbl_buffer, "MD Buffer: N/A")
            self._set_status_label_text(self.lbl_cfar, "Detector: N/A")
            self._set_status_label_text(self.lbl_queue, "Q(disp): N/A")
            self.rd_click_marker.setData([], [])
            self.rd_cfar_marker.setData([], [])
            self.update_top_targets_text(None)
            self.update_phase_probe_text()
            return
        self._set_status_label_text(self.lbl_display, f"Display: CH{self._selected_channel + 1}")

        frame = channel.latest_frame
        params = self.runtime.get_viewer_params()
        sender_ip, control_port = self.runtime.sender_endpoint()

        if sender_ip is None:
            self._set_status_label_text(
                self.lbl_sender,
                f"Sender: Detecting... | listen=0.0.0.0:{self.runtime.launch_cfg.port}"
            )
        else:
            self._set_status_label_text(
                self.lbl_sender,
                f"Sender(CH{channel.ch_id + 1}): {sender_ip}:{control_port} | Frame: {channel.last_frame_id}"
            )

        params_text = (
            f"Params(CH{channel.ch_id + 1}): {params.describe()} | "
            f"view={self.runtime.launch_cfg.display_range_bins}x{self.runtime.launch_cfg.display_doppler_bins}"
        )
        last_error = self.runtime.last_error()
        if last_error:
            params_text += f" | last_error={last_error}"
        if params_text != self._last_params_text:
            self._set_status_label_text(self.lbl_params, params_text)
            self._last_params_text = params_text
            self.txt_backend_cfar_rank.setText(f"{params.backend_os_rank_percent:.0f}")
            self.txt_backend_cfar_suppress_d.setText(str(params.backend_os_suppress_doppler))
            self.txt_backend_cfar_suppress_r.setText(str(params.backend_os_suppress_range))
            self._update_toggle_style(self.btn_mti, bool(params.flags & FLAG_ENABLE_MTI))

        if frame is None:
            self._set_status_label_text(self.lbl_status, "Status: Waiting for frames...")
            self._set_status_label_text(self.lbl_buffer, "MD Buffer: waiting")
            self._set_status_label_text(self.lbl_cfar, "Detector: waiting")
            self._set_status_label_text(
                self.lbl_queue,
                f"Q(agg/disp) AGG:{self.runtime.pending_bundle_count()} DISP:{self.runtime.display_queue_size()}",
            )
            self.rd_click_marker.setData([], [])
            self.rd_cfar_marker.setData([], [])
            self.update_top_targets_text(None)
            self.update_phase_probe_text()
            return

        render_key = (frame.ch_id, frame.frame_id)
        if self._force_ui_refresh or render_key != self._last_render_key:
            rd_data = frame.rd
            self.rd_img.setImage(rd_data, autoLevels=False)
            rows, cols = rd_data.shape
            doppler_min = -0.5 * float(rows)
            self.rd_doppler_min = doppler_min
            self.rd_doppler_span = float(rows)
            self.rd_img.setRect(QtCore.QRectF(doppler_min, 0.0, float(rows), float(cols)))

            if frame.cfar_points.size > 0:
                x_pts = doppler_min + frame.cfar_points[:, 0].astype(np.float32)
                y_pts = frame.cfar_points[:, 1].astype(np.float32)
                self.rd_cfar_marker.setData(x_pts, y_pts)
            else:
                self.rd_cfar_marker.setData([], [])
            if self.clicked_range_idx is not None and self.clicked_doppler_idx is not None:
                d_idx = int(np.clip(self.clicked_doppler_idx, 0, rows - 1))
                r_idx = int(np.clip(self.clicked_range_idx, 0, cols - 1))
                doppler_bin = self.rd_doppler_min + d_idx
                self.rd_click_marker.setData([doppler_bin], [r_idx])
            else:
                self.rd_click_marker.setData([], [])

            if frame.md is not None and getattr(frame.md, "size", 0) > 0:
                self.md_img.setImage(frame.md, autoLevels=False)
                if frame.md_extent is not None and len(frame.md_extent) == 4:
                    x0, x1, y0, y1 = [float(v) for v in frame.md_extent]
                    self.md_img.setRect(
                        QtCore.QRectF(x0, y0, max(1e-6, x1 - x0), max(1e-6, y1 - y0))
                    )
            else:
                self.md_img.setImage(np.zeros((1, 1), dtype=np.float32), autoLevels=False)

            self.update_top_targets_text(frame)
            self._last_render_key = render_key
            self._frame_counter += 1

        self.update_phase_probe_text()
        self._force_ui_refresh = False
        cluster_count = len(frame.target_clusters)
        md_status = "on" if frame.md is not None and getattr(frame.md, "size", 0) > 0 else "off"
        self._refresh_toggle_text()
        self._set_status_label_text(
            self.lbl_status,
            f"Status: CH{frame.ch_id + 1} frame={frame.frame_id} "
            f"CFAR={frame.cfar_hits}/{frame.cfar_shown_hits} "
            f"targets={cluster_count} md={md_status}"
        )
        self._set_status_label_text(
            self.lbl_buffer,
            f"MD Buffer(CH{frame.ch_id + 1}): {'backend ' + str(frame.md.shape[0]) + 'x' + str(frame.md.shape[1]) if frame.md is not None and getattr(frame.md, 'size', 0) > 0 else 'backend'}"
        )
        cfar_stats = frame.cfar_stats or {}
        self._set_status_label_text(
            self.lbl_cfar,
            f"Detector(CH{frame.ch_id + 1}): bCFAR {'on' if self.btn_cfar.isChecked() else 'off'} "
            f"raw={frame.cfar_hits} sh={frame.cfar_shown_hits} "
            f"p={cfar_stats.get('power_min_db', 0.0):.0f}"
        )
        self._set_status_label_text(
            self.lbl_queue,
            f"Q(agg/disp) AGG:{self.runtime.pending_bundle_count()} DISP:{self.runtime.display_queue_size()} CH{frame.ch_id + 1}"
        )
        now = time.time()
        elapsed = now - self._last_fps_ts
        if elapsed > 0.5:
            self._set_status_label_text(self.lbl_fps, f"FPS: {self._frame_counter / elapsed:.1f} | DSP: 0.0ms")
            self._frame_counter = 0
            self._last_fps_ts = now

    def closeEvent(self, event) -> None:  # noqa: N802
        self.runtime.stop()
        self.targets_window.close()
        event.accept()


def build_parser(
    *,
    default_mode: str | None = None,
    default_title: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone OpenISAC backend sensing viewer")
    parser.add_argument(
        "--mode",
        choices=("auto", "mono", "bi"),
        default=default_mode or "auto",
        help="Viewer mode",
    )
    parser.add_argument("--port", type=int, default=None, help="Override sensing UDP port")
    parser.add_argument("--control-port", type=int, default=None, help="Override control UDP port")
    parser.add_argument("--channels", type=int, default=None, help="Override logical channel count")
    parser.add_argument(
        "--display-range-bins",
        type=int,
        default=None,
        help="Visible range bins in the RD heatmap",
    )
    parser.add_argument(
        "--display-doppler-bins",
        type=int,
        default=None,
        help="Visible doppler bins in the RD heatmap",
    )
    parser.add_argument("--title", type=str, default=default_title, help="Window title override")
    return parser


def main(
    *,
    default_mode: str | None = None,
    default_title: str | None = None,
) -> int:
    parser = build_parser(
        default_mode=default_mode,
        default_title=default_title,
    )
    args = parser.parse_args()

    launch_cfg = load_launch_defaults(
        mode_hint=args.mode,
        default_mode=default_mode,
        default_title=args.title or default_title,
    )

    if args.port is not None:
        launch_cfg.port = int(args.port)
    if args.control_port is not None:
        launch_cfg.control_port = int(args.control_port)
    if args.channels is not None:
        launch_cfg.channels = max(1, int(args.channels))
    if args.display_range_bins is not None:
        launch_cfg.display_range_bins = max(1, int(args.display_range_bins))
    if args.display_doppler_bins is not None:
        launch_cfg.display_doppler_bins = max(1, int(args.display_doppler_bins))
    if args.title:
        launch_cfg.title = str(args.title)

    app = QtWidgets.QApplication(sys.argv)
    runtime = BackendViewerRuntime(launch_cfg)
    window = MainWindow(runtime)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
