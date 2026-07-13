from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
import zmq
from PyQt6 import QtCore, QtWidgets

try:
    from viewer_endpoint_store import default_settings_key, load_endpoint, save_endpoint
except ImportError:
    from scripts.viewer_endpoint_store import default_settings_key, load_endpoint, save_endpoint


PLOT_BACKGROUND = "#0f172a"
PLOT_FOREGROUND = "#e5e7eb"
PLOT_AXIS = "#94a3b8"

pg.setConfigOptions(background=PLOT_BACKGROUND, foreground=PLOT_FOREGROUND, antialias=True)


@dataclass(frozen=True)
class ViewerArgs:
    host: str
    port: int
    interval_ms: int


def parse_viewer_args(description: str, default_port: int) -> ViewerArgs:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--host", default=None, help="Backend host or tcp:// endpoint")
    parser.add_argument("--port", type=int, default=None, help="Backend ZMQ port")
    parser.add_argument("--interval-ms", type=int, default=50, help="Plot refresh interval")
    args = parser.parse_args()
    saved_host, saved_port = load_endpoint(default_settings_key(), "127.0.0.1", default_port)
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


def normalize_endpoint(text: str, default_port: int) -> tuple[str, str, int]:
    value = (text or "").strip()
    if value.startswith("tcp://"):
        value = value[len("tcp://"):]
    if not value:
        value = "127.0.0.1"

    host = value
    port = int(default_port)
    if value.startswith("["):
        end = value.find("]")
        if end >= 0:
            host = value[1:end]
            rest = value[end + 1:].strip()
            if rest.startswith(":") and rest[1:]:
                port = int(rest[1:])
    elif value.count(":") == 1:
        host_part, port_part = value.rsplit(":", 1)
        if port_part.strip():
            host = host_part.strip() or "127.0.0.1"
            port = int(port_part)

    if not 1 <= port <= 65535:
        raise ValueError("port must be in 1..65535")
    endpoint = f"tcp://{host}:{port}"
    return endpoint, host, port


def endpoint_text(host: str, port: int) -> str:
    if ":" in host and not (host.startswith("[") and host.endswith("]")):
        return f"[{host}]:{port}"
    return f"{host}:{port}"


def make_sub_socket(endpoint: str, *, conflate: bool) -> zmq.Socket:
    """Create a SUB socket connected to ``endpoint``.

    When ``conflate`` is true we intentionally do **not** set ZMQ_CONFLATE.
    That option keeps only a single frame and is incompatible with multipart
    messages; if a multi-part publisher lands on the same port (for example
    eRTM debug on 12362 vs a single-part self-scan viewer), libzmq can abort
    the process with ``Assertion failed: !_more (fq.cpp)``.

    Instead use a small RCVHWM and let ``DebugPlotWindow.poll_socket`` keep
    only the latest fully-received message in user space.
    """
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    # Small HWM: drop old frames when the GUI poll falls behind.
    sock.setsockopt(zmq.RCVHWM, 2 if conflate else 8)
    sock.connect(endpoint)
    return sock


class DebugPlotWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        *,
        title: str,
        default_port: int,
        host: str,
        interval_ms: int,
        multipart: bool = False,
        conflate: bool = True,
    ) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.default_port = int(default_port)
        self.settings_key = default_settings_key()
        self.multipart = multipart
        self.conflate = conflate and not multipart
        self.sock: zmq.Socket | None = None

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        self.root_layout = QtWidgets.QVBoxLayout(root)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Endpoint"))
        self.endpoint_edit = QtWidgets.QLineEdit(endpoint_text(host, default_port))
        self.endpoint_edit.setMinimumWidth(280)
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.status_label = QtWidgets.QLabel("Disconnected")
        self.status_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        controls.addWidget(self.endpoint_edit, 1)
        controls.addWidget(self.connect_button)
        controls.addWidget(self.status_label, 2)
        self.root_layout.addLayout(controls)

        self.plot_area = QtWidgets.QWidget()
        self.plot_layout = QtWidgets.QVBoxLayout(self.plot_area)
        self.root_layout.addWidget(self.plot_area, 1)

        self.connect_button.clicked.connect(self.connect_endpoint)
        self.endpoint_edit.returnPressed.connect(self.connect_endpoint)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll_socket)
        self.timer.start(max(1, interval_ms))

        self.connect_endpoint()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self.sock is not None:
            self.sock.close(linger=0)
            self.sock = None
        super().closeEvent(event)

    def connect_endpoint(self) -> None:
        try:
            endpoint, host, port = normalize_endpoint(self.endpoint_edit.text(), self.default_port)
        except Exception as exc:
            self.status_label.setText(f"Invalid endpoint: {exc}")
            return

        old_sock = self.sock
        self.sock = make_sub_socket(endpoint, conflate=self.conflate)
        if old_sock is not None:
            old_sock.close(linger=0)
        self.endpoint_edit.setText(endpoint_text(host, port))
        save_endpoint(self.settings_key, host, port)
        self.status_label.setText(f"Connected {endpoint}")

    def poll_socket(self) -> None:
        if self.sock is None:
            return
        latest = None
        while True:
            try:
                # Always receive full multi-part messages. Single-part streams
                # still arrive as a one-element list; this avoids partial-frame
                # state that can trip libzmq's fq.cpp `!_more` assertion.
                parts = self.sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception as exc:
                self.status_label.setText(f"Receive error: {exc}")
                return
            if not parts:
                continue
            if self.multipart:
                latest = parts
            else:
                latest = parts[0]
        if latest is None:
            return
        try:
            self.handle_message(latest)
        except Exception as exc:
            self.status_label.setText(f"Frame error: {exc}")

    def handle_message(self, message) -> None:
        raise NotImplementedError


def configure_plot(plot: pg.PlotWidget, title: str, left: str, bottom: str) -> None:
    plot.setBackground(PLOT_BACKGROUND)
    plot.setTitle(title, color=PLOT_FOREGROUND, size="11pt")
    label_style = {"color": PLOT_FOREGROUND, "font-size": "10pt"}
    plot.setLabel("left", left, **label_style)
    plot.setLabel("bottom", bottom, **label_style)
    plot.showGrid(x=True, y=True, alpha=0.28)

    plot_item = plot.getPlotItem()
    plot_item.getViewBox().setBackgroundColor(PLOT_BACKGROUND)
    for axis_name in ("left", "bottom", "right", "top"):
        axis = plot_item.getAxis(axis_name)
        axis.setPen(pg.mkPen(PLOT_AXIS, width=1))
        axis.setTextPen(pg.mkPen(PLOT_FOREGROUND))
        axis.setGrid(80)


def power_db(values: np.ndarray) -> np.ndarray:
    power = np.abs(values).astype(np.float64, copy=False) ** 2
    np.maximum(power, 1e-60, out=power)
    return 10.0 * np.log10(power)
