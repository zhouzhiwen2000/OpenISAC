#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

try:
    from PyQt6 import QtCore, QtGui, QtWidgets
except ImportError as exc:  # pragma: no cover - startup dependency check
    raise SystemExit("PyQt6 is required. Install it with `pip install PyQt6`.") from exc

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sensing_runtime_protocol import (  # noqa: E402
    BS_DL_UL_TIMING_DIFF_COMMAND,
    CTRL_HEADER,
    REQ_HEADER,
    REQUEST_PACKET_STRUCT,
    UE_TIMING_ADVANCE_COMMAND,
    build_control_command,
    make_control_dealer,
    make_tcp_endpoint,
)

try:
    import zmq
except ImportError as exc:  # pragma: no cover - startup dependency check
    raise SystemExit("pyzmq is required. Install it with `pip install pyzmq`.") from exc


def build_status_request(command: bytes) -> bytes:
    return REQUEST_PACKET_STRUCT.pack(REQ_HEADER, command, 0)


class TimingControlRow(QtWidgets.QGroupBox):
    statusChanged = QtCore.pyqtSignal(str)

    def __init__(
        self,
        title: str,
        command: bytes,
        default_host: str,
        default_port: int,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(title, parent)
        self.command = command
        self.command_label = command.decode("ascii", errors="replace").strip()
        self._socket = None
        self._connected_endpoint = ""
        self._last_request_s = 0.0
        self._last_send_s = 0.0
        self._pending_request = False

        self.host_edit = QtWidgets.QLineEdit(default_host)
        self.host_edit.setMinimumWidth(150)
        self.port_spin = QtWidgets.QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(int(default_port))
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_connection)

        self.value_spin = QtWidgets.QSpinBox()
        self.value_spin.setRange(-2_000_000_000, 2_000_000_000)
        self.value_spin.setSuffix(" samples")
        self.value_spin.setMinimumWidth(160)
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_current_value)
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.request_status)

        self.step_buttons: list[QtWidgets.QPushButton] = []
        for step in (-10, -1, 1, 10):
            btn = QtWidgets.QPushButton(f"{step:+d}")
            btn.setMinimumWidth(54)
            btn.clicked.connect(lambda _checked=False, delta=step: self.step_value(delta))
            self.step_buttons.append(btn)

        self.status_label = QtWidgets.QLabel("Disconnected")
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(24)

        self._build_layout()
        self._refresh_enabled_state()

    def _build_layout(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)

        conn = QtWidgets.QHBoxLayout()
        conn.addWidget(QtWidgets.QLabel("Host"))
        conn.addWidget(self.host_edit, 1)
        conn.addWidget(QtWidgets.QLabel("Port"))
        conn.addWidget(self.port_spin)
        conn.addWidget(self.connect_btn)
        layout.addLayout(conn)

        value_row = QtWidgets.QHBoxLayout()
        value_row.addWidget(QtWidgets.QLabel(self.command_label))
        value_row.addWidget(self.value_spin)
        value_row.addWidget(self.apply_btn)
        value_row.addWidget(self.refresh_btn)
        value_row.addStretch(1)
        layout.addLayout(value_row)

        steps = QtWidgets.QHBoxLayout()
        for btn in self.step_buttons:
            steps.addWidget(btn)
        steps.addStretch(1)
        layout.addLayout(steps)
        layout.addWidget(self.status_label)

    def toggle_connection(self) -> None:
        if self._socket is None:
            self.connect()
        else:
            self.disconnect()

    def connect(self) -> None:
        endpoint = make_tcp_endpoint(self.host_edit.text(), int(self.port_spin.value()))
        try:
            self._socket = make_control_dealer(
                endpoint,
                identity=f"uplink-timing-{self.command_label.lower()}-{os.getpid()}",
            )
        except Exception as exc:
            self._socket = None
            self._set_status(f"Connect failed: {exc}")
            self._refresh_enabled_state()
            return
        self._connected_endpoint = endpoint
        self._set_status(f"Connected to {endpoint}")
        self._refresh_enabled_state()
        self.request_status()

    def disconnect(self) -> None:
        if self._socket is not None:
            self._socket.close(0)
        self._socket = None
        self._connected_endpoint = ""
        self._pending_request = False
        self._set_status("Disconnected")
        self._refresh_enabled_state()

    def _refresh_enabled_state(self) -> None:
        connected = self._socket is not None
        self.connect_btn.setText("Disconnect" if connected else "Connect")
        self.host_edit.setEnabled(not connected)
        self.port_spin.setEnabled(not connected)
        self.value_spin.setEnabled(connected)
        self.apply_btn.setEnabled(connected)
        self.refresh_btn.setEnabled(connected)
        for btn in self.step_buttons:
            btn.setEnabled(connected)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)
        self.statusChanged.emit(text)

    def _ensure_connected(self) -> bool:
        if self._socket is not None:
            return True
        self.connect()
        return self._socket is not None

    def _rate_limited(self) -> bool:
        now = time.monotonic()
        if now - self._last_send_s < 0.055:
            self._set_status("Wait 50 ms before sending another timing update")
            return True
        self._last_send_s = now
        return False

    def send_value(self, value: int) -> None:
        if not self._ensure_connected() or self._rate_limited():
            return
        try:
            self._socket.send(build_control_command(self.command, int(value)), flags=zmq.NOBLOCK)
        except zmq.Again:
            self._set_status("Control socket is busy; command dropped")
            return
        except Exception as exc:
            self._set_status(f"Send failed: {exc}")
            return
        self.value_spin.setValue(int(value))
        self._set_status(f"Sent {self.command_label}={int(value)} samples")
        QtCore.QTimer.singleShot(80, self.request_status)

    def apply_current_value(self) -> None:
        self.send_value(int(self.value_spin.value()))

    def step_value(self, delta: int) -> None:
        self.send_value(int(self.value_spin.value()) + int(delta))

    def request_status(self) -> None:
        if not self._ensure_connected():
            return
        try:
            self._socket.send(build_status_request(self.command), flags=zmq.NOBLOCK)
            self._last_request_s = time.monotonic()
            self._pending_request = True
        except zmq.Again:
            return
        except Exception as exc:
            self._set_status(f"Request failed: {exc}")

    def poll_socket(self) -> None:
        if self._socket is None:
            return
        received = False
        while True:
            try:
                payload = self._socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception as exc:
                self._set_status(f"Receive failed: {exc}")
                break
            received = True
            self._handle_reply(payload)
        if (
            self._pending_request
            and not received
            and time.monotonic() - self._last_request_s > 0.8
        ):
            self._pending_request = False
            self._set_status(f"No status reply from {self._connected_endpoint}")

    def _handle_reply(self, payload: bytes) -> None:
        if len(payload) != REQUEST_PACKET_STRUCT.size:
            return
        header, command, value = REQUEST_PACKET_STRUCT.unpack(payload)
        if header != CTRL_HEADER or command != self.command:
            return
        self._pending_request = False
        self.value_spin.blockSignals(True)
        self.value_spin.setValue(int(value))
        self.value_spin.blockSignals(False)
        self._set_status(f"{self.command_label}={int(value)} samples @ {self._connected_endpoint}")

    def close(self) -> None:
        self.disconnect()


class UplinkTimingControlWindow(QtWidgets.QMainWindow):
    def __init__(self, bs_host: str, bs_port: int, ue_host: str, ue_port: int) -> None:
        super().__init__()
        self.setWindowTitle("OpenISAC Uplink Timing Control")
        self._build_ui(bs_host, bs_port, ue_host, ue_port)

        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(50)
        self._poll_timer.timeout.connect(self._poll_rows)
        self._poll_timer.start()

    def _build_ui(self, bs_host: str, bs_port: int, ue_host: str, ue_port: int) -> None:
        root = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.bs_row = TimingControlRow(
            "BS DUTI - DL/UL Timing Difference",
            BS_DL_UL_TIMING_DIFF_COMMAND,
            bs_host,
            bs_port,
        )
        self.ue_row = TimingControlRow(
            "UE TADV - Timing Advance",
            UE_TIMING_ADVANCE_COMMAND,
            ue_host,
            ue_port,
        )
        layout.addWidget(self.bs_row)
        layout.addWidget(self.ue_row)

        self.summary_label = QtWidgets.QLabel("Ready")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        self.bs_row.statusChanged.connect(lambda text: self._set_summary(f"BS: {text}"))
        self.ue_row.statusChanged.connect(lambda text: self._set_summary(f"UE: {text}"))

        self.setCentralWidget(root)
        self.resize(720, 360)

    def _set_summary(self, text: str) -> None:
        self.summary_label.setText(text)

    def _poll_rows(self) -> None:
        self.bs_row.poll_socket()
        self.ue_row.poll_socket()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 - Qt API name
        self.bs_row.close()
        self.ue_row.close()
        super().closeEvent(event)


def parse_args() -> tuple[Namespace, list[str]]:
    parser = ArgumentParser(description="GUI control panel for BS DUTI and UE TADV.")
    parser.add_argument("--bs-host", default="127.0.0.1", help="Initial BS control host/IP.")
    parser.add_argument("--bs-port", type=int, default=9999, help="Initial BS control port.")
    parser.add_argument("--ue-host", default="127.0.0.1", help="Initial UE control host/IP.")
    parser.add_argument("--ue-port", type=int, default=10001, help="Initial UE control port.")
    return parser.parse_known_args()


def main() -> int:
    args, qt_args = parse_args()
    app = QtWidgets.QApplication([sys.argv[0], *qt_args])
    win = UplinkTimingControlWindow(
        args.bs_host,
        int(args.bs_port),
        args.ue_host,
        int(args.ue_port),
    )
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
