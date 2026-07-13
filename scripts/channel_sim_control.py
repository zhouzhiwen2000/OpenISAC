#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

try:
    from PyQt6 import QtCore, QtWidgets
except ImportError as exc:  # pragma: no cover - startup dependency check
    raise SystemExit("PyQt6 is required. Install it with `pip install PyQt6`.") from exc

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sensing_runtime_protocol import (  # noqa: E402
    CHANNEL_SIM_SNR_COMMAND,
    CHANNEL_SIM_SNR_DISABLED,
    CTRL_HEADER,
    REQUEST_PACKET_STRUCT,
    build_channel_sim_snr_command,
    build_channel_sim_snr_request,
    make_control_dealer,
    make_tcp_endpoint,
)

try:
    import zmq
except ImportError as exc:  # pragma: no cover - startup dependency check
    raise SystemExit("pyzmq is required. Install it with `pip install pyzmq`.") from exc


def _slider_value_from_snr(snr_db: float) -> int:
    return int(round(float(snr_db) * 10.0))


def _snr_from_slider_value(value: int) -> float:
    return float(value) / 10.0


class ChannelSimControlWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OpenISAC Channel Simulator Control")
        self._socket = None
        self._connected_endpoint = ""
        self._last_request_s = 0.0
        self._syncing_controls = False

        self._build_ui()

        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(50)
        self._poll_timer.timeout.connect(self._poll_socket)
        self._poll_timer.start()

        self._apply_timer = QtCore.QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.setInterval(120)
        self._apply_timer.timeout.connect(self.apply_snr)

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        conn = QtWidgets.QHBoxLayout()
        self.host_edit = QtWidgets.QLineEdit("127.0.0.1")
        self.host_edit.setMinimumWidth(150)
        self.port_spin = QtWidgets.QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(10002)
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_connection)
        conn.addWidget(QtWidgets.QLabel("Host"))
        conn.addWidget(self.host_edit, 1)
        conn.addWidget(QtWidgets.QLabel("Port"))
        conn.addWidget(self.port_spin)
        conn.addWidget(self.connect_btn)
        layout.addLayout(conn)

        snr_row = QtWidgets.QHBoxLayout()
        self.enable_check = QtWidgets.QCheckBox("Target SNR")
        self.enable_check.setChecked(True)
        self.enable_check.stateChanged.connect(self._schedule_apply)
        self.snr_spin = QtWidgets.QDoubleSpinBox()
        self.snr_spin.setRange(-20.0, 80.0)
        self.snr_spin.setDecimals(1)
        self.snr_spin.setSingleStep(0.5)
        self.snr_spin.setSuffix(" dB")
        self.snr_spin.setValue(40.0)
        self.snr_spin.valueChanged.connect(self._spin_changed)
        snr_row.addWidget(self.enable_check)
        snr_row.addStretch(1)
        snr_row.addWidget(self.snr_spin)
        layout.addLayout(snr_row)

        self.snr_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.snr_slider.setRange(_slider_value_from_snr(-20.0), _slider_value_from_snr(80.0))
        self.snr_slider.setSingleStep(1)
        self.snr_slider.setPageStep(50)
        self.snr_slider.setValue(_slider_value_from_snr(40.0))
        self.snr_slider.valueChanged.connect(self._slider_changed)
        layout.addWidget(self.snr_slider)

        buttons = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_snr)
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.request_status)
        buttons.addWidget(self.apply_btn)
        buttons.addWidget(self.refresh_btn)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.status_label = QtWidgets.QLabel("Disconnected")
        self.status_label.setMinimumHeight(28)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.setCentralWidget(root)
        self.resize(520, 210)
        self._refresh_enabled_state()

    def _refresh_enabled_state(self) -> None:
        connected = self._socket is not None
        self.connect_btn.setText("Disconnect" if connected else "Connect")
        self.host_edit.setEnabled(not connected)
        self.port_spin.setEnabled(not connected)
        self.apply_btn.setEnabled(connected)
        self.refresh_btn.setEnabled(connected)

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
                identity=f"channel-sim-control-{os.getpid()}",
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
        self._set_status("Disconnected")
        self._refresh_enabled_state()

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _spin_changed(self, value: float) -> None:
        if self._syncing_controls:
            return
        self._syncing_controls = True
        self.snr_slider.setValue(_slider_value_from_snr(value))
        self._syncing_controls = False
        self._schedule_apply()

    def _slider_changed(self, value: int) -> None:
        if self._syncing_controls:
            return
        self._syncing_controls = True
        self.snr_spin.setValue(_snr_from_slider_value(value))
        self._syncing_controls = False
        self._schedule_apply()

    def _schedule_apply(self) -> None:
        if self._syncing_controls:
            return
        if self._socket is not None:
            self._apply_timer.start()

    def apply_snr(self) -> None:
        if self._socket is None:
            return
        snr_db = float(self.snr_spin.value()) if self.enable_check.isChecked() else None
        try:
            self._socket.send(build_channel_sim_snr_command(snr_db), flags=zmq.NOBLOCK)
        except zmq.Again:
            self._set_status("Control socket is busy; command dropped")
            return
        except Exception as exc:
            self._set_status(f"Send failed: {exc}")
            return
        if snr_db is None:
            self._set_status("Sent: target SNR scaling off")
        else:
            self._set_status(f"Sent: target SNR {snr_db:.1f} dB")
        self.request_status()

    def request_status(self) -> None:
        if self._socket is None:
            return
        try:
            self._socket.send(build_channel_sim_snr_request(), flags=zmq.NOBLOCK)
            self._last_request_s = time.monotonic()
        except zmq.Again:
            return
        except Exception as exc:
            self._set_status(f"Request failed: {exc}")

    def _poll_socket(self) -> None:
        if self._socket is None:
            return
        while True:
            try:
                payload = self._socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception as exc:
                self._set_status(f"Receive failed: {exc}")
                break
            self._handle_reply(payload)
        if time.monotonic() - self._last_request_s > 1.0:
            self.request_status()

    def _handle_reply(self, payload: bytes) -> None:
        if len(payload) != REQUEST_PACKET_STRUCT.size:
            return
        header, command, value = REQUEST_PACKET_STRUCT.unpack(payload)
        if header != CTRL_HEADER or command != CHANNEL_SIM_SNR_COMMAND:
            return
        self._syncing_controls = True
        if int(value) == CHANNEL_SIM_SNR_DISABLED:
            self.enable_check.setChecked(False)
            self._set_status(f"{self._connected_endpoint}: target SNR scaling off")
        else:
            snr_db = float(value) / 100.0
            self.enable_check.setChecked(True)
            self.snr_spin.setValue(snr_db)
            self.snr_slider.setValue(_slider_value_from_snr(snr_db))
            self._set_status(f"{self._connected_endpoint}: target SNR {snr_db:.2f} dB")
        self._syncing_controls = False

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        self.disconnect()
        super().closeEvent(event)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    win = ChannelSimControlWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
