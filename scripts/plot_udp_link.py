#!/usr/bin/env python3
"""Live UDP link dashboard for the C++ UdpLinkTest sender and receiver."""
from __future__ import annotations

import argparse
from collections import deque
from fractions import Fraction
import json
import math
import sys
import time

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets
import zmq

BACKGROUND = "#080f1c"
PANEL = "#111d30"
TEXT = "#e8effa"
MUTED = "#8eabc9"
CYAN = "#49cfff"
GREEN = "#56e1b0"
AMBER = "#ffcf75"


def parse_code_rate(text: str) -> float:
    value = float(Fraction(text.strip()))
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("LDPC 码率应大于 0 且不超过 1")
    return value


def number(data: dict, key: str) -> float:
    value = float(data.get(key, 0))
    return value if math.isfinite(value) and value >= 0 else 0.0


class MetricCard(QtWidgets.QFrame):
    def __init__(self, title: str, color: str, large: bool = False):
        super().__init__()
        self.setObjectName("card")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.setSpacing(8)
        heading = QtWidgets.QLabel(title)
        heading.setStyleSheet(f"color: {MUTED}; font-size: 16px;")
        self.value = QtWidgets.QLabel("—")
        self.value.setStyleSheet(
            f"color: {color}; font: 600 {48 if large else 31}px 'DejaVu Sans Mono';"
        )
        self.detail = QtWidgets.QLabel(" ")
        self.detail.setStyleSheet(f"color: {MUTED}; font-size: {17 if large else 13}px;")
        layout.addWidget(heading)
        layout.addWidget(self.value)
        layout.addWidget(self.detail)


class LinkDashboard(QtWidgets.QMainWindow):
    def __init__(self, tx_endpoint: str, rx_endpoint: str, control_endpoint: str,
                 code_rate: str = "1/2", target_mbps: float = 50):
        super().__init__()
        self.setWindowTitle("OpenISAC · 链路性能")
        self.resize(1440, 940)
        self.setMinimumSize(1100, 790)
        self.code_rate = parse_code_rate(code_rate)
        self.data: dict[str, dict] = {"tx": {}, "rx": {}}
        self.seen = {"tx": 0.0, "rx": 0.0}
        self.started = time.monotonic()
        self.history = deque(maxlen=600)
        self.context = zmq.Context()
        self.subscribers = {}
        for role, endpoint in (("tx", tx_endpoint), ("rx", rx_endpoint)):
            sock = self.context.socket(zmq.SUB)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.RCVHWM, 8)
            sock.setsockopt(zmq.SUBSCRIBE, b"")
            sock.connect(endpoint)
            self.subscribers[role] = sock
        self.control_endpoint = control_endpoint
        self.request = None
        self.request_time = 0.0

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(32, 24, 32, 24)
        outer.setSpacing(18)

        header = QtWidgets.QHBoxLayout()
        identity = QtWidgets.QVBoxLayout()
        brand = QtWidgets.QLabel("OPENISAC")
        brand.setStyleSheet(f"color: {CYAN}; font-size: 14px; font-weight: 700; letter-spacing: 4px;")
        title = QtWidgets.QLabel("链路性能")
        title.setStyleSheet("font-size: 30px; font-weight: 700;")
        identity.addWidget(brand)
        identity.addWidget(title)
        header.addLayout(identity)
        header.addStretch()
        self.badges = {}
        for role, label in (("tx", "BS"), ("rx", "UE")):
            badge = QtWidgets.QLabel(f"●  {label} 等待连接")
            badge.setStyleSheet(f"color: {MUTED}; font-size: 14px; padding: 8px;")
            self.badges[role] = badge
            header.addWidget(badge)
        outer.addLayout(header)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(12)
        controls.addWidget(QtWidgets.QLabel("目标码率"))
        self.target = QtWidgets.QDoubleSpinBox()
        self.target.setRange(0, 10000)
        self.target.setDecimals(1)
        self.target.setValue(target_mbps)
        self.target.setSuffix(" Mbps")
        self.target.setFixedWidth(170)
        controls.addWidget(self.target)
        self.apply_rate = QtWidgets.QPushButton("应用")
        self.apply_rate.setFixedWidth(88)
        self.apply_rate.clicked.connect(self.change_rate)
        controls.addWidget(self.apply_rate)
        controls.addSpacing(24)
        controls.addWidget(QtWidgets.QLabel("LDPC 码率"))
        self.ldpc_input = QtWidgets.QLineEdit(code_rate)
        self.ldpc_input.setFixedWidth(100)
        self.ldpc_input.editingFinished.connect(self.change_code_rate)
        controls.addWidget(self.ldpc_input)
        controls.addStretch()
        self.notice = QtWidgets.QLabel("")
        self.notice.setStyleSheet(f"color: {AMBER}; font-size: 13px;")
        controls.addWidget(self.notice)
        outer.addLayout(controls)

        rates = QtWidgets.QHBoxLayout()
        rates.setSpacing(18)
        self.tx_card = MetricCard("发送速率", CYAN, True)
        self.rx_card = MetricCard("接收速率", GREEN, True)
        rates.addWidget(self.tx_card)
        rates.addWidget(self.rx_card)
        outer.addLayout(rates)

        chart_frame = QtWidgets.QFrame()
        chart_frame.setObjectName("card")
        chart_layout = QtWidgets.QVBoxLayout(chart_frame)
        chart_layout.setContentsMargins(20, 16, 20, 12)
        chart_header = QtWidgets.QHBoxLayout()
        chart_title = QtWidgets.QLabel("实时流量")
        chart_title.setStyleSheet("font-size: 18px; font-weight: 600;")
        chart_header.addWidget(chart_title)
        chart_header.addStretch()
        self.elapsed = QtWidgets.QLabel("00:00:00")
        self.elapsed.setStyleSheet(f"color: {MUTED}; font: 14px 'DejaVu Sans Mono';")
        chart_header.addWidget(self.elapsed)
        chart_layout.addLayout(chart_header)
        self.plot = pg.PlotWidget(background=PANEL)
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.showGrid(x=True, y=True, alpha=.12)
        self.plot.setLabel("left", "Mbps", color=MUTED)
        self.plot.setLabel("bottom", "时间 / s", color=MUTED)
        for name in ("left", "bottom"):
            self.plot.getAxis(name).setTextPen(MUTED)
            self.plot.getAxis(name).setPen("#2c405c")
            self.plot.getAxis(name).setTickFont(QtGui.QFont("DejaVu Sans", 10))
            self.plot.getAxis(name).label.setDefaultTextColor(QtGui.QColor(MUTED))
        legend = self.plot.addLegend(offset=(14, 10), labelTextColor=TEXT, labelTextSize="13pt")
        legend.setBrush(pg.mkBrush(PANEL))
        legend.setPen(pg.mkPen(None))
        self.tx_curve = self.plot.plot(pen=pg.mkPen(CYAN, width=2.5), name="BS 发送")
        self.rx_curve = self.plot.plot(pen=pg.mkPen(GREEN, width=2.5), name="UE 接收",
                                       fillLevel=0, brush=pg.mkBrush(86, 225, 176, 18))
        self.plot.setMinimumHeight(230)
        chart_layout.addWidget(self.plot)
        outer.addWidget(chart_frame, 1)

        quality = QtWidgets.QHBoxLayout()
        quality.setSpacing(18)
        self.loss_card = MetricCard("丢包率", AMBER)
        self.ber_card = MetricCard("误比特率 BER", AMBER)
        self.packet_card = MetricCard("累计接收", TEXT)
        self.error_card = MetricCard("累计误比特", TEXT)
        for card in (self.loss_card, self.ber_card, self.packet_card, self.error_card):
            quality.addWidget(card)
        outer.addLayout(quality)
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background: {BACKGROUND}; color: {TEXT}; font-family: 'Noto Sans CJK SC', 'DejaVu Sans'; font-size: 15px; }}
            QFrame#card {{ background: {PANEL}; border: 1px solid #24334b; border-radius: 14px; }}
            QFrame#card QLabel {{ background: transparent; border: none; }}
            QLineEdit, QDoubleSpinBox {{ background: {PANEL}; border: 1px solid #354c69; border-radius: 7px; padding: 8px; }}
            QPushButton {{ background: #1c789b; border: none; border-radius: 7px; padding: 10px 16px; font-weight: 600; }}
            QPushButton:hover {{ background: #2498bd; }}
            QPushButton:disabled {{ background: #27384d; color: {MUTED}; }}
        """)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(200)

    def change_code_rate(self):
        try:
            self.code_rate = parse_code_rate(self.ldpc_input.text())
            self.ldpc_input.setStyleSheet("")
            self.notice.setText("")
        except (ValueError, ZeroDivisionError):
            self.ldpc_input.setStyleSheet("border: 1px solid #ff7887;")
            self.notice.setText("请输入有效的 LDPC 码率")

    def change_rate(self):
        if self.request is not None:
            return
        self.request = self.context.socket(zmq.REQ)
        self.request.setsockopt(zmq.LINGER, 0)
        self.request.connect(self.control_endpoint)
        self.request.send_string(f"RATE {self.target.value():.6f}", flags=zmq.DONTWAIT)
        self.request_time = time.monotonic()
        self.apply_rate.setEnabled(False)
        self.notice.setText("正在应用…")

    def accept(self, role: str, data: dict, now: float):
        if isinstance(data, dict) and data.get("version") == 1 and data.get("role") == role:
            self.data[role] = data
            self.seen[role] = now

    def tick(self):
        now = time.monotonic()
        for role, sock in self.subscribers.items():
            for _ in range(16):
                try:
                    data = sock.recv_json(flags=zmq.DONTWAIT)
                    self.accept(role, data, now)
                except zmq.Again:
                    break
                except (ValueError, TypeError):
                    continue
        if self.request is not None:
            reply = None
            try:
                reply = self.request.recv_json(flags=zmq.DONTWAIT)
            except zmq.Again:
                pass
            except ValueError:
                reply = {"ok": False}
            if reply is not None or now - self.request_time > 2:
                self.notice.setText("码率已更新" if reply and reply.get("ok") else "未能更新码率")
                self.request.close()
                self.request = None
                self.apply_rate.setEnabled(True)
        fresh = {}
        for role, label in (("tx", "BS"), ("rx", "UE")):
            fresh[role] = now - self.seen[role] < 2 and self.data[role].get("running", False)
            self.badges[role].setText(f"●  {label} {'在线' if fresh[role] else '等待连接'}")
            self.badges[role].setStyleSheet(f"color: {GREEN if fresh[role] else MUTED}; font-size: 14px; padding: 8px;")
        tx, rx = self.data["tx"], self.data["rx"]
        session_match = not tx or not rx or tx.get("session") == rx.get("session")
        if not session_match:
            self.notice.setText("等待同一轮测试数据")
            fresh["rx"] = False
        tx_rate = number(tx, "tx_mbps") if fresh["tx"] else float("nan")
        rx_rate = number(rx, "rx_mbps") if fresh["rx"] else float("nan")
        for card, rate in ((self.tx_card, tx_rate), (self.rx_card, rx_rate)):
            card.value.setText(f"{rate:,.2f} Mbps" if math.isfinite(rate) else "— Mbps")
            card.detail.setText(f"LDPC 编码后速率  {rate / self.code_rate:,.2f} Mbps" if math.isfinite(rate) else "LDPC 编码后速率  — Mbps")
        self.history.append((now - self.started, tx_rate, rx_rate))
        values = np.asarray(self.history)
        self.tx_curve.setData(values[:, 0], values[:, 1], connect="finite")
        self.rx_curve.setData(values[:, 0], values[:, 2], connect="finite")
        self.plot.setXRange(max(0, values[-1, 0] - 120), max(10, values[-1, 0]), padding=0)
        finite_rates = values[:, 1:][np.isfinite(values[:, 1:])]
        self.plot.setYRange(0, max(10, float(finite_rates.max()) * 1.18 if finite_rates.size else 10), padding=0)
        elapsed = int(number(tx, "elapsed_s"))
        self.elapsed.setText(f"{elapsed // 3600:02d}:{elapsed // 60 % 60:02d}:{elapsed % 60:02d}")
        valid_stats = bool(rx) and session_match
        self.loss_card.value.setText(f"{number(rx, 'loss_percent'):.3f}%" if valid_stats else "—")
        self.loss_card.detail.setText(f"丢失 {int(number(rx, 'lost_packets')):,} 包" if valid_stats else " ")
        self.ber_card.value.setText(f"{number(rx, 'ber'):.2e}" if valid_stats else "—")
        self.ber_card.detail.setText("")
        self.packet_card.value.setText(f"{int(number(rx, 'unique_packets') + number(rx, 'invalid_packets')):,}" if valid_stats else "—")
        self.packet_card.detail.setText(f"{number(rx, 'received_bytes') / 1e9:,.3f} GB" if valid_stats else " ")
        self.error_card.value.setText(f"{int(number(rx, 'bit_errors')):,}" if valid_stats else "—")
        self.error_card.detail.setText("")

    def closeEvent(self, event):
        self.timer.stop()
        for sock in self.subscribers.values():
            sock.close()
        if self.request is not None:
            self.request.close()
        self.context.term()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tx", default="tcp://127.0.0.1:18100")
    parser.add_argument("--rx", default="tcp://127.0.0.1:18101")
    parser.add_argument("--control", default="tcp://127.0.0.1:18102")
    parser.add_argument("--code-rate", default="1/2")
    parser.add_argument("--rate-mbps", type=float, default=50)
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setFont(QtGui.QFont("Noto Sans CJK SC", 11))
    pg.setConfigOptions(antialias=True)
    window = LinkDashboard(args.tx, args.rx, args.control, args.code_rate, args.rate_mbps)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
