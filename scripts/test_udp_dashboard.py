#!/usr/bin/env python3
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import time
import unittest
import zmq
from PyQt6 import QtWidgets
from plot_udp_link import LinkDashboard, parse_code_rate


class DashboardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_fraction_input(self):
        self.assertEqual(parse_code_rate("1/2"), .5)
        self.assertAlmostEqual(parse_code_rate("2/3"), 2 / 3)
        for invalid in ("0", "2", "-1/2", "abc", "1/0"):
            with self.assertRaises((ValueError, ZeroDivisionError)):
                parse_code_rate(invalid)

    def test_metrics_and_stale_status(self):
        window = LinkDashboard("inproc://unused-tx", "inproc://unused-rx", "inproc://unused-control")
        try:
            now = time.monotonic()
            window.accept("tx", dict(version=1, role="tx", session="test", running=True,
                                     tx_mbps=50, elapsed_s=65), now)
            window.accept("rx", dict(version=1, role="rx", session="test", running=True,
                                     rx_mbps=40, loss_percent=20, bit_errors=69, ber=1e-6,
                                     unique_packets=10000, received_bytes=13160000, lost_packets=2500), now)
            window.tick()
            self.assertIn("40.00", window.rx_card.value.text())
            self.assertIn("80.00", window.rx_card.detail.text())
            self.assertEqual(window.loss_card.value.text(), "20.000%")
            self.assertEqual(window.ber_card.value.text(), "1.00e-06")
            window.ldpc_input.setText("2/3")
            window.change_code_rate()
            window.tick()
            self.assertIn("60.00", window.rx_card.detail.text())
            window.ldpc_input.setText("0")
            window.change_code_rate()
            self.assertAlmostEqual(window.code_rate, 2 / 3)
            window.seen["rx"] = now - 10
            window.tick()
            self.assertEqual(window.rx_card.value.text(), "— Mbps")
            window.seen["rx"] = time.monotonic()
            window.data["rx"]["session"] = "other"
            window.tick()
            self.assertEqual(window.rx_card.value.text(), "— Mbps")
        finally:
            window.close()

    def test_first_packet_and_independent_clear_buttons(self):
        window = LinkDashboard("inproc://tx", "inproc://rx", "inproc://tx-control",
                               rx_control_endpoint="inproc://rx-control")
        window.timer.stop()
        control = window.context.socket(zmq.REP)
        control.bind("inproc://rx-control")
        try:
            now = time.monotonic()
            window.accept("tx", dict(version=1, role="tx", session="test", running=True), now)
            rx = dict(version=1, role="rx", session="test", running=True, loss_started=False,
                      unique_packets=0, compared_bits=0, bit_errors=0, loss_percent=0, lost_packets=0)
            window.accept("rx", rx.copy(), now); window.tick()
            self.assertEqual(window.loss_card.detail.text(), "等待首个 RX 包")
            self.assertFalse(window.loss_card.clear_button.isEnabled())
            self.assertFalse(window.ber_card.clear_button.isEnabled())

            rx.update(loss_started=True, unique_packets=30, received_bytes=39480,
                      compared_bits=315840, bit_errors=12, ber=12/315840, lost_packets=10,
                      loss_percent=25, rx_mbps=50)
            window.accept("rx", rx.copy(), now); window.tick()
            for button, command, message in [
                (window.loss_card.clear_button, "RESET_LOSS", "丢包统计已清除"),
                (window.ber_card.clear_button, "RESET_ERRORS", "误码统计已清除"),
            ]:
                self.assertTrue(button.isEnabled())
                button.click()
                self.assertFalse(window.apply_rate.isEnabled())
                self.assertTrue(control.poll(500))
                self.assertEqual(control.recv_string(), command)
                control.send_json({"ok": True})
                if command == "RESET_LOSS":
                    rx.update(lost_packets=0, loss_percent=0)
                else:
                    rx.update(bit_errors=0, compared_bits=0, ber=0)
                window.accept("rx", rx.copy(), now); window.tick()
                self.assertEqual(window.notice.text(), message)
                self.assertEqual(window.packet_card.value.text(), "30")
                self.assertIn("50.00", window.rx_card.value.text())
                if command == "RESET_LOSS":
                    self.assertEqual(window.loss_card.value.text(), "0.000%")
                    self.assertEqual(window.error_card.value.text(), "12")
                else:
                    self.assertEqual(window.ber_card.value.text(), "0.00e+00")
                    self.assertEqual(window.error_card.value.text(), "0")

            window.loss_card.clear_button.click()
            self.assertTrue(control.poll(500)); control.recv()
            window.request_time -= 3; window.tick()
            self.assertEqual(window.notice.text(), "未能清除丢包统计")
            self.assertTrue(window.apply_rate.isEnabled())
        finally:
            control.close(0)
            window.close()


if __name__ == "__main__":
    unittest.main()
