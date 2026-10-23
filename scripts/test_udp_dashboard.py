#!/usr/bin/env python3
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import time
import unittest
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


if __name__ == "__main__":
    unittest.main()
