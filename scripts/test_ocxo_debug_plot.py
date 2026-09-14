#!/usr/bin/env python3
from __future__ import annotations

import os
import struct
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets

from plot_ocxo_debug import (
    CHANNELS,
    CHANNEL_COUNT,
    LEGACY_MAGIC,
    LEGACY_VERSION,
    MAGIC,
    PACKET_STRUCT,
    PACKET_V2_STRUCT,
    PPS_DIAGNOSTIC_NAMES,
    VERSION,
    DebugPlotWindow,
    OcxoDebugWindow,
    ViewerArgs,
    _csv_header,
    _csv_row,
    _daily_record_path,
    _unique_path,
    parse_packet,
)


class OcxoDebugProtocolTest(unittest.TestCase):
    def test_daily_recording_name_increments_two_digit_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "ocxo_debug_20260720_01.csv"
            self.assertEqual(_unique_path(first), first)
            first.touch()
            second = Path(tmp) / "ocxo_debug_20260720_02.csv"
            self.assertEqual(_unique_path(first), second)
            second.touch()
            self.assertEqual(
                _unique_path(first),
                Path(tmp) / "ocxo_debug_20260720_03.csv",
            )
            self.assertEqual(
                _daily_record_path(Path(tmp), datetime(2026, 7, 21)),
                Path(tmp) / "ocxo_debug_20260721_01.csv",
            )

    def test_parse_packet(self) -> None:
        values = tuple(float(index) + 0.5 for index in range(CHANNEL_COUNT))
        diagnostics = (12.25, 34.5, 9_876_543_210, 123_456,
                       8_765_432_100, 200.0e6, 50.0e6)
        data = PACKET_STRUCT.pack(
            MAGIC, VERSION, CHANNEL_COUNT, 17, 123456789,
            *values, *diagnostics,
        )
        frame = parse_packet(data)
        self.assertEqual(frame.sequence, 17)
        self.assertEqual(frame.timestamp_ns, 123456789)
        self.assertEqual(frame.channels.tolist(), list(values))
        self.assertEqual(frame.r_bs_samples, 12.25)
        self.assertEqual(frame.r_ue_samples, 34.5)
        self.assertEqual(frame.bs_last_pps_ticks, 9_876_543_210)
        self.assertEqual(frame.bs_t0_to_pps_ticks, 123_456)
        self.assertEqual(frame.ue_last_pps_ns, 8_765_432_100)
        self.assertEqual(frame.bs_tick_rate, 200.0e6)
        self.assertEqual(frame.sample_rate, 50.0e6)
        self.assertEqual(
            PPS_DIAGNOSTIC_NAMES,
            (
                "r_bs_samples",
                "r_ue_samples",
                "bs_last_pps_ticks",
                "bs_t0_to_pps_ticks",
                "ue_last_pps_ns",
                "bs_tick_rate",
                "sample_rate",
            ),
        )
        self.assertEqual(_csv_header()[-7:], list(PPS_DIAGNOSTIC_NAMES))
        self.assertEqual(
            _csv_row(frame, 222)[-7:],
            [
                "12.25",
                "34.5",
                "9876543210",
                "123456",
                "8765432100",
                "200000000",
                "50000000",
            ],
        )

    def test_parse_legacy_v2_packet(self) -> None:
        values = tuple(float(index) for index in range(CHANNEL_COUNT))
        data = PACKET_V2_STRUCT.pack(
            LEGACY_MAGIC, LEGACY_VERSION, CHANNEL_COUNT, 3, 4, *values
        )
        frame = parse_packet(data)
        self.assertEqual(frame.channels.tolist(), list(values))
        self.assertEqual(frame.bs_last_pps_ticks, -1)
        self.assertTrue(np.isnan(frame.r_bs_samples))

    def test_rejects_bad_header(self) -> None:
        values = (0.0,) * CHANNEL_COUNT
        diagnostics = (0.0, 0.0, 0, 0, 0, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "magic"):
            parse_packet(PACKET_STRUCT.pack(
                b"BADMAGIC", VERSION, CHANNEL_COUNT, 0, 0,
                *values, *diagnostics,
            ))
        with self.assertRaisesRegex(ValueError, "version"):
            parse_packet(PACKET_STRUCT.pack(
                MAGIC, VERSION + 1, CHANNEL_COUNT, 0, 0,
                *values, *diagnostics,
            ))
        with self.assertRaisesRegex(ValueError, "channel count"):
            parse_packet(PACKET_STRUCT.pack(
                MAGIC, VERSION, CHANNEL_COUNT - 1, 0, 0,
                *values, *diagnostics,
            ))
        with self.assertRaisesRegex(ValueError, "packet size"):
            parse_packet(struct.pack("<8s", MAGIC))


class OcxoDebugLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_reopened_plot_group_recovers_equal_height(self) -> None:
        args = ViewerArgs(
            host="127.0.0.1",
            port=12347,
            interval_ms=50,
            history_seconds=60.0,
            channels=(2, 3, 6, 7, 8, 11, 12, 13, 14, 15),
            record_path=None,
        )
        with (
            mock.patch.object(DebugPlotWindow, "connect_endpoint", lambda _self: None),
            mock.patch("plot_ocxo_debug.load_settings", return_value={}),
            mock.patch("plot_ocxo_debug.save_settings"),
        ):
            window = OcxoDebugWindow(args)
            window.show()
            self.app.processEvents()

            ppm_channels = [
                spec.index for spec in CHANNELS
                if spec.group == "Frequency / command"
            ]
            for index in ppm_channels:
                window.channel_checks[index].setChecked(False)
            self.app.processEvents()
            window.channel_checks[ppm_channels[0]].setChecked(True)
            self.app.processEvents()

            visible_heights = [
                plot.height() for plot in window.plots.values()
                if plot.isVisible()
            ]
            self.assertLessEqual(
                max(visible_heights) - min(visible_heights),
                2,
                f"reopened plot group heights are unbalanced: {visible_heights}",
            )
            window.close()


if __name__ == "__main__":
    unittest.main()
