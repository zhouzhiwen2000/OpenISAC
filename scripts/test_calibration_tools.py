"""Offline calibration workflow regressions, including real ZeroMQ transport."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

import numpy as np
import yaml
import zmq

import calibrate_hsys as hsys
from calibrate_system_delay import analyze
from sensing_runtime_protocol import (
    FLAG_BISTATIC, FLAG_COMPACT_MASK, PARAMS_PACKET_STRUCT,
    REQUEST_PACKET_STRUCT, ViewerRuntimeParams,
)


def response_bytes(role="monostatic", channel=0, symbols=32, values=None):
    values = np.ones(8, dtype=np.complex64) if values is None else np.asarray(values, dtype=np.complex64)
    return hsys.HSYS_HEADER.pack(b"OISACH1\0", role.encode(), 1,
                                 hsys.ROLE_IDS[role], channel, len(values),
                                 50e6, 2.4e9, 40e6, symbols) + values.tobytes()


def params_packet(role="monostatic", version=9):
    return PARAMS_PACKET_STRUCT.pack(
        b"CTRL", b"PARM", version, FLAG_BISTATIC if role == "bistatic" else 0,
        0, 100, 8, 100, 8, 100, 8, 100, 0, 0, 1, 1, 7500, 2, 2,
        24000000, 500000, 0, 8, 2, 1, 1)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.file = self.root / "response.bin"

    def test_header_quality_and_invalid_files(self):
        values = np.ones(8, dtype=np.complex64)
        values[0] = 0
        self.file.write_bytes(response_bytes(values=values))
        result = hsys.inspect_response(self.file)
        self.assertEqual(result["weak_bins"], 1)
        self.assertEqual(result["captured_symbols"], 32)
        self.assertEqual(len(result["sha256"]), 64)
        invalid = [b"", response_bytes()[:-1], response_bytes() + b"x",
                   response_bytes(symbols=0), response_bytes(values=np.zeros(8)),
                   response_bytes(values=[np.nan] * 8), response_bytes(values=[np.inf] * 8),
                   response_bytes(values=[3e38] * 8),
                   b"WRONG!!!" + response_bytes()[8:]]
        for raw in invalid:
            with self.subTest(raw=raw[:16]):
                self.file.write_bytes(raw)
                with self.assertRaises(ValueError):
                    hsys.inspect_response(self.file)

    def test_old_file_cannot_satisfy_new_request(self):
        self.file.write_bytes(response_bytes())
        with self.assertRaisesRegex(TimeoutError, "no newly saved file"):
            hsys.wait_for_response(self.file, hsys.file_stamp(self.file), {},
                                   "monostatic", 0, 32, .05)

    def test_wrong_channel_count_and_rf_metadata_rejected(self):
        profile = dict(fft_size=8, sample_rate_hz=50e6, center_freq_hz=2.4e9, bandwidth_hz=40e6)
        for channel, symbols, freq in [(1, 32, 2.4e9), (0, 33, 2.4e9), (0, 32, 5.8e9)]:
            self.file.write_bytes(response_bytes(channel=channel, symbols=symbols))
            with self.assertRaises(TimeoutError):
                hsys.wait_for_response(self.file, None, dict(profile, center_freq_hz=freq),
                                       "monostatic", 0, 32, .05)

    def test_params_preflight(self):
        profile = dict(fft_size=8, sample_rate_hz=50e6, center_freq_hz=2.4e9)
        params = ViewerRuntimeParams(ofdm_fft_size=8, sample_rate_hz=50e6, center_freq_hz=2.4e9)
        hsys.check_params(params, profile, "monostatic")
        for bad in [replace(params, flags=FLAG_BISTATIC), replace(params, ofdm_fft_size=16),
                    replace(params, sample_rate_hz=1e6), replace(params, flags=FLAG_COMPACT_MASK)]:
            with self.assertRaises(ValueError):
                hsys.check_params(bad, profile, "monostatic")

    def test_unreachable_backend_times_out(self):
        reserve = zmq.Context.instance().socket(zmq.ROUTER)
        port = reserve.bind_to_random_port("tcp://127.0.0.1")
        reserve.close(linger=0)
        sock = hsys.make_control_dealer(f"tcp://127.0.0.1:{port}")
        sock.setsockopt(zmq.IMMEDIATE, 1)
        sock.setsockopt(zmq.SNDTIMEO, 50)
        try:
            with self.assertRaises((TimeoutError, zmq.Again)):
                hsys.request_params(sock, .05)
        finally:
            sock.close(linger=0)

    def test_disabled_channel_rejected_before_connection(self):
        path = self.root / "BS.yaml"
        path.write_text(yaml.safe_dump(dict(sensing=dict(rx_channels=[dict(enable_system_delay_estimation=True)]))))
        for channel in [-1, 0, 1]:
            with self.assertRaises(ValueError):
                hsys.load_profile(self.root, "monostatic", channel)

    def test_current_and_legacy_ofdm_sections(self):
        for role in ["monostatic", "bistatic"]:
            for section in ["ofdm_frame", "ofdm"]:
                with self.subTest(role=role, section=section):
                    config = dict(rf_sampling=dict(sample_rate=100e6, bandwidth=160e6),
                                  downlink=dict(center_freq=3.1e9),
                                  sensing=dict(rx_channels=[{}], bi_enabled=True,
                                               range_fft_size=10240, doppler_fft_size=1000),
                                  network_output=dict(control_port=9999))
                    config[section] = dict(fft_size=1024)
                    if section == "ofdm_frame":
                        config["ofdm"] = dict(fft_size=8)
                    name = "BS.yaml" if role == "monostatic" else "UE.yaml"
                    (self.root / name).write_text(yaml.safe_dump(config))
                    profile = hsys.load_profile(self.root, role, 0)
                    self.assertEqual(profile["fft_size"], 1024)
                    self.assertEqual(profile["sample_rate_hz"], 100e6)
                    self.assertEqual(profile["center_freq_hz"], 3.1e9)
                    self.assertEqual(profile["bandwidth_hz"], 160e6)

    def run_capture(self, role="monostatic", version=9, save=True):
        channel = 1 if role == "monostatic" else 0
        config = dict(rf_sampling=dict(sample_rate=50e6, bandwidth=40e6),
                      ofdm_frame=dict(fft_size=8), downlink=dict(center_freq=2.4e9),
                      sensing=dict(rx_channel_count=2, rx_channels=[{}, {}], bi_enabled=True),
                      network_output=dict(control_port=9999))
        (self.root / ("BS.yaml" if role == "monostatic" else "UE.yaml")).write_text(yaml.safe_dump(config))
        output = self.root / f"sensing_system_response_{role}_ch{channel}.bin"
        old = response_bytes(role, channel, symbols=16)
        output.write_bytes(old)
        ready, stop = threading.Event(), threading.Event()
        state = dict(commands=[], errors=[])

        def server():
            sock = zmq.Context.instance().socket(zmq.ROUTER)
            try:
                state["port"] = sock.bind_to_random_port("tcp://127.0.0.1")
                ready.set()
                while not stop.is_set():
                    if not sock.poll(20):
                        continue
                    peer, raw = sock.recv_multipart()
                    header, command, value = REQUEST_PACKET_STRUCT.unpack(raw)
                    state["commands"].append((header, command, value))
                    if command == b"PARM":
                        sock.send_multipart([peer, params_packet(role, version)])
                    elif command == b"CALB" and save:
                        # File is visible before the asynchronous writer finishes.
                        raw = response_bytes(role, channel, symbols=value)
                        with output.open("wb") as stream:
                            stream.write(raw[:hsys.HSYS_HEADER.size])
                            stream.flush()
                            time.sleep(.08)
                            stream.write(raw[hsys.HSYS_HEADER.size:])
            except Exception as exc:
                state["errors"].append(exc)
                ready.set()
            finally:
                sock.close(linger=0)

        worker = threading.Thread(target=server)
        worker.start()
        self.assertTrue(ready.wait(2))
        args = SimpleNamespace(backend_dir=self.root, role=role, channel=channel,
                               host="127.0.0.1", control_port=state["port"],
                               symbols=32, connect_timeout=.5, timeout=.3, yes=True)
        try:
            with redirect_stdout(io.StringIO()):
                if version != 9:
                    with self.assertRaisesRegex(ValueError, "incompatible"):
                        hsys.capture(args)
                elif not save:
                    with self.assertRaises(TimeoutError):
                        hsys.capture(args)
                else:
                    result = hsys.capture(args)
                    self.assertEqual(result["captured_symbols"], 32)
                    self.assertEqual(result["channel"], channel)
                    backups = list(self.root.glob("*.bak"))
                    self.assertEqual(len(backups), 1)
                    self.assertEqual(backups[0].read_bytes(), old)
        finally:
            stop.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertFalse(state["errors"])
        commands = [(c, v) for h, c, v in state["commands"] if h == b"CMD "]
        expected = [] if version != 9 else ([(b"ALCH", channel)] if role == "monostatic" else []) + [(b"CALB", 32)]
        self.assertEqual(commands, expected)  # No TXGN, RXGN or CALD on any path.

    def test_monostatic_transport_and_backup(self):
        self.run_capture()

    def test_bistatic_transport(self):
        self.run_capture("bistatic")

    def test_incompatible_backend_sends_no_mutations(self):
        self.run_capture(version=8)

    def test_capture_timeout_does_not_restore_gain(self):
        self.run_capture(save=False)


def delay_line(seq, value=63, ratio=20, channel=0, target=60, gpu=False):
    return (f"[{'CUDA ' if gpu else ''}SYSDLY CH {channel}] "
            f"{'seq' if gpu else 'frame_seq'}={seq}, "
            f"{'ratio' if gpu else 'corr_ratio'}={ratio}, "
            f"{'target_algn' if gpu else 'target_alignment'}={target}, "
            f"{'suggest' if gpu else 'alignment_suggest'}={value}, next_frame_seq={seq + 100}")


class DelayTests(unittest.TestCase):
    def test_multiple_channels_and_log_formats(self):
        lines = [delay_line(i, channel=ch, gpu=bool(ch)) for i in range(15) for ch in range(2)]
        results = analyze(lines)
        self.assertEqual([r["alignment"] for r in results], [63, 63])

    def test_unstable_and_low_quality_do_not_recommend(self):
        for last in [delay_line(11, value=80), delay_line(11, ratio=1), delay_line(11, ratio="nan"),
                     delay_line(11, value=2**31), delay_line(9, ratio=1)]:
            result = analyze([delay_line(i) for i in range(10)] + [last])[0]
            self.assertFalse(result["stable"])
            self.assertIsNone(result["alignment"])

    def test_restart_and_alignment_change_reset_history(self):
        for last in [delay_line(0), delay_line(11, target=61)]:
            result = analyze([delay_line(i) for i in range(10)] + [last])[0]
            self.assertEqual(result["samples"], 1)
            self.assertFalse(result["stable"])

    def test_duplicates_and_recent_window(self):
        self.assertFalse(analyze([delay_line(1)] * 20)[0]["stable"])
        result = analyze([delay_line(i, value=1000 if i < 10 else 63) for i in range(40)])[0]
        self.assertTrue(result["stable"])
        self.assertEqual(result["samples"], 20)

    def test_threshold_validation(self):
        for kwargs in [dict(window=0), dict(min_samples=21), dict(max_spread=-1), dict(min_ratio=float("nan"))]:
            with self.assertRaises(ValueError):
                analyze([], **kwargs)


if __name__ == "__main__":
    unittest.main()
