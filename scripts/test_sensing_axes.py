from __future__ import annotations

from dataclasses import replace
import struct
import unittest

import numpy as np

from sensing_physical_axes import C_LIGHT_MPS, display_doppler_mapping, image_bounds_from_centers, physical_axes
from sensing_runtime_protocol import (
    FLAG_BISTATIC, PARAMS_PACKET_STRUCT, PARAMS_VERSION, ViewerRuntimeParams, parse_params_packet,
)


class PhysicalAxesTest(unittest.TestCase):
    def setUp(self):
        self.params = ViewerRuntimeParams(
            version=PARAMS_VERSION, sample_rate_hz=50e6, ofdm_fft_size=1024,
            cp_length=128, sensing_symbol_stride=8, sensing_subcarrier_stride=1,
        )

    def test_mono_range_and_bistatic_ns_with_zero_padding(self):
        mono = physical_axes(self.params, 2048, 512)
        bi = physical_axes(replace(self.params, flags=FLAG_BISTATIC), 2048, 512)
        self.assertAlmostEqual(mono.delay_bin_ns, 10.0)
        self.assertAlmostEqual(mono.range_bin_scale, C_LIGHT_MPS * 5e-9)
        self.assertAlmostEqual(bi.range_bin_scale, 10.0)
        self.assertEqual(bi.range_unit, "ns")
        self.assertAlmostEqual(mono.doppler_bin_hz, 50e6 / (1152 * 8 * 512))

    def test_synthetic_echo_keeps_units_after_padding_crop_and_decimation(self):
        # A 120 ns echo with 8 slow-time cycles per 128 samples: physical
        # location must survive doubling both FFT lengths and cropping the image.
        params = replace(self.params, ofdm_fft_size=64, cp_length=8, sensing_symbol_stride=4)
        tau = 120e-9
        slow_rate = 50e6 / (72 * 4)
        fd = slow_rate * 8 / 128
        k = np.arange(64)
        t = np.arange(128) / slow_rate
        samples = np.exp(-2j * np.pi * k[None, :] * (50e6 / 64) * tau
                         + 2j * np.pi * fd * t[:, None])
        for nr, nd in [(64, 128), (128, 256)]:
            rd = np.fft.fftshift(np.fft.fft(np.fft.ifft(samples, n=nr, axis=1), n=nd, axis=0), axes=0)
            start = nd // 2 - 32
            cropped = np.abs(rd[start:start + 64:2, :32:2])
            row, col = np.unravel_index(np.argmax(cropped), cropped.shape)
            axes = physical_axes(params, nr, nd)
            factor, offset = display_doppler_mapping(nd, 64, cropped.shape[0], 2, axes.doppler_bin_hz)
            self.assertAlmostEqual((row - cropped.shape[0] / 2) * factor + offset, fd)
            self.assertAlmostEqual(col * 2 * axes.delay_bin_ns, tau * 1e9)

    def test_odd_crop_and_backend_cropped_fft(self):
        axes = physical_axes(self.params, 2048, 1024)
        # Backend sent 101 centered FFT bins; UI displays 51, every third bin.
        factor, offset = display_doppler_mapping(101, 51, 17, 3, axes.doppler_bin_hz)
        for row in (0, 8, 16):
            actual = (row - 17 / 2) * factor + offset
            self.assertAlmostEqual(actual, (-25 + row * 3) * axes.doppler_bin_hz)

    def test_runtime_stride_and_compact_frequency_spacing(self):
        a = physical_axes(self.params, 2048, 512)
        b = physical_axes(replace(self.params, sensing_symbol_stride=16, sensing_subcarrier_stride=2), 2048, 512)
        self.assertEqual(b.doppler_bin_hz, a.doppler_bin_hz / 2)
        self.assertEqual(b.delay_bin_ns, a.delay_bin_ns / 2)
        nonuniform = physical_axes(replace(self.params, sensing_subcarrier_stride=0), 2048, 512)
        self.assertTrue(np.isnan(nonuniform.delay_bin_ns))

    def test_exact_protocol_only(self):
        fields = [PARAMS_VERSION, FLAG_BISTATIC, 0, 100, 2048, 100, 1024,
                  100, 2048, 512, 0, 0, 1, 1, 7500, 2, 2,
                  58_000_000, 500_000, 0, 1024, 128, 8, 1, 5]
        packet = PARAMS_PACKET_STRUCT.pack(b"CTRL", b"PARM", *fields)
        self.assertEqual(len(packet), 108)
        p = parse_params_packet(packet)
        self.assertEqual(p.center_freq_hz, 5.8e9)
        self.assertEqual(p.sensing_symbol_stride, 8)
        self.assertEqual(p.sensing_subcarrier_stride, 1)
        self.assertEqual(p.frame_decimation, 5)
        a = physical_axes(p, 2048, 512)
        b = physical_axes(replace(p, frame_decimation=1), 2048, 512)
        self.assertEqual(a, b)
        self.assertEqual(physical_axes(p, 2048, 512).range_unit, "ns")
        with self.assertLogs("sensing_runtime_protocol", level="WARNING"):
            for old_size in (52, 60, 64, 76, 88, 100, 104):
                self.assertIsNone(parse_params_packet(packet[:old_size]))
            self.assertIsNone(parse_params_packet(packet + b"\x00"))
            for version in (7, 8, 9, 11):
                invalid = packet[:8] + struct.pack("!I", version) + packet[12:]
                self.assertIsNone(parse_params_packet(invalid))

    def test_stft_pixels_match_first_and_last_sample_centers(self):
        x, y, width, height = image_bounds_from_centers([0, 128, -0.5, 127 / 256], 256, 3)
        self.assertAlmostEqual(x + width / 3 / 2, 0)
        self.assertAlmostEqual(x + width * 2.5 / 3, 128)
        self.assertAlmostEqual(y + height / 256 / 2, -0.5)
        self.assertAlmostEqual(y + height * 255.5 / 256, 127 / 256)


if __name__ == "__main__":
    unittest.main()
