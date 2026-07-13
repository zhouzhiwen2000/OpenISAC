from __future__ import annotations

import unittest

import numpy as np

from sensing_runtime_protocol import FRAME_FORMAT_DENSE_RANGE_DOPPLER, ViewerRuntimeParams

from sensing_viewer.config import default_config
from sensing_viewer.processing import (
    MicroDopplerBuffer,
    ProcessingOptions,
    process_range_doppler,
    process_range_doppler_batch,
)


class FastViewerConfigTest(unittest.TestCase):
    def test_default_configs(self):
        mono = default_config("mono")
        bi = default_config("bi")

        self.assertEqual(mono.default_port, 8888)
        self.assertEqual(mono.default_control_port, 9999)
        self.assertEqual(mono.default_channels, 2)
        self.assertEqual(mono.settings_key, "plot_sensing_fast")
        self.assertTrue(mono.supports_aggregate_stream)

        self.assertEqual(bi.default_port, 8889)
        self.assertEqual(bi.default_control_port, 10001)
        self.assertEqual(bi.default_channels, 1)
        self.assertEqual(bi.settings_key, "plot_bi_sensing_fast")
        self.assertFalse(bi.supports_aggregate_stream)
        self.assertTrue(bi.supports_os_cfar)
        self.assertFalse(bi.supports_phase_calibration)
        self.assertFalse(bi.supports_superresolution)


class ProcessingTest(unittest.TestCase):
    def test_dense_channel_buffer_processing(self):
        params = ViewerRuntimeParams(
            active_rows=8,
            active_cols=16,
            wire_rows=8,
            wire_cols=16,
            range_fft_size=16,
            doppler_fft_size=8,
        )
        frame = self._synthetic_frame(8, 16)
        opts = ProcessingOptions(
            range_fft_size=32,
            doppler_fft_size=16,
            display_range_bins=20,
            enable_range_window=False,
            enable_doppler_window=False,
        )

        result = process_range_doppler(frame, params, opts)
        padded = np.zeros((8, 32), dtype=np.complex64)
        padded[:, :16] = frame
        expected_range_time = np.fft.ifft(padded, axis=1) * 32

        self.assertEqual(result.magnitude_db.shape, (16, 20))
        self.assertEqual(result.magnitude_db.dtype, np.float32)
        self.assertEqual(result.range_time.shape, (8, 20))
        self.assertEqual(result.rd_complex.shape, (16, 20))
        self.assertTrue(np.all(np.isfinite(result.magnitude_db)))
        np.testing.assert_allclose(result.range_time, expected_range_time[:, :20], rtol=1e-6, atol=1e-6)

    def test_dense_range_doppler_passthrough(self):
        params = ViewerRuntimeParams(
            frame_format=FRAME_FORMAT_DENSE_RANGE_DOPPLER,
            wire_rows=6,
            wire_cols=10,
            active_rows=6,
            active_cols=10,
        )
        frame = self._synthetic_frame(6, 10)

        result = process_range_doppler(frame, params)

        self.assertEqual(result.magnitude_db.shape, (6, 10))
        self.assertIsNone(result.range_time)
        np.testing.assert_allclose(result.rd_complex, np.fft.fftshift(frame, axes=0).astype(np.complex64))

    def test_batch_matches_single(self):
        params = ViewerRuntimeParams(
            active_rows=8,
            active_cols=16,
            wire_rows=8,
            wire_cols=16,
            range_fft_size=16,
            doppler_fft_size=8,
        )
        frames = [(0, self._synthetic_frame(8, 16)), (1, self._synthetic_frame(8, 16) * 0.5)]
        opts = ProcessingOptions(range_fft_size=32, doppler_fft_size=16, display_range_bins=12)

        batch = process_range_doppler_batch(frames, params, opts)

        for ch_idx, frame in frames:
            single = process_range_doppler(frame, params, opts)
            np.testing.assert_allclose(batch[ch_idx].magnitude_db, single.magnitude_db, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(batch[ch_idx].rd_complex, single.rd_complex, rtol=1e-5, atol=1e-5)

    def test_micro_doppler_buffer(self):
        md = MicroDopplerBuffer()
        small = np.ones((16, 4), dtype=np.complex64)
        md.extend_range_bin(small, 1)
        self.assertIsNone(md.spectrum())

        enough = np.exp(1j * np.linspace(0, 8 * np.pi, 320, dtype=np.float32)).reshape(320, 1)
        md.extend_range_bin(enough.astype(np.complex64), 0)
        spectrum = md.spectrum()

        self.assertIsNotNone(spectrum)
        f, t, pxx = spectrum
        self.assertGreater(f.size, 0)
        self.assertGreater(t.size, 0)
        self.assertEqual(pxx.shape[0], f.size)
        self.assertEqual(pxx.shape[1], t.size)
        self.assertTrue(np.all(np.isfinite(pxx)))

    @staticmethod
    def _synthetic_frame(rows: int, cols: int) -> np.ndarray:
        rng = np.random.default_rng(1234 + rows + cols)
        real = rng.normal(size=(rows, cols)).astype(np.float32)
        imag = rng.normal(size=(rows, cols)).astype(np.float32)
        return (real + 1j * imag).astype(np.complex64)


if __name__ == "__main__":
    unittest.main()
