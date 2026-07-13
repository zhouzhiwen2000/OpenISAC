from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore
from PyQt6 import QtWidgets

try:
    from qt_debug_plot_common import DebugPlotWindow, configure_plot, parse_viewer_args, power_db
except ImportError:
    from scripts.qt_debug_plot_common import DebugPlotWindow, configure_plot, parse_viewer_args, power_db


DEFAULT_PORT = 12362
MAGIC = b"ERTMDBG1"
HEADER_FORMAT_V1 = "<8sIIIIqii7d"
HEADER_FORMAT_V3 = "<8sIIIIIIqii7d"
HEADER_SIZE_V1 = struct.calcsize(HEADER_FORMAT_V1)
HEADER_SIZE_V3 = struct.calcsize(HEADER_FORMAT_V3)


@dataclass(frozen=True)
class ErtmDebugFrame:
    fft_size: int
    base_fft_size: int
    oversample_factor: int
    seq: int
    peak_index: int
    signed_shift_bins: int
    duti_samples: int
    tadv_samples: int
    sample_rate: float
    metric: float
    rf_delay_samples: float
    tau_c_samples: float
    to_bs_ue_samples: float
    to_ue_samples: float
    to_bs_samples: float
    uplink_delay: np.ndarray
    downlink_delay: np.ndarray
    correlation: np.ndarray
    corrected_uplink_delay: np.ndarray | None
    corrected_downlink_delay: np.ndarray | None


def parse_frame(parts: list[bytes]) -> ErtmDebugFrame:
    if len(parts) not in (4, 6):
        raise ValueError(f"expected 4 or 6 ZMQ parts, got {len(parts)}")
    if len(parts[0]) not in (HEADER_SIZE_V1, HEADER_SIZE_V3):
        raise ValueError(f"bad eRTM debug header size: {len(parts[0])}")

    header_size = len(parts[0])
    unpacked = struct.unpack(HEADER_FORMAT_V3 if header_size == HEADER_SIZE_V3 else HEADER_FORMAT_V1, parts[0])
    magic = unpacked[0]
    if magic != MAGIC:
        raise ValueError(f"bad eRTM debug magic: {magic!r}")

    version = unpacked[1]
    if header_size == HEADER_SIZE_V3:
        (
            _magic,
            _version,
            fft_size,
            base_fft_size,
            oversample_factor,
            seq,
            peak_index,
            signed_shift_bins,
            duti_samples,
            tadv_samples,
            sample_rate,
            metric,
            rf_delay_samples,
            tau_c_samples,
            to_bs_ue_samples,
            to_ue_samples,
            to_bs_samples,
        ) = unpacked
    else:
        (
            _magic,
            _version,
            fft_size,
            seq,
            peak_index,
            signed_shift_bins,
            duti_samples,
            tadv_samples,
            sample_rate,
            metric,
            rf_delay_samples,
            tau_c_samples,
            to_bs_ue_samples,
            to_ue_samples,
            to_bs_samples,
        ) = unpacked
        base_fft_size = fft_size
        oversample_factor = 1
    if version not in (1, 2, 3):
        raise ValueError(f"unsupported eRTM debug version: {version}")
    if version in (2, 3) and len(parts) != 6:
        raise ValueError(f"eRTM debug version {version} requires corrected delay spectra")

    complex_bytes = fft_size * np.dtype("<c8").itemsize
    corr_bytes = fft_size * np.dtype("<f4").itemsize
    if len(parts[1]) != complex_bytes or len(parts[2]) != complex_bytes:
        raise ValueError("delay spectrum byte length does not match fft_size")
    if len(parts[3]) != corr_bytes:
        raise ValueError("correlation spectrum byte length does not match fft_size")
    corrected_uplink_delay = None
    corrected_downlink_delay = None
    if version in (2, 3):
        if len(parts[4]) != complex_bytes or len(parts[5]) != complex_bytes:
            raise ValueError("corrected delay spectrum byte length does not match fft_size")
        corrected_uplink_delay = np.frombuffer(parts[4], dtype="<c8").copy()
        corrected_downlink_delay = np.frombuffer(parts[5], dtype="<c8").copy()

    return ErtmDebugFrame(
        fft_size=fft_size,
        base_fft_size=base_fft_size,
        oversample_factor=oversample_factor,
        seq=seq,
        peak_index=peak_index,
        signed_shift_bins=signed_shift_bins,
        duti_samples=duti_samples,
        tadv_samples=tadv_samples,
        sample_rate=sample_rate,
        metric=metric,
        rf_delay_samples=rf_delay_samples,
        tau_c_samples=tau_c_samples,
        to_bs_ue_samples=to_bs_ue_samples,
        to_ue_samples=to_ue_samples,
        to_bs_samples=to_bs_samples,
        uplink_delay=np.frombuffer(parts[1], dtype="<c8").copy(),
        downlink_delay=np.frombuffer(parts[2], dtype="<c8").copy(),
        correlation=np.frombuffer(parts[3], dtype="<f4").copy(),
        corrected_uplink_delay=corrected_uplink_delay,
        corrected_downlink_delay=corrected_downlink_delay,
    )


def signed_axis(n: int) -> np.ndarray:
    raw = np.arange(n, dtype=np.int64)
    raw[raw >= (n + 1) // 2] -= n
    return np.fft.fftshift(raw)


def delay_axis(n: int, oversample_factor: int) -> np.ndarray:
    factor = max(int(oversample_factor), 1)
    return signed_axis(n).astype(np.float64, copy=False) / float(factor)


def set_full_x_range(plot: pg.PlotWidget, x: np.ndarray) -> None:
    if x.size == 0:
        return
    plot.setXRange(float(x[0]) - 0.5, float(x[-1]) + 0.5, padding=0)


def shifted_power_db(values: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(power_db(values))


def shifted_real(values: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(values.astype(np.float64, copy=False))


class ErtmDebugWindow(DebugPlotWindow):
    def __init__(self, host: str, port: int, interval_ms: int) -> None:
        super().__init__(
            title="eRTM Debug Spectra",
            default_port=port,
            host=host,
            interval_ms=interval_ms,
            multipart=True,
            conflate=False,
        )
        self.info_label = QtWidgets.QLabel("Waiting for eRTM debug frames")
        self.info_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.root_layout.insertWidget(1, self.info_label)

        self.to_correction_check = QtWidgets.QCheckBox("TO correction")
        self.to_correction_check.setToolTip(
            "Display C++ frequency-domain TO-compensated uplink/downlink delay spectra when available."
        )
        self.root_layout.insertWidget(2, self.to_correction_check)

        self.uplink_plot = pg.PlotWidget()
        self.downlink_plot = pg.PlotWidget()
        self.corr_plot = pg.PlotWidget()
        configure_plot(self.uplink_plot, "BS Uplink Delay Spectrum", "Power (dB)", "Signed Delay (samples)")
        configure_plot(self.downlink_plot, "UE Downlink Delay Spectrum", "Power (dB)", "Signed Delay (samples)")
        configure_plot(self.corr_plot, "eRTM Correlation Spectrum", "Correlation", "Signed Shift (samples)")

        self.uplink_curve = self.uplink_plot.plot([], [], pen=pg.mkPen("#2563eb", width=1.3))
        self.downlink_curve = self.downlink_plot.plot([], [], pen=pg.mkPen("#16a34a", width=1.3))
        self.corr_curve = self.corr_plot.plot([], [], pen=pg.mkPen("#7c3aed", width=1.3))
        self.peak_line = pg.InfiniteLine(
            pos=0,
            angle=90,
            pen=pg.mkPen("#dc2626", width=1.2, style=QtCore.Qt.PenStyle.DashLine),
        )
        self.corr_plot.addItem(self.peak_line)

        self.plot_layout.addWidget(self.uplink_plot)
        self.plot_layout.addWidget(self.downlink_plot)
        self.plot_layout.addWidget(self.corr_plot)

    def handle_message(self, parts: list[bytes]) -> None:
        frame = parse_frame(parts)
        x = delay_axis(frame.fft_size, frame.oversample_factor)
        correction_enabled = self.to_correction_check.isChecked()
        uplink_delay = frame.uplink_delay
        downlink_delay = frame.downlink_delay
        if correction_enabled:
            if frame.corrected_uplink_delay is not None:
                uplink_delay = frame.corrected_uplink_delay
            if frame.corrected_downlink_delay is not None:
                downlink_delay = frame.corrected_downlink_delay
        uplink_power = shifted_power_db(uplink_delay)
        downlink_power = shifted_power_db(downlink_delay)
        corr = shifted_real(frame.correlation)

        self.uplink_curve.setData(x, uplink_power)
        self.downlink_curve.setData(x, downlink_power)
        self.corr_curve.setData(x, corr)
        peak_x = frame.to_bs_ue_samples
        self.peak_line.setPos(peak_x)

        set_full_x_range(self.uplink_plot, x)
        set_full_x_range(self.downlink_plot, x)
        set_full_x_range(self.corr_plot, x)
        self.uplink_plot.setYRange(float(np.max(uplink_power) - 60.0), float(np.max(uplink_power) + 5.0), padding=0)
        self.downlink_plot.setYRange(float(np.max(downlink_power) - 60.0), float(np.max(downlink_power) + 5.0), padding=0)
        corr_min = float(np.min(corr))
        corr_max = float(np.max(corr))
        pad = max((corr_max - corr_min) * 0.08, 1e-3)
        self.corr_plot.setYRange(corr_min - pad, corr_max + pad, padding=0)

        self.info_label.setText(
            f"seq={frame.seq} | peak_index={frame.peak_index} | "
            f"peak_sample={frame.to_bs_ue_samples:.3f} | shift={frame.to_bs_ue_samples:.3f} samples | "
            f"os={frame.oversample_factor}x | metric={frame.metric:.6f} | "
            f"TO_corr={'on' if correction_enabled and frame.corrected_uplink_delay is not None else 'off'} | "
            f"tau_c={frame.tau_c_samples:.3f} | TO_UE={frame.to_ue_samples:.3f} | "
            f"TO_BS={frame.to_bs_samples:.3f} | DUTI={frame.duti_samples} | TADV={frame.tadv_samples}"
        )


def main() -> None:
    args = parse_viewer_args("Plot UE eRTM debug spectra.", DEFAULT_PORT)
    app = QtWidgets.QApplication([])
    window = ErtmDebugWindow(args.host, args.port, args.interval_ms)
    window.resize(1120, 860)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
