from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets

try:
    from qt_debug_plot_common import DebugPlotWindow, configure_plot, parse_viewer_args
except ImportError:
    from scripts.qt_debug_plot_common import DebugPlotWindow, configure_plot, parse_viewer_args


DEFAULT_PORT = 12348
FFT_SIZE = 1024


def _positive_span(value: float, fallback: float = 1.0) -> float:
    return max(float(value), fallback)


class ChannelWindow(DebugPlotWindow):
    def __init__(self, host: str, port: int, interval_ms: int) -> None:
        self.x = np.arange(FFT_SIZE)
        super().__init__(
            title="Channel Response",
            default_port=port,
            host=host,
            interval_ms=interval_ms,
        )
        self.real_plot = pg.PlotWidget()
        self.imag_plot = pg.PlotWidget()
        self.mag_plot = pg.PlotWidget()
        configure_plot(self.real_plot, "Channel Response - Real Part", "Real", "Frequency Bin")
        configure_plot(self.imag_plot, "Channel Response - Imaginary Part", "Imaginary", "Frequency Bin")
        configure_plot(self.mag_plot, "Channel Response - Magnitude", "Magnitude", "Frequency Bin")
        self.real_curve = self.real_plot.plot(self.x, np.zeros(FFT_SIZE), pen=pg.mkPen("#2563eb", width=1.4))
        self.imag_curve = self.imag_plot.plot(self.x, np.zeros(FFT_SIZE), pen=pg.mkPen("#dc2626", width=1.4))
        self.mag_curve = self.mag_plot.plot(self.x, np.zeros(FFT_SIZE), pen=pg.mkPen("#16a34a", width=1.4))
        self.plot_layout.addWidget(self.real_plot)
        self.plot_layout.addWidget(self.imag_plot)
        self.plot_layout.addWidget(self.mag_plot)

    def handle_message(self, data: bytes) -> None:
        expected = FFT_SIZE * np.dtype(np.complex64).itemsize
        if len(data) != expected:
            self.status_label.setText(f"Ignored {len(data)} B frame; expected {expected} B")
            return
        cdata = np.fft.fftshift(np.frombuffer(data, dtype=np.complex64))
        real_part = cdata.real
        imag_part = cdata.imag
        magnitude = np.abs(cdata)

        self.real_curve.setData(self.x, real_part)
        self.imag_curve.setData(self.x, imag_part)
        self.mag_curve.setData(self.x, magnitude)

        real_max = _positive_span(np.max(np.abs(real_part)))
        imag_max = _positive_span(np.max(np.abs(imag_part)))
        mag_max = _positive_span(np.max(magnitude))
        self.real_plot.setYRange(-real_max * 1.1, real_max * 1.1, padding=0)
        self.imag_plot.setYRange(-imag_max * 1.1, imag_max * 1.1, padding=0)
        self.mag_plot.setYRange(0, mag_max * 1.1, padding=0)


def main() -> None:
    args = parse_viewer_args("Plot channel-estimate debug stream.", DEFAULT_PORT)
    app = QtWidgets.QApplication([])
    window = ChannelWindow(args.host, args.port, args.interval_ms)
    window.resize(1000, 820)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
