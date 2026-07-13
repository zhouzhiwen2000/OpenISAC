from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore
from PyQt6 import QtWidgets

try:
    from qt_debug_plot_common import (
        PLOT_BACKGROUND,
        PLOT_FOREGROUND,
        DebugPlotWindow,
        configure_plot,
        parse_viewer_args,
        power_db,
    )
except ImportError:
    from scripts.qt_debug_plot_common import (
        PLOT_BACKGROUND,
        PLOT_FOREGROUND,
        DebugPlotWindow,
        configure_plot,
        parse_viewer_args,
        power_db,
    )


DEFAULT_PORT = 12349
FFT_SIZE = 1024


class PdfWindow(DebugPlotWindow):
    def __init__(self, host: str, port: int, interval_ms: int) -> None:
        self.x = np.arange(FFT_SIZE)
        super().__init__(
            title="Channel Delay Profile",
            default_port=port,
            host=host,
            interval_ms=interval_ms,
        )
        self.plot = pg.PlotWidget()
        configure_plot(self.plot, "Channel Delay Profile with Peak Marker", "Power (dB)", "Delay Tap")
        self.curve = self.plot.plot(self.x, np.zeros(FFT_SIZE), pen=pg.mkPen("#2563eb", width=1.4))
        self.peak_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("#dc2626", width=1.2, style=QtCore.Qt.PenStyle.DashLine))
        self.plot.addItem(self.peak_line)
        self.peak_label = pg.TextItem(
            "",
            anchor=(0, 1),
            color=PLOT_FOREGROUND,
            fill=pg.mkBrush(PLOT_BACKGROUND),
            border=pg.mkPen("#64748b"),
        )
        self.plot.addItem(self.peak_label)
        self.plot_layout.addWidget(self.plot)

    def handle_message(self, data: bytes) -> None:
        expected = FFT_SIZE * np.dtype(np.complex64).itemsize
        if len(data) != expected:
            self.status_label.setText(f"Ignored {len(data)} B frame; expected {expected} B")
            return

        cdata = np.frombuffer(data, dtype=np.complex64)
        power = power_db(cdata)
        max_index = int(np.argmax(power))
        max_value = float(power[max_index])
        adjusted_index = max_index if max_index < FFT_SIZE // 2 else max_index - FFT_SIZE

        self.curve.setData(self.x, power)
        self.peak_line.setPos(max_index)
        self.peak_label.setText(f"Peak {max_value:.1f} dB\nIndex {max_index} ({adjusted_index})")
        self.peak_label.setPos(max_index, max_value)
        self.plot.setYRange(max_value - 60.0, max_value + 5.0, padding=0)


def main() -> None:
    args = parse_viewer_args("Plot delay-profile debug stream.", DEFAULT_PORT)
    app = QtWidgets.QApplication([])
    window = PdfWindow(args.host, args.port, args.interval_ms)
    window.resize(1000, 620)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
