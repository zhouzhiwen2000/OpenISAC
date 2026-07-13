from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets

try:
    from qt_debug_plot_common import (
        PLOT_BACKGROUND,
        PLOT_FOREGROUND,
        DebugPlotWindow,
        configure_plot,
        parse_viewer_args,
    )
except ImportError:
    from scripts.qt_debug_plot_common import (
        PLOT_BACKGROUND,
        PLOT_FOREGROUND,
        DebugPlotWindow,
        configure_plot,
        parse_viewer_args,
    )


DEFAULT_PORT = 12346
FFT_SIZE = 1024
SHOW_GUARD_BAND = False

PILOT_INDICES = np.array([571, 631, 692, 752, 812, 872, 933, 993, 29, 89, 150, 210, 270, 330, 391, 451])
ALL_ACTIVE_INDICES = np.concatenate([np.arange(1, 490), np.arange(535, 1024)])
DATA_INDICES = np.setdiff1d(ALL_ACTIVE_INDICES, PILOT_INDICES)
GUARD_BAND_INDICES = np.arange(490, 534)


class ConstellationWindow(DebugPlotWindow):
    def __init__(self, host: str, port: int, interval_ms: int) -> None:
        super().__init__(
            title="OFDM Constellation",
            default_port=port,
            host=host,
            interval_ms=interval_ms,
        )
        self.plot = pg.PlotWidget()
        configure_plot(self.plot, "OFDM Constellation", "Imaginary (Q)", "Real (I)")
        self.plot.setXRange(-2.0, 2.0, padding=0)
        self.plot.setYRange(-2.0, 2.0, padding=0)
        self.plot.addLegend(
            offset=(12, 12),
            labelTextColor=PLOT_FOREGROUND,
            brush=pg.mkBrush(PLOT_BACKGROUND),
            pen=pg.mkPen("#64748b"),
        )

        self.data_scatter = pg.ScatterPlotItem(
            size=6,
            brush=pg.mkBrush(37, 99, 235, 150),
            pen=pg.mkPen(None),
            name="Data Subcarriers",
        )
        self.pilot_scatter = pg.ScatterPlotItem(
            size=10,
            symbol="x",
            brush=pg.mkBrush(249, 115, 22, 180),
            pen=pg.mkPen("#f97316", width=1.4),
            name="Pilot Subcarriers",
        )
        self.guard_scatter = pg.ScatterPlotItem(
            size=9,
            symbol="star",
            brush=pg.mkBrush(220, 38, 38, 150),
            pen=pg.mkPen("#dc2626", width=1.2),
            name="Guard Band",
        )
        self.plot.addItem(self.data_scatter)
        self.plot.addItem(self.pilot_scatter)
        if SHOW_GUARD_BAND:
            self.plot.addItem(self.guard_scatter)
        self.plot_layout.addWidget(self.plot)

        self.current_symbol = np.zeros(FFT_SIZE, dtype=np.complex64)
        self._update_scatter()

    def _update_scatter(self) -> None:
        data_sc = self.current_symbol[DATA_INDICES]
        pilot_sc = self.current_symbol[PILOT_INDICES]
        self.data_scatter.setData(data_sc.real, data_sc.imag)
        self.pilot_scatter.setData(pilot_sc.real, pilot_sc.imag)
        if SHOW_GUARD_BAND:
            guard_sc = self.current_symbol[GUARD_BAND_INDICES]
            self.guard_scatter.setData(guard_sc.real, guard_sc.imag)

    def handle_message(self, data: bytes) -> None:
        expected = FFT_SIZE * np.dtype(np.complex64).itemsize
        if len(data) != expected:
            self.status_label.setText(f"Ignored {len(data)} B frame; expected {expected} B")
            return
        symbol = np.frombuffer(data, dtype=np.complex64)
        if symbol.size != FFT_SIZE:
            self.status_label.setText(f"Ignored symbol with {symbol.size} bins")
            return
        self.current_symbol = symbol
        self._update_scatter()


def main() -> None:
    args = parse_viewer_args("Plot constellation debug stream.", DEFAULT_PORT)
    app = QtWidgets.QApplication([])
    window = ConstellationWindow(args.host, args.port, args.interval_ms)
    window.resize(820, 760)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
