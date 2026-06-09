"""
Shared collapsible panel widget and settings persistence for OpenISAC viewer scripts.

Provides:
- CollapsibleSection: A QWidget with a clickable header that expands/collapses
  to show/hide a content widget. Multiple sections can be stacked vertically.
- ViewerSettings: JSON-based persistence for viewer parameters, saved to
  ~/.openisac_viewer_settings_<script_name>.json on every change.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg


# ---------------------------------------------------------------------------
# Shared viewer theme
# ---------------------------------------------------------------------------

VIEWER_COLORS = {
    "window": "#08111f",
    "panel": "#0f1b2d",
    "panel_alt": "#142238",
    "plot": "#07101d",
    "border": "#29405f",
    "grid": "#526985",
    "text": "#dbeafe",
    "muted": "#8fa6c0",
    "accent": "#38bdf8",
    "accent_hover": "#67d4ff",
    "success": "#34d399",
    "warning": "#fbbf24",
    "danger": "#fb7185",
    "target": "#ffb86b",
    "raw": "#ffd166",
    "calibrated": "#4cc9f0",
}


def sensing_colormap() -> pg.ColorMap:
    """Return a dark-background sequential map for power spectra."""
    positions = [0.0, 0.18, 0.38, 0.58, 0.77, 0.91, 1.0]
    colors = [
        "#050816",
        "#172554",
        "#4338ca",
        "#0891b2",
        "#22c55e",
        "#facc15",
        "#fff7d6",
    ]
    return pg.ColorMap(positions, [QtGui.QColor(color) for color in colors])


def apply_viewer_theme(app: QtWidgets.QApplication):
    """Apply the shared OpenISAC dark theme to a viewer application."""
    app.setStyle("Fusion")
    pg.setConfigOptions(
        background=VIEWER_COLORS["plot"],
        foreground=VIEWER_COLORS["text"],
        antialias=True,
    )
    app.setStyleSheet(f"""
        QWidget {{
            background-color: {VIEWER_COLORS["window"]};
            color: {VIEWER_COLORS["text"]};
            font-size: 12px;
        }}
        QMainWindow, QDialog {{
            background-color: {VIEWER_COLORS["window"]};
        }}
        QScrollArea, QScrollArea > QWidget > QWidget {{
            background-color: {VIEWER_COLORS["panel"]};
            border: none;
        }}
        QLabel {{
            background-color: transparent;
        }}
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {{
            background-color: {VIEWER_COLORS["panel_alt"]};
            border: 1px solid {VIEWER_COLORS["border"]};
            border-radius: 5px;
            padding: 5px 7px;
            selection-background-color: {VIEWER_COLORS["accent"]};
            selection-color: {VIEWER_COLORS["window"]};
        }}
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
            border-color: {VIEWER_COLORS["accent"]};
        }}
        QPushButton {{
            background-color: {VIEWER_COLORS["panel_alt"]};
            border: 1px solid {VIEWER_COLORS["border"]};
            border-radius: 5px;
            padding: 5px 10px;
            font-weight: 600;
        }}
        QPushButton:hover {{
            background-color: #1b3150;
            border-color: {VIEWER_COLORS["accent"]};
        }}
        QPushButton:pressed {{
            background-color: #204363;
        }}
        QPushButton:checked {{
            background-color: #126044;
            border-color: {VIEWER_COLORS["success"]};
        }}
        QPushButton[active="true"] {{
            background-color: #126044;
            border-color: {VIEWER_COLORS["success"]};
        }}
        QPushButton:disabled {{
            color: #60758f;
            background-color: #101a29;
            border-color: #22344d;
        }}
        QCheckBox, QRadioButton {{
            spacing: 6px;
            background-color: transparent;
        }}
        QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
            background-color: {VIEWER_COLORS["accent"]};
            border: 1px solid {VIEWER_COLORS["accent_hover"]};
        }}
        QScrollBar:vertical {{
            background: {VIEWER_COLORS["panel"]};
            width: 10px;
            margin: 0;
        }}
        QScrollBar::handle:vertical {{
            background: #365270;
            min-height: 28px;
            border-radius: 5px;
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0;
        }}
        QToolTip {{
            background-color: {VIEWER_COLORS["panel_alt"]};
            color: {VIEWER_COLORS["text"]};
            border: 1px solid {VIEWER_COLORS["accent"]};
            padding: 4px;
        }}
    """)


def style_spectrum_plot(plot: pg.PlotWidget, *, grid: bool = True):
    """Apply consistent axes, background, and grid styling to a plot."""
    plot.setBackground(VIEWER_COLORS["plot"])
    plot.getPlotItem().setContentsMargins(8, 8, 8, 8)
    for axis_name in ("left", "bottom"):
        axis = plot.getAxis(axis_name)
        axis.setPen(pg.mkPen(VIEWER_COLORS["border"], width=1))
        axis.setTextPen(pg.mkPen(VIEWER_COLORS["muted"]))
    if grid:
        plot.showGrid(x=True, y=True, alpha=0.14)


def set_button_active(button: QtWidgets.QPushButton, active: bool):
    """Update a non-checkable toggle button while preserving the app theme."""
    button.setProperty("active", bool(active))
    button.style().unpolish(button)
    button.style().polish(button)


# ---------------------------------------------------------------------------
# CollapsibleSection
# ---------------------------------------------------------------------------

_COLLAPSED_ARROW = "▸"
_EXPANDED_ARROW = "▾"

_HEADER_STYLE = """
QPushButton {
    text-align: left;
    padding: 7px 10px;
    border: 1px solid #29405f;
    border-radius: 5px;
    background-color: #142238;
    font-weight: bold;
    font-size: 13px;
    color: #dbeafe;
}
QPushButton:hover {
    background-color: #1b3150;
    border-color: #38bdf8;
}
"""


class CollapsibleSection(QtWidgets.QWidget):
    """A collapsible section with a clickable header and hideable content.

    Usage::

        section = CollapsibleSection("Display Controls")
        content_layout = section.content_layout()
        content_layout.addWidget(...)
        parent_layout.addWidget(section)
    """

    def __init__(
        self,
        title: str,
        parent: Optional[QtWidgets.QWidget] = None,
        collapsed: bool = False,
    ):
        super().__init__(parent)
        self._collapsed = collapsed
        self._title = title

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(2)

        # Header button
        self._header_btn = QtWidgets.QPushButton(self._make_header_text())
        self._header_btn.setStyleSheet(_HEADER_STYLE)
        self._header_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._header_btn.clicked.connect(self._toggle)
        main_layout.addWidget(self._header_btn)

        # Content widget
        self._content = QtWidgets.QWidget()
        self._content.setVisible(not collapsed)
        self._content_layout = QtWidgets.QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(6, 4, 6, 4)
        self._content_layout.setSpacing(4)
        main_layout.addWidget(self._content)

    def _make_header_text(self) -> str:
        arrow = _COLLAPSED_ARROW if self._collapsed else _EXPANDED_ARROW
        return f"  {arrow}  {self._title}"

    def _toggle(self):
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool):
        self._collapsed = collapsed
        self._header_btn.setText(self._make_header_text())
        self._content.setVisible(not collapsed)

    def is_collapsed(self) -> bool:
        return self._collapsed

    def content_layout(self) -> QtWidgets.QVBoxLayout:
        """Return the layout inside the collapsible content area.

        Add your parameter widgets to this layout.
        """
        return self._content_layout

    def header_button(self) -> QtWidgets.QPushButton:
        """Expose the header button (e.g., for custom styling)."""
        return self._header_btn


# ---------------------------------------------------------------------------
# ViewerSettings - JSON-based parameter persistence
# ---------------------------------------------------------------------------

SETTINGS_DIR = Path.home() / ".openisac_viewer_settings"


def load_viewer_setting(script_name: str, key: str, default: Any = None) -> Any:
    """Load one saved viewer setting without creating Qt timer state."""
    path = SETTINGS_DIR / f"{script_name}.json"
    if not path.is_file():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default
    return data.get(key, default)


class ViewerSettings:
    """Per-script parameter persistence backed by a JSON file.

    Parameters that change during a session are automatically saved and
    restored when the viewer is reopened.

    Usage::

        settings = ViewerSettings("plot_sensing_fast")
        # After creating all UI widgets:
        settings.restore_all(callbacks={
            "host": lambda v: txt_backend_host.setText(v),
            "range_bin": lambda v: txt_range_bin.setText(str(v)),
            ...
        })
        # Connect widget changes:
        txt_range_bin.textChanged.connect(lambda v: settings.set("range_bin", v))
    """

    def __init__(self, script_name: str):
        SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        self._path = SETTINGS_DIR / f"{script_name}.json"
        self._data: Dict[str, Any] = {}
        self._dirty = False
        self._save_timer = QtCore.QTimer()
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)  # debounce 500 ms
        self._save_timer.timeout.connect(self._flush)
        self._load()

    # -- public API ---------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Return a saved value, or *default* if missing."""
        return self._data.get(key, default)

    def set(self, key: str, value: Any):
        """Store a value and schedule a save."""
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            # Non-serializable; store as string representation
            value = str(value)
        self._data[key] = value
        self._mark_dirty()

    def restore_all(self, callbacks: Dict[str, Callable[[Any], None]]):
        """Apply all saved values via callbacks.

        *callbacks* maps setting key -> callable(value).  Only keys present
        in the saved file are dispatched; missing keys are silently skipped.
        """
        for key, cb in callbacks.items():
            if key in self._data:
                try:
                    cb(self._data[key])
                except Exception:
                    pass  # malformed value; ignore

    def save_now(self):
        """Flush immediately (useful before exit)."""
        self._flush()

    # -- internals ----------------------------------------------------------

    def _load(self):
        if self._path.is_file():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}
        else:
            self._data = {}

    def _mark_dirty(self):
        if not self._dirty:
            self._dirty = True
            self._save_timer.start()

    def _flush(self):
        if not self._dirty:
            return
        try:
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._path)
            self._dirty = False
        except OSError:
            pass  # best-effort
