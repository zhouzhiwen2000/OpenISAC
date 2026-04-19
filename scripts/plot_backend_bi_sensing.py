from __future__ import annotations

try:
    from backend_sensing_viewer import main
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing Python dependency: {exc.name}. "
        "Install the viewer GUI dependencies first, for example PyQt6 and pyqtgraph."
    ) from exc


if __name__ == "__main__":
    raise SystemExit(
        main(
            default_mode="bi",
            default_title="OpenISAC Backend Bi-Sensing Viewer",
        )
    )
