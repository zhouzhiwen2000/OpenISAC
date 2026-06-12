from __future__ import annotations


def _load_viewer_module(mode: str):
    import os

    normalized = (mode or "").strip().lower()
    if normalized not in {"mono", "bi"}:
        raise ValueError(f"Unsupported fast sensing viewer mode: {mode!r}")
    os.environ["OPENISAC_FAST_VIEWER_MODE"] = normalized
    from . import fast_viewer as viewer
    return viewer


def FastSensingWindow(mode: str):
    """Construct the mode-specific fast viewer window.

    The current implementation keeps the mode-heavy Qt classes in compatibility
    modules, while this factory gives callers a unified UI entry point.
    """
    return _load_viewer_module(mode).MainWindow()


def run_fast_viewer(mode: str) -> int:
    return _load_viewer_module(mode).run_app()
