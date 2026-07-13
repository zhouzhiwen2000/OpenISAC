from __future__ import annotations


def main(mode: str) -> int:
    """Launch one of the fast sensing viewers.

    Imports are intentionally lazy because the mode-specific viewer modules
    create sockets and worker threads during module initialization.
    """
    from .ui import run_fast_viewer

    return int(run_fast_viewer(mode))
