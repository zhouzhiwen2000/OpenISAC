#!/usr/bin/env python3
"""Publish Astro build output into docs/ without deleting unrelated files."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "site" / "dist"
DOCS = ROOT / "docs"
LEGACY_MANAGED_NAMES = (
    "style.css",
    "index_zh.html",
    "architecture.html",
    "architecture_zh.html",
    "signal_processing.html",
    "signal_processing_zh.html",
    "documentation",
    "documentation_zh",
)


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def main() -> None:
    if not DIST.exists():
        raise FileNotFoundError(f"Build output not found: {DIST}")

    DOCS.mkdir(parents=True, exist_ok=True)

    for legacy_name in LEGACY_MANAGED_NAMES:
        _remove_path(DOCS / legacy_name)

    for src in DIST.iterdir():
        dst = DOCS / src.name
        _remove_path(dst)
        _copy_path(src, dst)

    print(f"Published {DIST.relative_to(ROOT)} into {DOCS.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
