#!/usr/bin/env python3
"""Record all files installed by the release's verified UHD image downloader."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def write_manifest(directory: Path, destination: Path) -> None:
    hashes = {}
    for path in sorted(directory.rglob('*')):
        if path.is_file():
            with path.open('rb') as stream:
                hashes[path.relative_to(directory).as_posix()] = hashlib.file_digest(stream, 'sha256').hexdigest()
    if not hashes:
        raise RuntimeError(f'No UHD image files found in {directory}.')
    destination.write_text(json.dumps(hashes, sort_keys=True, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', type=Path)
    parser.add_argument('destination', type=Path)
    options = parser.parse_args()
    write_manifest(options.images, options.destination)
