#!/usr/bin/env python3
"""Build and run deterministic PPS device-time alignment helper tests."""

import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="openisac-pps-align-test-") as tmp:
        tmp_path = Path(tmp)
        source = repo_root / "tests" / "pps_time_alignment_test.cpp"
        binary = tmp_path / "test_pps_time_alignment"

        build_cmd = [
            "g++",
            "-std=c++17",
            "-O2",
            f"-I{repo_root / 'include'}",
            "-o",
            str(binary),
            str(source),
            "-lpthread",
        ]
        print(f"Building: {' '.join(build_cmd)}")
        result = subprocess.run(
            build_cmd, capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"BUILD FAILED:\n{result.stderr}")
            return 1

        result = subprocess.run(
            [str(binary)], capture_output=True, text=True, timeout=30
        )
        print(result.stdout)
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())
