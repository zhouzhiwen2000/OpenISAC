#!/usr/bin/env python3
"""Check the effective container CPU mask, FIFO scheduling and locked memory."""
from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
from pathlib import Path
import resource
import sys

import yaml


def configured_cpus(value: object, path: str = "") -> dict[str, int]:
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{path}.{key}" if path else str(key)
            if str(key).endswith(("cpu_core", "cpu_cores")) and key != "cpu_cores":
                items = item if isinstance(item, list) else [item]
                for index, cpu in enumerate(items):
                    if type(cpu) is not int or cpu < -1:
                        raise ValueError(f"{name}: expected integer CPU IDs or -1")
                    if cpu >= 0:
                        result[f"{name}[{index}]"] = cpu
            else:
                result.update(configured_cpus(item, name))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.update(configured_cpus(item, f"{path}[{index}]"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path)
    args = parser.parse_args()
    allowed = sorted(os.sched_getaffinity(0))
    assignments = {}
    if args.config:
        data = yaml.safe_load(args.config.read_text())
        if not isinstance(data, dict):
            raise ValueError("YAML must contain a configuration mapping")
        assignments = configured_cpus(data)
    invalid = {name: cpu for name, cpu in assignments.items() if cpu not in allowed}
    report = {"allowed_cpus": allowed, "configured_cpus": assignments,
              "invalid_cpus": invalid, "rtprio_limit": resource.getrlimit(resource.RLIMIT_RTPRIO)[0],
              "memlock_limit": resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]}
    policy, param = os.sched_getscheduler(0), os.sched_getparam(0)
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(99))
        report["fifo_priority_99"] = True
    except OSError as exc:
        report["fifo_priority_99"] = False
        report["fifo_error"] = str(exc)
    finally:
        os.sched_setscheduler(0, policy, param)
    libc = ctypes.CDLL(None, use_errno=True)
    with mmap.mmap(-1, mmap.PAGESIZE) as page:
        address = ctypes.addressof(ctypes.c_char.from_buffer(page))
        locked = libc.mlock(ctypes.c_void_p(address), ctypes.c_size_t(mmap.PAGESIZE)) == 0
        report["mlock_page"] = locked
        if locked:
            libc.munlock(ctypes.c_void_p(address), ctypes.c_size_t(mmap.PAGESIZE))
    rt_ready = report["fifo_priority_99"] and report["mlock_page"] and report["memlock_limit"] == resource.RLIM_INFINITY
    report["realtime_ready"] = bool(rt_ready)
    print("OpenISAC preflight: " + json.dumps(report, sort_keys=True), flush=True)
    if invalid:
        print("Configured CPU IDs are outside the container CPU set. Adjust YAML or --cpuset-cpus.", file=sys.stderr)
        return 2
    if not rt_ready:
        print("Real-time prerequisites unavailable: use --cap-add SYS_NICE --cap-add IPC_LOCK "
              "--ulimit rtprio=99 --ulimit memlock=-1 and a host that permits FIFO scheduling.", file=sys.stderr)
        if os.environ.get("OPENISAC_REQUIRE_RT", "1") == "1":
            return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"OpenISAC preflight: {exc}", file=sys.stderr)
        sys.exit(2)
