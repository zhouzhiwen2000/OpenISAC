#!/usr/bin/env python3
"""Summarize system-delay calibration logs without editing YAML or hardware.

Run delay estimation over a stable attenuated RF loopback first. Supply logs
from one backend run. Suggested alignment includes the calibration cable path;
it is not an absolute propagation-delay measurement.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import re
import statistics
import sys

PREFIX = re.compile(r"\[(?:CUDA |HIP )?SYSDLY CH (\d+)\]")
FIELDS = re.compile(r"\b(frame_seq|seq|alignment_suggest|suggest|corr_ratio|ratio|target_alignment|target_algn)=([^,\s]+)")


def analyze(lines, *, window: int = 20, min_samples: int = 10,
            max_spread: int = 2, min_ratio: float = 10.0) -> list[dict]:
    """Use recent observations per channel; reset on restart or alignment edits."""
    if not 1 <= min_samples <= window or max_spread < 0 or not math.isfinite(min_ratio) or min_ratio <= 0:
        raise ValueError("require 1 <= min_samples <= window, max_spread >= 0, finite min_ratio > 0")
    channels = {}
    for line in lines:
        match = PREFIX.search(line)
        if match is None:
            continue
        ch = int(match[1])
        state = channels.setdefault(ch, dict(records=deque(maxlen=window), seq=None, target=None))
        fields = dict(FIELDS.findall(line[match.end():]))
        try:
            seq = int(fields.get("frame_seq", fields.get("seq")))
            target = int(fields.get("target_alignment", fields.get("target_algn")))
            suggestion = int(fields.get("alignment_suggest", fields.get("suggest")))
            ratio = float(fields.get("corr_ratio", fields.get("ratio")))
            if (seq < 0 or not math.isfinite(ratio) or ratio < 0
                    or not -(2**31) <= suggestion < 2**31):
                raise ValueError("invalid sample")
        except (ValueError, TypeError):
            # A damaged recent record must not leave a stale 'stable' result.
            state["records"].clear()
            state["seq"] = None
            continue
        if state["seq"] is not None:
            if seq == state["seq"] and target == state["target"]:
                if state["records"] and state["records"][-1] == (suggestion, ratio):
                    continue
                state["records"].clear()
            if seq < state["seq"] or target != state["target"]:
                state["records"].clear()
        state["seq"], state["target"] = seq, target
        state["records"].append((suggestion, ratio))

    results = []
    for ch, state in sorted(channels.items()):
        records = list(state["records"])
        values = [value for value, _ in records]
        spread = max(values) - min(values) if values else None
        median = statistics.median(values) if values else None
        minimum_ratio = min((ratio for _, ratio in records), default=None)
        stable = (len(records) >= min_samples and spread <= max_spread
                  and minimum_ratio >= min_ratio)
        # Match integer sample alignment: half ties away from zero.
        alignment = (int(math.copysign(math.floor(abs(median) + 0.5), median))
                     if stable else None)
        results.append(dict(channel=ch, samples=len(records), stable=stable,
                            alignment=alignment, median_samples=median,
                            spread_samples=spread, min_corr_ratio=minimum_ratio,
                            last_frame_seq=state["seq"], current_alignment=state["target"]))
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="backend log file, or - to read stdin until EOF")
    parser.add_argument("--channel", type=int, help="report only this zero-based logical channel")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--max-spread", type=int, default=2, help="maximum alignment spread in samples")
    parser.add_argument("--min-ratio", type=float, default=10.0, help="minimum peak/mean correlation ratio for every observation")
    args = parser.parse_args(argv)
    try:
        if args.log == "-":
            results = analyze(sys.stdin, window=args.window, min_samples=args.min_samples,
                              max_spread=args.max_spread, min_ratio=args.min_ratio)
        else:
            with Path(args.log).open(errors="replace") as stream:
                results = analyze(stream, window=args.window, min_samples=args.min_samples,
                                  max_spread=args.max_spread, min_ratio=args.min_ratio)
        if args.channel is not None:
            results = [r for r in results if r["channel"] == args.channel]
        print(json.dumps(results, indent=2, allow_nan=False))
        if not results or not all(r["stable"] for r in results):
            print("No stable recommendation for every selected channel; collect more data or check the RF path.", file=sys.stderr)
            return 1
        print("Stop the backend, review each alignment, update that channel's YAML and disable enable_system_delay_estimation.")
        return 0
    except (OSError, ValueError) as exc:
        print(f"Delay analysis failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
