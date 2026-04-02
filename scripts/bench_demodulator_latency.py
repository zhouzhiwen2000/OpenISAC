#!/usr/bin/env python3
"""Benchmark demodulator end-to-end latency across configurations.

Requires root (for isolate_cpus.bash/systemd-run) and connected TX/RX USRPs.

Usage:
    sudo python3 scripts/bench_demodulator_latency.py

The script sweeps sample_rate × fft_size × num_symbols, injects UDP traffic
into the modulator, and parses the demodulator latency reports emitted by
`process_proc`. Results are written to
measurement/demodulator_latency_bench/latency_summary.csv.
"""
from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import threading
import time
from pathlib import Path

from bench_utils import (
    apply_fft_sample_rate_sweep,
    load_yaml,
    safe_stem,
    save_yaml,
    write_csv,
)
from bench_demodulator_cpu import (
    build_isolated_cpu_spec,
    collect_unit_logs,
    launch_demodulator_with_isolation,
    prepare_isolated_cpus,
    read_unit_status,
    stop_unit,
    terminate_process_tree,
    unit_hit_coredump,
    unit_status_text,
)


def _udp_sender(ip: str, port: int, payload_size: int, stop_event: threading.Event) -> None:
    """Send UDP packets fast enough to keep the modulator producing valid frames."""
    payload = bytes(payload_size)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        while not stop_event.is_set():
            try:
                sock.sendto(payload, (ip, port))
            except OSError:
                pass
            time.sleep(0.001)
    finally:
        sock.close()


_LATENCY_BLOCK_RE = re.compile(
    r"RX frame queue wait:\s*([\d.]+)\s*ms.*?"
    r"Dequeue \+ FFT/EQ/LLR queue:\s*([\d.]+)\s*ms.*?"
    r"Bit queue \+ LDPC/UDP out:\s*([\d.]+)\s*ms.*?"
    r"TOTAL E2E(?: \(excl\. RX wait\))?:\s*([\d.]+)\s*ms",
    re.DOTALL,
)


def parse_latency_from_log(log_bytes: bytes) -> list[dict[str, float]]:
    """Return per-report latency dicts from the demodulator profiling logs."""
    text = log_bytes.decode(errors="ignore")
    results: list[dict[str, float]] = []
    for match in _LATENCY_BLOCK_RE.finditer(text):
        results.append({
            "rx_queue_ms": float(match.group(1)),
            "demod_ms": float(match.group(2)),
            "bit_ms": float(match.group(3)),
            "e2e_ms": float(match.group(4)),
        })
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark demodulator E2E latency across configurations.")
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument(
        "--mod-config",
        type=Path,
        default=Path("scripts/bench_demodulator_latency_modulator_template.yaml"),
        help="Base Modulator YAML template used for reproducible demodulator latency runs.",
    )
    parser.add_argument(
        "--demod-config",
        type=Path,
        default=Path("scripts/bench_demodulator_latency_demodulator_template.yaml"),
        help="Base Demodulator YAML template used for reproducible latency runs.",
    )
    parser.add_argument("--isolate-script", type=Path, default=Path("scripts/isolate_cpus.bash"))
    parser.add_argument("--sample-rates", default="50e6,100e6,200e6")
    parser.add_argument("--fft-sizes", default="1024")
    parser.add_argument(
        "--num-symbols-list",
        default="50,100,200",
        help="Comma-separated num_symbols values to sweep",
    )
    parser.add_argument(
        "--payload-bytes",
        type=int,
        default=1024,
        help="UDP payload size in bytes for injected traffic",
    )
    parser.add_argument(
        "--startup-gap",
        type=float,
        default=1.0,
        help="Seconds to wait after launching modulator before launching demodulator",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=30.0,
        help="Seconds to wait after launch before collecting latency samples",
    )
    parser.add_argument(
        "--collect",
        type=float,
        default=30.0,
        help="Seconds to collect latency samples after warmup",
    )
    parser.add_argument(
        "--crash-retries",
        type=int,
        default=2,
        help="How many times to retry a point automatically if the demodulator unit crashes.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("measurement/demodulator_latency_bench"))
    return parser.parse_args()


def _avg(key: str, reports: list[dict[str, float]]) -> float:
    vals = [report[key] for report in reports if key in report]
    return sum(vals) / len(vals) if vals else float("nan")


def _std(key: str, reports: list[dict[str, float]]) -> float:
    vals = [report[key] for report in reports if key in report]
    if len(vals) < 2:
        return float("nan")
    mean = sum(vals) / len(vals)
    return (sum((val - mean) ** 2 for val in vals) / len(vals)) ** 0.5


def _unit_failed(status: dict[str, str]) -> bool:
    active_state = status.get("ActiveState", "").strip()
    result = status.get("Result", "").strip()
    exec_main_code = status.get("ExecMainCode", "").strip()
    exec_main_status = status.get("ExecMainStatus", "").strip()
    return (
        unit_hit_coredump(status)
        or active_state == "failed"
        or result not in {"", "success"}
        or exec_main_code not in {"", "0", "1"}
        or exec_main_status not in {"", "0"}
    )


def main() -> None:
    args = parse_args()
    sample_rates = [float(item) for item in args.sample_rates.split(",") if item.strip()]
    fft_sizes = [int(item) for item in args.fft_sizes.split(",") if item.strip()]
    num_symbols_list = [int(item) for item in args.num_symbols_list.split(",") if item.strip()]
    base_mod_cfg = load_yaml(args.mod_config)
    base_demod_cfg = load_yaml(args.demod_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []

    for sample_rate in sample_rates:
        for fft_size in fft_sizes:
            for num_symbols in num_symbols_list:
                run_id = (
                    f"lat_sr{safe_stem(str(int(sample_rate)))}"
                    f"_fft{fft_size}_sym{num_symbols}"
                )
                run_dir = args.output_dir / run_id
                run_dir.mkdir(parents=True, exist_ok=True)

                mod_cfg = dict(base_mod_cfg)
                demod_cfg = dict(base_demod_cfg)
                apply_fft_sample_rate_sweep(mod_cfg, demod_cfg, sample_rate=sample_rate, fft_size=fft_size)
                mod_cfg["num_symbols"] = num_symbols
                mod_cfg["sensing_symbol_num"] = num_symbols
                mod_cfg["doppler_fft_size"] = num_symbols
                demod_cfg["num_symbols"] = num_symbols
                demod_cfg["sensing_symbol_num"] = num_symbols
                demod_cfg["doppler_fft_size"] = num_symbols
                mod_cfg["range_fft_size"] = fft_size
                demod_cfg["range_fft_size"] = fft_size

                save_yaml(run_dir / "Modulator.yaml", mod_cfg)
                save_yaml(run_dir / "Demodulator.yaml", demod_cfg)

                udp_port = int(mod_cfg.get("udp_input_port", 50000))
                isolated_cpu_spec = build_isolated_cpu_spec(mod_cfg, demod_cfg)
                unit_name = f"bench-demod-lat-{run_id}".replace("_", "-")
                max_attempts = max(0, args.crash_retries) + 1
                reports: list[dict[str, float]] = []
                last_status: dict[str, str] = {}

                for attempt_idx in range(1, max_attempts + 1):
                    mod_proc = None
                    demod_proc = None
                    log_since = 0.0
                    collect_since = 0.0
                    stable_log = b""
                    demod_status: dict[str, str] = {}
                    crash_detected = False

                    stop_sender = threading.Event()
                    sender_thread = threading.Thread(
                        target=_udp_sender,
                        args=("127.0.0.1", udp_port, args.payload_bytes, stop_sender),
                        daemon=True,
                    )

                    try:
                        prepare_isolated_cpus(run_dir, args.isolate_script, isolated_cpu_spec)
                        mod_proc = subprocess.Popen(
                            [str((args.build_dir / "OFDMModulator").resolve())],
                            cwd=run_dir,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid,
                        )
                        sender_thread.start()
                        time.sleep(args.startup_gap)
                        demod_proc, _demod_pid, log_since = launch_demodulator_with_isolation(
                            args.build_dir,
                            run_dir,
                            isolated_cpu_spec,
                            unit_name,
                        )

                        warmup_deadline = time.time() + args.warmup
                        while time.time() < warmup_deadline:
                            demod_status = read_unit_status(unit_name, run_dir)
                            if _unit_failed(demod_status):
                                crash_detected = True
                                break
                            time.sleep(min(1.0, max(0.1, warmup_deadline - time.time())))

                        if not crash_detected:
                            collect_since = time.time()
                            collect_deadline = collect_since + args.collect
                            while time.time() < collect_deadline:
                                demod_status = read_unit_status(unit_name, run_dir)
                                if _unit_failed(demod_status):
                                    crash_detected = True
                                    break
                                time.sleep(min(1.0, max(0.1, collect_deadline - time.time())))
                    finally:
                        stop_sender.set()
                        if sender_thread.is_alive():
                            sender_thread.join(timeout=1.0)
                        mod_log = terminate_process_tree(mod_proc) if mod_proc is not None else b""
                        if demod_proc is not None and collect_since > 0.0:
                            stable_log = collect_unit_logs(unit_name, collect_since)
                        if demod_proc is not None:
                            latest_status = read_unit_status(unit_name, run_dir)
                            if latest_status:
                                demod_status = latest_status
                        stop_unit(unit_name)
                        demod_log = collect_unit_logs(unit_name, log_since) if demod_proc is not None else b""
                        _ = terminate_process_tree(demod_proc) if demod_proc is not None else b""

                        attempt_suffix = f".attempt{attempt_idx}"
                        (run_dir / f"modulator{attempt_suffix}.log").write_bytes(mod_log)
                        (run_dir / f"demodulator{attempt_suffix}.log").write_bytes(demod_log)
                        (run_dir / f"demodulator_unit_status{attempt_suffix}.txt").write_text(
                            unit_status_text(demod_status),
                            encoding="utf-8",
                        )
                        (run_dir / "modulator.log").write_bytes(mod_log)
                        (run_dir / "demodulator.log").write_bytes(demod_log)
                        (run_dir / "demodulator_unit_status.txt").write_text(
                            unit_status_text(demod_status),
                            encoding="utf-8",
                        )

                    reports = parse_latency_from_log(stable_log)
                    last_status = demod_status
                    crash_detected = crash_detected or _unit_failed(demod_status)

                    if crash_detected:
                        if attempt_idx < max_attempts:
                            print(f"{run_id}: demodulator crashed on attempt {attempt_idx}/{max_attempts}; retrying")
                        else:
                            print(f"{run_id}: demodulator crashed on final attempt {attempt_idx}/{max_attempts}")
                        if attempt_idx < max_attempts:
                            continue
                        raise RuntimeError(
                            f"OFDMDemodulator crashed for run {run_id} after {max_attempts} attempts.\n"
                            f"{unit_status_text(last_status)}"
                        )

                    if reports:
                        break

                    if attempt_idx < max_attempts:
                        print(f"{run_id}: no valid latency reports on attempt {attempt_idx}/{max_attempts}; retrying")
                    else:
                        print(f"{run_id}: no valid latency reports on final attempt {attempt_idx}/{max_attempts}")
                    if attempt_idx < max_attempts:
                        continue
                    raise RuntimeError(
                        f"No valid latency reports were collected for run {run_id} after {max_attempts} attempts.\n"
                        f"{unit_status_text(last_status)}"
                    )

                print(
                    f"{run_id}: reports={len(reports)}"
                    f"  rx_queue={_avg('rx_queue_ms', reports):.3f}±{_std('rx_queue_ms', reports):.3f} ms"
                    f"  demod={_avg('demod_ms', reports):.3f}±{_std('demod_ms', reports):.3f} ms"
                    f"  bit={_avg('bit_ms', reports):.3f}±{_std('bit_ms', reports):.3f} ms"
                    f"  e2e={_avg('e2e_ms', reports):.3f}±{_std('e2e_ms', reports):.3f} ms"
                )

                summary_rows.append({
                    "run_id": run_id,
                    "sample_rate": sample_rate,
                    "fft_size": fft_size,
                    "num_symbols": num_symbols,
                    "payload_bytes": args.payload_bytes,
                    "lat_reports": len(reports),
                    "avg_rx_queue_ms": _avg("rx_queue_ms", reports),
                    "std_rx_queue_ms": _std("rx_queue_ms", reports),
                    "avg_demod_ms": _avg("demod_ms", reports),
                    "std_demod_ms": _std("demod_ms", reports),
                    "avg_bit_ms": _avg("bit_ms", reports),
                    "std_bit_ms": _std("bit_ms", reports),
                    "avg_e2e_ms": _avg("e2e_ms", reports),
                    "std_e2e_ms": _std("e2e_ms", reports),
                    "run_dir": str(run_dir),
                })

    write_csv(
        args.output_dir / "latency_summary.csv",
        [
            "run_id",
            "sample_rate",
            "fft_size",
            "num_symbols",
            "payload_bytes",
            "lat_reports",
            "avg_rx_queue_ms",
            "std_rx_queue_ms",
            "avg_demod_ms",
            "std_demod_ms",
            "avg_bit_ms",
            "std_bit_ms",
            "avg_e2e_ms",
            "std_e2e_ms",
            "run_dir",
        ],
        summary_rows,
    )
    print(f"\nSummary written to {args.output_dir / 'latency_summary.csv'}")


if __name__ == "__main__":
    main()
