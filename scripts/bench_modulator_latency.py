#!/usr/bin/env python3
"""Benchmark end-to-end modulator latency (UDP ingest → TX send) across configurations.

Requires root (for isolate_cpus.bash) and a connected USRP.

Usage:
    sudo python3 scripts/bench_modulator_latency.py --mod-config scripts/bench_modulator_latency_template.yaml

The script sweeps fft_size × num_symbols × sample_rate, injects UDP traffic into
the modulator, and parses the "[Latency]" log lines emitted by _tx_proc.
Results are written to measurement/modulator_latency_bench/latency_summary.csv.
"""
from __future__ import annotations

import argparse
import re
import socket
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
from bench_modulator_cpu import (
    build_isolated_cpu_spec,
    collect_unit_logs,
    launch_modulator_with_isolation,
    stop_unit,
    terminate_process_tree,
)


# ---------------------------------------------------------------------------
# UDP traffic injector
# ---------------------------------------------------------------------------

def _udp_sender(ip: str, port: int, payload_size: int, stop_event: threading.Event) -> None:
    """Send UDP packets as fast as possible until stop_event is set."""
    payload = bytes(payload_size)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        while not stop_event.is_set():
            try:
                sock.sendto(payload, (ip, port))
            except OSError:
                pass
            time.sleep(0.001)  # ~1 kpps — enough to keep the modulator fed
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# Log parser
# ---------------------------------------------------------------------------

_LATENCY_BLOCK_RE = re.compile(
    r"LDPC encode \+ ingest queue:\s*([\d.]+)\s*ms.*?"
    r"Dequeue \+ IFFT/CP \+ mod queue:\s*([\d.]+)\s*ms.*?"
    r"TX circular buffer wait:\s*([\d.]+)\s*ms.*?"
    r"TOTAL E2E \(excl\. TX wait\):\s*([\d.]+)\s*ms",
    re.DOTALL,
)


def parse_latency_from_log(log_bytes: bytes) -> list[dict[str, float]]:
    """Return per-report latency dicts from either old or merged profiling logs."""
    text = log_bytes.decode(errors="ignore")
    results: list[dict[str, float]] = []
    for match in _LATENCY_BLOCK_RE.finditer(text):
        results.append({
            "ldpc_ms": float(match.group(1)),
            "mod_ms": float(match.group(2)),
            "tx_wait_ms": float(match.group(3)),
            "e2e_ms": float(match.group(4)),
        })
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark modulator E2E latency across configurations.")
    p.add_argument("--build-dir", type=Path, default=Path("build"))
    p.add_argument(
        "--mod-config",
        type=Path,
        default=Path("scripts/bench_modulator_latency_template.yaml"),
        help="Base Modulator YAML template used for reproducible latency benchmark runs.",
    )
    p.add_argument("--isolate-script", type=Path, default=Path("scripts/isolate_cpus.bash"))
    p.add_argument("--sample-rates", default="50e6,100e6,200e6")
    p.add_argument("--fft-sizes", default="1024")
    p.add_argument("--num-symbols-list", default="50,100,200",
                   help="Comma-separated num_symbols values to sweep")
    p.add_argument("--payload-bytes", type=int, default=1024,
                   help="UDP payload size in bytes for injected traffic")
    p.add_argument("--warmup", type=float, default=30.0,
                   help="Seconds to wait after launch before collecting latency samples")
    p.add_argument("--collect", type=float, default=30.0,
                   help="Seconds to collect latency samples after warmup")
    p.add_argument("--output-dir", type=Path, default=Path("measurement/modulator_latency_bench"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _find_base_config() -> dict:
    candidates = [
        Path("scripts/bench_modulator_latency_template.yaml"),
        Path("build/Modulator.yaml"),
        Path("config/Modulator_X310.yaml"),
        Path("config/Modulator_B210.yaml"),
    ]
    for p in candidates:
        if p.exists():
            print(f"Using base config: {p}")
            return load_yaml(p)
    raise FileNotFoundError(
        "No Modulator YAML found. Pass --mod-config explicitly."
    )


def main() -> None:
    args = parse_args()
    sample_rates = [float(x) for x in args.sample_rates.split(",") if x.strip()]
    fft_sizes = [int(x) for x in args.fft_sizes.split(",") if x.strip()]
    num_symbols_list = [int(x) for x in args.num_symbols_list.split(",") if x.strip()]
    base_cfg = load_yaml(args.mod_config) if args.mod_config else _find_base_config()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    for sample_rate in sample_rates:
        for fft_size in fft_sizes:
            for num_symbols in num_symbols_list:
                run_id = (
                    f"lat_sr{safe_stem(str(int(sample_rate)))}"
                    f"_fft{fft_size}_sym{num_symbols}"
                )
                run_dir = args.output_dir / run_id
                run_dir.mkdir(parents=True, exist_ok=True)

                cfg = dict(base_cfg)
                apply_fft_sample_rate_sweep(cfg, cfg, sample_rate=sample_rate, fft_size=fft_size)
                cfg["num_symbols"] = num_symbols
                save_yaml(run_dir / "Modulator.yaml", cfg)

                udp_port = int(cfg.get("udp_input_port", 50000))
                isolated_cpu_spec = build_isolated_cpu_spec(cfg)
                unit_name = f"bench-lat-{run_id}".replace("_", "-")

                mod_proc = None
                log_since = 0.0
                reports: list[dict[str, float]] = []

                stop_sender = threading.Event()
                sender_thread = threading.Thread(
                    target=_udp_sender,
                    args=("127.0.0.1", udp_port, args.payload_bytes, stop_sender),
                    daemon=True,
                )

                try:
                    mod_proc, _pid, log_since = launch_modulator_with_isolation(
                        args.build_dir, run_dir, args.isolate_script,
                        isolated_cpu_spec, unit_name,
                    )
                    sender_thread.start()
                    time.sleep(args.warmup)
                    collect_since = time.time()  # only parse logs after warmup
                    time.sleep(args.collect)
                finally:
                    stop_sender.set()
                    stop_unit(unit_name)
                    log_bytes = collect_unit_logs(unit_name, log_since) if mod_proc else b""
                    if mod_proc:
                        terminate_process_tree(mod_proc)
                    (run_dir / "modulator.log").write_bytes(log_bytes)

                # Re-fetch logs from collect_since to exclude warmup period
                stable_log = collect_unit_logs(unit_name, collect_since) if mod_proc else b""
                reports = parse_latency_from_log(stable_log)

                def _avg(key: str, rpts: list[dict[str, float]] = reports) -> float:
                    vals = [r[key] for r in rpts if key in r]
                    return sum(vals) / len(vals) if vals else float("nan")

                def _std(key: str, rpts: list[dict[str, float]] = reports) -> float:
                    vals = [r[key] for r in rpts if key in r]
                    if len(vals) < 2:
                        return float("nan")
                    mean = sum(vals) / len(vals)
                    return (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5

                print(
                    f"{run_id}: reports={len(reports)}"
                    f"  ldpc={_avg('ldpc_ms'):.3f}±{_std('ldpc_ms'):.3f} ms"
                    f"  mod={_avg('mod_ms'):.3f}±{_std('mod_ms'):.3f} ms"
                    f"  tx_wait={_avg('tx_wait_ms'):.3f}±{_std('tx_wait_ms'):.3f} ms"
                    f"  e2e={_avg('e2e_ms'):.3f}±{_std('e2e_ms'):.3f} ms"
                )

                summary_rows.append({
                    "run_id": run_id,
                    "sample_rate": sample_rate,
                    "fft_size": fft_size,
                    "num_symbols": num_symbols,
                    "payload_bytes": args.payload_bytes,
                    "lat_reports": len(reports),
                    "avg_ldpc_ms": _avg("ldpc_ms"),
                    "std_ldpc_ms": _std("ldpc_ms"),
                    "avg_mod_ms": _avg("mod_ms"),
                    "std_mod_ms": _std("mod_ms"),
                    "avg_tx_wait_ms": _avg("tx_wait_ms"),
                    "std_tx_wait_ms": _std("tx_wait_ms"),
                    "avg_e2e_ms": _avg("e2e_ms"),
                    "std_e2e_ms": _std("e2e_ms"),
                    "run_dir": str(run_dir),
                })

    write_csv(
        args.output_dir / "latency_summary.csv",
        ["run_id", "sample_rate", "fft_size", "num_symbols", "payload_bytes",
         "lat_reports",
         "avg_ldpc_ms", "std_ldpc_ms",
         "avg_mod_ms",  "std_mod_ms",
         "avg_tx_wait_ms", "std_tx_wait_ms",
         "avg_e2e_ms",  "std_e2e_ms",
         "run_dir"],
        summary_rows,
    )
    print(f"\nSummary written to {args.output_dir / 'latency_summary.csv'}")


if __name__ == "__main__":
    main()
