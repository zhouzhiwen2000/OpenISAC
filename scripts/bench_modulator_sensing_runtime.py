#!/usr/bin/env python3
"""Benchmark monostatic sensing on OFDMModulator with runtime STRD/SKIP/MTI control."""
from __future__ import annotations

import argparse
import re
import socket
import struct
import threading
import time
from pathlib import Path

from bench_modulator_cpu import (
    build_isolated_cpu_spec,
    build_mod_role_map,
    collect_unit_logs,
    launch_modulator_with_isolation,
    stop_unit,
)
from bench_utils import (
    load_yaml,
    mean_of,
    safe_stem,
    sample_cpu_usage_with_threads,
    save_yaml,
    summarize_thread_rows,
    terminate_process_tree,
    write_csv,
)


def _udp_sender(ip: str, port: int, payload_size: int, stop_event: threading.Event) -> None:
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


def _parse_bool_tokens(raw: str) -> list[bool]:
    values: list[bool] = []
    for token in raw.split(","):
        value = token.strip().lower()
        if not value:
            continue
        if value in {"1", "on", "true", "yes"}:
            values.append(True)
        elif value in {"0", "off", "false", "no"}:
            values.append(False)
        else:
            raise ValueError(f"Invalid boolean token: {token}")
    return values


def _send_control_command(
    ip: str,
    port: int,
    command: bytes,
    value: int,
) -> None:
    packet = struct.pack("!4s4si", b"CMD ", command, int(value))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(packet, (ip, port))
    finally:
        sock.close()


def _send_control_triplet(
    ip: str,
    port: int,
    *,
    stride: int,
    skip: bool,
    mti: bool,
) -> None:
    _send_control_command(
        ip,
        port,
        b"STRD",
        stride,
    )
    _send_control_command(
        ip,
        port,
        b"SKIP",
        1 if skip else 0,
    )
    _send_control_command(
        ip,
        port,
        b"MTI ",
        1 if mti else 0,
    )


_APPLIED_RE = re.compile(
    r"\[Sensing CH (\d+)\] applied shared params.*?"
    r"\(stride=(\d+), MTI=(\d+), SKIP=(\d+)\)"
)

_PROFILE_RE = re.compile(
    r"========== Sensing CH (\d+) Profiling \(avg per batch, us\) ==========\s*"
    r"Batch gather:\s*([\d.]+)\s*us\s*"
    r"RX symbol prep:\s*([\d.]+)\s*us\s*"
    r"ChEst \+ Shift:\s*([\d.]+)\s*us\s*"
    r"MTI:\s*([\d.]+)\s*us\s*"
    r"Windows\+IFFT\+DopFFT:\s*([\d.]+)\s*us\s*"
    r"Send queue push:\s*([\d.]+)\s*us\s*"
    r"TOTAL LATENCY \(excl\. gather/send\):\s*([\d.]+)\s*us\s*"
    r"Profile batch count:\s*(\d+)",
    re.DOTALL,
)

_DROP_PATTERNS = {
    "paired_rx_queue_full": re.compile(r"paired RX queue full, dropping newest RX frame"),
    "paired_tx_queue_full": re.compile(r"paired TX queue full, dropping newest TX sensing frame"),
    "drop_rx_seq_mismatch": re.compile(r"drop RX frame due to seq mismatch"),
    "drop_tx_seq_mismatch": re.compile(r"drop TX frame due to seq mismatch"),
}


def _wait_for_mode_apply(
    unit_name: str,
    since_epoch: float,
    *,
    stride: int,
    skip: bool,
    mti: bool,
    timeout_s: float,
    channel_id: int = 0,
) -> tuple[float, bytes]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        log_bytes = collect_unit_logs(unit_name, since_epoch)
        text = log_bytes.decode(errors="ignore")
        for match in _APPLIED_RE.finditer(text):
            if (
                int(match.group(1)) == channel_id and
                int(match.group(2)) == stride and
                int(match.group(3)) == int(mti) and
                int(match.group(4)) == int(skip)
            ):
                return time.time(), log_bytes
        time.sleep(0.2)
    raise TimeoutError(
        f"Timed out waiting for sensing params apply: CH{channel_id} stride={stride} skip={int(skip)} mti={int(mti)}"
    )


def _received_control_values(log_bytes: bytes) -> set[tuple[str, int]]:
    text = log_bytes.decode(errors="ignore")
    received: set[tuple[str, int]] = set()
    if "Received STRD command: 1" in text:
        received.add(("STRD", 1))
    strd_matches = re.findall(r"Received STRD command:\s*(\d+)", text)
    for raw in strd_matches:
        received.add(("STRD", int(raw)))
    skip_matches = re.findall(r"Received SKIP command:\s*(\d+)", text)
    for raw in skip_matches:
        received.add(("SKIP", int(raw)))
    if "Received MTI command: Enable" in text:
        received.add(("MTI ", 1))
    if "Received MTI command: Disable" in text:
        received.add(("MTI ", 0))
    return received


def _parse_profile_reports(log_bytes: bytes, channel_id: int = 0) -> list[dict[str, float]]:
    text = log_bytes.decode(errors="ignore")
    reports: list[dict[str, float]] = []
    for match in _PROFILE_RE.finditer(text):
        if int(match.group(1)) != channel_id:
            continue
        reports.append({
            "gather_us": float(match.group(2)),
            "prep_us": float(match.group(3)),
            "chest_shift_us": float(match.group(4)),
            "mti_us": float(match.group(5)),
            "fft_us": float(match.group(6)),
            "send_us": float(match.group(7)),
            "total_us": float(match.group(8)),
            "profile_batch_count": float(match.group(9)),
        })
    return reports


def _count_drop_events(log_bytes: bytes) -> dict[str, int]:
    text = log_bytes.decode(errors="ignore")
    counts: dict[str, int] = {}
    for key, pattern in _DROP_PATTERNS.items():
        counts[key] = len(pattern.findall(text))
    counts["drop_events_total"] = sum(counts.values())
    return counts


def _find_role_cpu(role_rows: list[dict[str, object]], role_name: str) -> tuple[float, float]:
    for row in role_rows:
        if str(row.get("role", "")).strip() == role_name:
            return float(row["avg_cpu_pct"]), float(row["peak_cpu_pct"])
    return float("nan"), float("nan")


def _avg(reports: list[dict[str, float]], key: str) -> float:
    vals = [report[key] for report in reports if key in report]
    return sum(vals) / len(vals) if vals else float("nan")


def _std(reports: list[dict[str, float]], key: str) -> float:
    vals = [report[key] for report in reports if key in report]
    if len(vals) < 2:
        return float("nan")
    mean = sum(vals) / len(vals)
    return (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark modulator monostatic sensing with runtime control changes.")
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument(
        "--mod-config",
        type=Path,
        default=Path("scripts/bench_modulator_sensing_runtime_template.yaml"),
        help="Base Modulator YAML template used for the sensing runtime benchmark.",
    )
    parser.add_argument("--isolate-script", type=Path, default=Path("scripts/isolate_cpus.bash"))
    parser.add_argument("--sample-rate", type=float, default=100e6)
    parser.add_argument("--fft-size", type=int, default=1024)
    parser.add_argument("--strides", default="1,2,5,10,20")
    parser.add_argument("--skip-values", default="on,off")
    parser.add_argument("--mti-values", default="on,off")
    parser.add_argument("--payload-bytes", type=int, default=1024)
    parser.add_argument("--warmup", type=float, default=10.0)
    parser.add_argument("--settle", type=float, default=3.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--apply-timeout", type=float, default=15.0)
    parser.add_argument(
        "--control-retries",
        type=int,
        default=3,
        help="Retry count for the whole STRD/SKIP/MTI command triplet.",
    )
    parser.add_argument(
        "--control-retry-delay",
        type=float,
        default=0.01,
        help="Delay in seconds between command-triplet retries.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("measurement/modulator_sensing_runtime_bench"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mod_cfg = load_yaml(args.mod_config)
    mod_cfg["sample_rate"] = float(args.sample_rate)
    mod_cfg["bandwidth"] = float(args.sample_rate)
    mod_cfg["fft_size"] = int(args.fft_size)
    mod_cfg["range_fft_size"] = int(args.fft_size)
    mod_cfg["profiling_modules"] = "sensing,latency"
    mod_cfg["mono_sensing_output_enabled"] = False
    if isinstance(mod_cfg.get("sensing_rx_channels"), list):
        for ch in mod_cfg["sensing_rx_channels"]:
            if isinstance(ch, dict):
                ch["enable_sensing_output"] = False

    strides = [int(item) for item in args.strides.split(",") if item.strip()]
    skip_values = _parse_bool_tokens(args.skip_values)
    mti_values = _parse_bool_tokens(args.mti_values)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"mono_sensing_sr{safe_stem(str(int(args.sample_rate)))}_fft{args.fft_size}"
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(run_dir / "Modulator.yaml", mod_cfg)

    isolated_cpu_spec = build_isolated_cpu_spec(mod_cfg)
    udp_port = int(mod_cfg.get("udp_input_port", 50000))
    control_port = int(mod_cfg.get("control_port", 9999))
    unit_name = f"bench-mod-sense-{run_id}".replace("_", "-")

    mod_proc = None
    mod_pid = 0
    log_since = 0.0
    stop_sender = threading.Event()
    sender_thread = threading.Thread(
        target=_udp_sender,
        args=("127.0.0.1", udp_port, args.payload_bytes, stop_sender),
        daemon=True,
    )

    summary_rows: list[dict[str, object]] = []

    try:
        mod_proc, mod_pid, log_since = launch_modulator_with_isolation(
            args.build_dir,
            run_dir,
            args.isolate_script,
            isolated_cpu_spec,
            unit_name,
        )
        sender_thread.start()
        time.sleep(args.warmup)

        role_map = build_mod_role_map(mod_cfg, mod_pid)

        for stride in strides:
            for skip in skip_values:
                for mti in mti_values:
                    case_id = f"strd{stride}_skip{int(skip)}_mti{int(mti)}"
                    control_since = time.time()
                    expected_control_receives = {
                        ("STRD", stride),
                        ("SKIP", 1 if skip else 0),
                        ("MTI ", 1 if mti else 0),
                    }
                    total_attempts = max(1, int(args.control_retries))
                    for attempt in range(1, total_attempts + 1):
                        _send_control_triplet(
                            "127.0.0.1",
                            control_port,
                            stride=stride,
                            skip=skip,
                            mti=mti,
                        )
                        if attempt >= total_attempts:
                            break
                        if args.control_retry_delay > 0.0:
                            time.sleep(args.control_retry_delay)
                        preapply_logs = collect_unit_logs(unit_name, control_since)
                        received_values = _received_control_values(preapply_logs)
                        if expected_control_receives.issubset(received_values):
                            break
                    applied_since, apply_logs = _wait_for_mode_apply(
                        unit_name,
                        control_since,
                        stride=stride,
                        skip=skip,
                        mti=mti,
                        timeout_s=args.apply_timeout,
                    )
                    time.sleep(args.settle)

                    cpu_rows, thread_rows = sample_cpu_usage_with_threads(
                        [mod_pid],
                        args.duration,
                        args.interval,
                    )
                    sample_logs = collect_unit_logs(unit_name, applied_since)
                    reports = _parse_profile_reports(sample_logs)
                    drop_counts = _count_drop_events(sample_logs)
                    role_rows = summarize_thread_rows(thread_rows, role_map)

                    for row in thread_rows:
                        row["run_id"] = run_id
                        row["case_id"] = case_id
                        row["role"] = role_map.get((int(row["pid"]), str(row["allowed_cpus"])), "")
                    for row in role_rows:
                        row["run_id"] = run_id
                        row["case_id"] = case_id

                    thread_csv = run_dir / f"{case_id}_thread_cpu_summary.csv"
                    role_csv = run_dir / f"{case_id}_thread_role_cpu_summary.csv"
                    apply_log = run_dir / f"{case_id}_apply.log"
                    sample_log = run_dir / f"{case_id}_sample.log"

                    write_csv(
                        thread_csv,
                        ["run_id", "case_id", "pid", "tid", "comm", "allowed_cpus", "last_processor", "role", "avg_cpu_pct", "peak_cpu_pct", "samples"],
                        thread_rows,
                    )
                    write_csv(
                        role_csv,
                        ["run_id", "case_id", "pid", "allowed_cpus", "role", "avg_cpu_pct", "peak_cpu_pct", "thread_count", "threads"],
                        role_rows,
                    )
                    apply_log.write_bytes(apply_logs)
                    sample_log.write_bytes(sample_logs)

                    sensing_avg_cpu, sensing_peak_cpu = _find_role_cpu(
                        role_rows,
                        "mod:sensing_process_loop_ch0",
                    )

                    summary_rows.append({
                        "run_id": run_id,
                        "case_id": case_id,
                        "sample_rate": args.sample_rate,
                        "fft_size": args.fft_size,
                        "strd": stride,
                        "skip": int(skip),
                        "mti": int(mti),
                        "profile_reports": len(reports),
                        "sensing_thread_avg_cpu_pct": sensing_avg_cpu,
                        "sensing_thread_peak_cpu_pct": sensing_peak_cpu,
                        "avg_gather_us": _avg(reports, "gather_us"),
                        "std_gather_us": _std(reports, "gather_us"),
                        "avg_prep_us": _avg(reports, "prep_us"),
                        "std_prep_us": _std(reports, "prep_us"),
                        "avg_chest_shift_us": _avg(reports, "chest_shift_us"),
                        "std_chest_shift_us": _std(reports, "chest_shift_us"),
                        "avg_mti_us": _avg(reports, "mti_us"),
                        "std_mti_us": _std(reports, "mti_us"),
                        "avg_fft_us": _avg(reports, "fft_us"),
                        "std_fft_us": _std(reports, "fft_us"),
                        "avg_send_us": _avg(reports, "send_us"),
                        "std_send_us": _std(reports, "send_us"),
                        "avg_total_us": _avg(reports, "total_us"),
                        "std_total_us": _std(reports, "total_us"),
                        "avg_profile_batch_count": _avg(reports, "profile_batch_count"),
                        "paired_rx_queue_full": drop_counts["paired_rx_queue_full"],
                        "paired_tx_queue_full": drop_counts["paired_tx_queue_full"],
                        "drop_rx_seq_mismatch": drop_counts["drop_rx_seq_mismatch"],
                        "drop_tx_seq_mismatch": drop_counts["drop_tx_seq_mismatch"],
                        "drop_events_total": drop_counts["drop_events_total"],
                        "thread_cpu_summary": str(thread_csv),
                        "thread_role_cpu_summary": str(role_csv),
                        "apply_log": str(apply_log),
                        "sample_log": str(sample_log),
                    })

                    print(
                        f"{case_id}: reports={len(reports)}"
                        f" sensing_cpu={sensing_avg_cpu:.2f}%"
                        f" total={_avg(reports, 'total_us'):.2f}±{_std(reports, 'total_us'):.2f} us"
                        f" drops={drop_counts['drop_events_total']}"
                    )
    finally:
        stop_sender.set()
        if sender_thread.is_alive():
            sender_thread.join(timeout=1.0)
        stop_unit(unit_name)
        mod_log = collect_unit_logs(unit_name, log_since) if mod_proc is not None else b""
        _ = terminate_process_tree(mod_proc) if mod_proc is not None else b""
        (run_dir / "modulator.log").write_bytes(mod_log)

    base_fields = [
        "run_id",
        "case_id",
        "sample_rate",
        "fft_size",
        "strd",
        "skip",
        "mti",
        "profile_reports",
        "sensing_thread_avg_cpu_pct",
        "sensing_thread_peak_cpu_pct",
        "avg_gather_us",
        "std_gather_us",
        "avg_prep_us",
        "std_prep_us",
        "avg_chest_shift_us",
        "std_chest_shift_us",
        "avg_mti_us",
        "std_mti_us",
        "avg_fft_us",
        "std_fft_us",
        "avg_send_us",
        "std_send_us",
        "avg_total_us",
        "std_total_us",
        "avg_profile_batch_count",
        "paired_rx_queue_full",
        "paired_tx_queue_full",
        "drop_rx_seq_mismatch",
        "drop_tx_seq_mismatch",
        "drop_events_total",
        "thread_cpu_summary",
        "thread_role_cpu_summary",
        "apply_log",
        "sample_log",
    ]
    write_csv(
        run_dir / "modulator_sensing_runtime_summary.csv",
        base_fields,
        summary_rows,
    )
    print(f"\nSummary written to {run_dir / 'modulator_sensing_runtime_summary.csv'}")


if __name__ == "__main__":
    main()
