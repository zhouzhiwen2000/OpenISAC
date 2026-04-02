#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import socket
import struct
import subprocess
import time
from pathlib import Path

from bench_utils import (
    save_yaml,
    configure_measurement,
    ensure_distinct_control_ports,
    estimate_single_frame_payload_limit,
    load_yaml,
    parse_float,
    read_csv_rows,
    safe_stem,
    terminate_process_tree,
    write_csv,
)
from bench_demodulator_cpu import (
    build_isolated_cpu_spec,
    collect_unit_logs,
    launch_demodulator_with_isolation,
    prepare_isolated_cpus,
    read_unit_status,
    stop_unit,
    unit_status_text,
)


def send_control_command(port: int, command: bytes, value: int) -> None:
    packet = struct.pack("!4s4si", b"CMD ", command, int(value))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(packet, ("127.0.0.1", port))
    finally:
        sock.close()


def wait_for_epoch(summary_path: Path, epoch_id: int, timeout_s: float) -> dict[str, str]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        rows = read_csv_rows(summary_path)
        for row in rows:
            if int(row.get("epoch_id", "-1")) == epoch_id:
                return row
        time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for epoch {epoch_id} in {summary_path}")


def wait_for_demod_sync(unit_name: str, since_epoch: float, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        log_text = collect_unit_logs(unit_name, since_epoch).decode("utf-8", errors="ignore")
        if "Sync found at pos:" in log_text or "DSync found at pos:" in log_text:
            return
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for initial demod sync in systemd unit {unit_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-only online TX-gain sweep for BER/BLER/EVM.")
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument(
        "--mod-config",
        type=Path,
        default=Path("scripts/bench_ber_bler_evm_modulator_template.yaml"),
    )
    parser.add_argument(
        "--demod-config",
        type=Path,
        default=Path("scripts/bench_ber_bler_evm_demodulator_template.yaml"),
    )
    parser.add_argument("--isolate-script", type=Path, default=Path("scripts/isolate_cpus.bash"))
    parser.add_argument("--tx-gains", type=str, required=True,
                        help="Comma-separated TX gains in dB, for example 60,55,50,45")
    parser.add_argument("--payload-bytes", type=int, default=1024)
    parser.add_argument("--packets-per-point", type=int, default=256)
    parser.add_argument("--prbs-seed", type=lambda value: int(value, 0), default=0x5A)
    parser.add_argument("--rx-gain", type=float, default=None)
    parser.add_argument("--startup-gap", type=float, default=1.0)
    parser.add_argument("--warmup", type=float, default=3.0)
    parser.add_argument("--sync-timeout", type=float, default=30.0)
    parser.add_argument("--gain-settle", type=float, default=1.5)
    parser.add_argument("--drain", type=float, default=2.0)
    parser.add_argument("--epoch-timeout", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=Path("measurement/ber_bler_evm_sweep"))
    parser.add_argument("--run-id", type=str, default="")
    return parser.parse_args()


def maybe_write_plot(summary_rows: list[dict[str, object]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    tx_gain = [float(row["tx_gain_db"]) for row in summary_rows]
    ber = [float(row["ber_decoded"]) for row in summary_rows]
    bler = [float(row["bler"]) for row in summary_rows]
    evm = [float(row["evm_rms_mean"]) for row in summary_rows]
    snr = [float(row["estimated_snr_db_mean"]) for row in summary_rows]

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    axes[0].semilogy(tx_gain, ber, marker="o")
    axes[0].semilogy(tx_gain, bler, marker="s")
    axes[0].set_ylabel("BER / BLER")
    axes[0].grid(True)
    axes[0].legend(["BER", "BLER"])

    axes[1].plot(tx_gain, evm, marker="o")
    axes[1].set_ylabel("EVM RMS")
    axes[1].grid(True)

    axes[2].plot(tx_gain, snr, marker="o")
    axes[2].set_ylabel("Estimated SNR (dB)")
    axes[2].set_xlabel("TX Gain (dB)")
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig(output_dir / "sweep_summary.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    tx_gains = [float(item) for item in args.tx_gains.split(",") if item.strip()]
    if not tx_gains:
        raise RuntimeError("At least one TX gain is required.")

    run_id = args.run_id.strip() or dt.datetime.now().strftime("ber_bler_evm_%Y%m%d_%H%M%S")
    run_id = safe_stem(run_id)
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_dir = run_dir.resolve()

    mod_cfg = load_yaml(args.mod_config)
    demod_cfg = load_yaml(args.demod_config)
    ensure_distinct_control_ports(mod_cfg, demod_cfg)
    configure_measurement(mod_cfg, run_id, run_dir, args.payload_bytes, args.prbs_seed, args.packets_per_point)
    configure_measurement(demod_cfg, run_id, run_dir, args.payload_bytes, args.prbs_seed, args.packets_per_point)
    demod_cfg["rx_agc_enable"] = False
    if args.rx_gain is not None:
        demod_cfg["rx_gain"] = float(args.rx_gain)
    isolated_cpu_spec = build_isolated_cpu_spec(mod_cfg, demod_cfg)

    max_payload = estimate_single_frame_payload_limit(mod_cfg)
    if args.payload_bytes > max_payload:
        raise RuntimeError(
            f"payload_bytes={args.payload_bytes} exceeds single-frame measurement limit {max_payload}"
        )
    if os.geteuid() != 0:
        raise RuntimeError("bench_ber_bler_evm.py must run as root to use isolate_cpus.bash")

    mod_cfg["tx_gain"] = float(tx_gains[0])
    mod_summary_path = run_dir / "modulator_measurement_summary.csv"
    demod_summary_path = run_dir / "demodulator_measurement_summary.csv"
    demod_status_path = run_dir / "demodulator_unit_status.txt"
    for stale_path in (
        mod_summary_path,
        demod_summary_path,
        demod_status_path,
        run_dir / "sweep_summary.csv",
        run_dir / "modulator.log",
        run_dir / "demodulator.log",
        run_dir / "sweep_summary.png",
    ):
        if stale_path.exists():
            stale_path.unlink()

    mod_proc = None
    demod_proc = None
    demod_log_since = 0.0
    demod_unit_name = f"bench-ber-bler-evm-{run_id}".replace("_", "-")
    try:
        save_yaml(run_dir / "Modulator.yaml", mod_cfg)
        save_yaml(run_dir / "Demodulator.yaml", demod_cfg)
        prepare_isolated_cpus(run_dir, args.isolate_script, isolated_cpu_spec)
        mod_proc = subprocess.Popen(
            [str((args.build_dir / "OFDMModulator").resolve())],
            cwd=run_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        time.sleep(args.startup_gap)
        demod_proc, _demod_pid, demod_log_since = launch_demodulator_with_isolation(
            args.build_dir,
            run_dir,
            isolated_cpu_spec,
            demod_unit_name,
        )
        time.sleep(args.warmup)
        wait_for_demod_sync(demod_unit_name, demod_log_since, args.sync_timeout)

        mod_control_port = int(mod_cfg.get("control_port", 9999))
        demod_control_port = int(demod_cfg.get("control_port", 10000))
        for epoch_id, tx_gain_db in enumerate(tx_gains, start=1):
            send_control_command(mod_control_port, b"TXGN", int(round(tx_gain_db * 10.0)))
            time.sleep(args.gain_settle)
            send_control_command(demod_control_port, b"MRST", epoch_id)
            send_control_command(mod_control_port, b"MRST", epoch_id)
            wait_for_epoch(mod_summary_path, epoch_id, args.epoch_timeout)
            time.sleep(args.drain)
        if tx_gains:
            time.sleep(args.gain_settle)
            send_control_command(demod_control_port, b"MRST", len(tx_gains) + 1)
            wait_for_epoch(demod_summary_path, len(tx_gains), args.epoch_timeout)
    finally:
        mod_log = terminate_process_tree(mod_proc) if mod_proc is not None else b""
        demod_status = read_unit_status(demod_unit_name, run_dir) if demod_proc is not None else {}
        stop_unit(demod_unit_name)
        demod_log = collect_unit_logs(demod_unit_name, demod_log_since) if demod_proc is not None else b""
        _ = terminate_process_tree(demod_proc) if demod_proc is not None else b""
        (run_dir / "modulator.log").write_bytes(mod_log)
        (run_dir / "demodulator.log").write_bytes(demod_log)
        demod_status_path.write_text(unit_status_text(demod_status), encoding="utf-8")

    mod_rows = {int(row["epoch_id"]): row for row in read_csv_rows(mod_summary_path)}
    demod_csv_rows = read_csv_rows(demod_summary_path)
    demod_rows = {int(row["epoch_id"]): row for row in demod_csv_rows}
    demod_summary_updated = False

    for epoch_id, tx_gain_db in enumerate(tx_gains, start=1):
        demod_row = demod_rows.get(epoch_id)
        if demod_row is None:
            continue
        demod_tx_gain_db = parse_float(demod_row.get("tx_gain_db"))
        if not math.isnan(demod_tx_gain_db):
            continue
        mod_row = mod_rows.get(epoch_id, {})
        fallback_tx_gain_db = parse_float(mod_row.get("tx_gain_db"), tx_gain_db)
        demod_row["tx_gain_db"] = f"{fallback_tx_gain_db:.6f}"
        demod_summary_updated = True

    if demod_summary_updated and demod_csv_rows:
        write_csv(demod_summary_path, list(demod_csv_rows[0].keys()), demod_csv_rows)

    summary_rows: list[dict[str, object]] = []
    for epoch_id, tx_gain_db in enumerate(tx_gains, start=1):
        mod_row = mod_rows.get(epoch_id, {})
        demod_row = demod_rows.get(epoch_id, {})
        summary_rows.append(
            {
                "run_id": run_id,
                "epoch_id": epoch_id,
                "tx_gain_db": float(mod_row.get("tx_gain_db", tx_gain_db)),
                "packets_sent": int(mod_row.get("packets_sent", args.packets_per_point)),
                "packets_expected": int(demod_row.get("packets_expected", args.packets_per_point)),
                "packets_successful": int(demod_row.get("packets_successful", 0)),
                "packets_failed": int(demod_row.get("packets_failed", args.packets_per_point)),
                "compared_bits": int(demod_row.get("compared_bits", 0)),
                "bit_errors": int(demod_row.get("bit_errors", 0)),
                "ber_decoded": float(demod_row.get("ber_decoded", float("nan"))),
                "bler": float(demod_row.get("bler", float("nan"))),
                "frame_count": int(demod_row.get("frame_count", 0)),
                "estimated_snr_db_mean": float(demod_row.get("estimated_snr_db_mean", float("nan"))),
                "evm_rms_mean": float(demod_row.get("evm_rms_mean", float("nan"))),
                "evm_db_mean": float(demod_row.get("evm_db_mean", float("nan"))),
            }
        )

    write_csv(
        run_dir / "sweep_summary.csv",
        [
            "run_id",
            "epoch_id",
            "tx_gain_db",
            "packets_sent",
            "packets_expected",
            "packets_successful",
            "packets_failed",
            "compared_bits",
            "bit_errors",
            "ber_decoded",
            "bler",
            "frame_count",
            "estimated_snr_db_mean",
            "evm_rms_mean",
            "evm_db_mean",
        ],
        summary_rows,
    )
    maybe_write_plot(summary_rows, run_dir)
    print(run_dir)


if __name__ == "__main__":
    main()
