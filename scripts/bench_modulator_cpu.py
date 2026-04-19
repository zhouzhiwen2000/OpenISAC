from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import time
from pathlib import Path

from bench_utils import (
    apply_fft_sample_rate_sweep,
    load_yaml,
    mean_of,
    peak_of,
    safe_stem,
    sample_cpu_usage_with_threads,
    summarize_thread_rows,
    terminate_process_tree,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CPU load for CPU OFDMModulator only.")
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument(
        "--mod-config",
        type=Path,
        default=Path("scripts/bench_modulator_cpu_template.yaml"),
        help="Base Modulator YAML template used for reproducible CPU benchmark runs.",
    )
    parser.add_argument("--isolate-script", type=Path, default=Path("scripts/isolate_cpus.bash"))
    parser.add_argument("--sample-rates", type=str, default="50e6,100e6,200e6")
    parser.add_argument("--fft-sizes", type=str, default="256,512,1024,2048")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--warmup", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=Path("measurement/modulator_cpu_bench"))
    return parser.parse_args()

def build_mod_role_map(mod_cfg: dict, pid: int) -> dict[tuple[int, str], str]:
    role_map: dict[tuple[int, str], str] = {}
    mod_cores = [int(core) for core in mod_cfg.get("cpu_cores", [])]
    if mod_cores:
        base_roles = ["mod:_tx_proc", "mod:_modulation_proc", "mod:_data_ingest_proc"]
        for idx, role in enumerate(base_roles):
            if idx < len(mod_cores) and mod_cores[idx] >= 0:
                role_map[(pid, str(mod_cores[idx]))] = role
        sensing_count = int(mod_cfg.get("sensing_rx_channel_count", len(mod_cfg.get("sensing_rx_channels", [])) or 0))
        for ch_idx in range(sensing_count):
            rx_idx = 3 + ch_idx * 2
            proc_idx = rx_idx + 1
            if rx_idx < len(mod_cores) and mod_cores[rx_idx] >= 0:
                role_map[(pid, str(mod_cores[rx_idx]))] = f"mod:sensing_rx_loop_ch{ch_idx}"
            if proc_idx < len(mod_cores) and mod_cores[proc_idx] >= 0:
                role_map[(pid, str(mod_cores[proc_idx]))] = f"mod:sensing_process_loop_ch{ch_idx}"
        if mod_cores[-1] >= 0:
            role_map[(pid, str(mod_cores[-1]))] = "mod:main"
    return role_map


def build_isolated_cpu_spec(mod_cfg: dict) -> str:
    cpus = {int(cpu) for cpu in mod_cfg.get("cpu_cores", []) if int(cpu) >= 0}
    if not cpus:
        raise RuntimeError("Modulator config must define at least one non-negative cpu_cores entry for isolate_cpus.bash")
    return ",".join(str(cpu) for cpu in sorted(cpus))


def launch_modulator_with_isolation(
    build_dir: Path,
    run_dir: Path,
    isolate_script: Path,
    isolated_cpu_spec: str,
    unit_name: str,
) -> tuple[subprocess.Popen[bytes], int, float]:
    if os.geteuid() != 0:
        raise RuntimeError("bench_modulator_cpu.py must run as root to use isolate_cpus.bash")

    build_dir = build_dir.resolve()
    run_dir = run_dir.resolve()
    isolate_script = isolate_script.resolve()
    subprocess.run(
        [str(isolate_script), isolated_cpu_spec],
        cwd=run_dir,
        check=True,
    )

    env_args: list[str] = []
    home = os.environ.get("HOME")
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    uhd_config_file = os.environ.get("UHD_CONFIG_FILE")
    ld_library_path = os.environ.get("LD_LIBRARY_PATH")
    if home:
        env_args.extend(["--setenv", f"HOME={home}"])
    if xdg_config_home:
        env_args.extend(["--setenv", f"XDG_CONFIG_HOME={xdg_config_home}"])
    elif home:
        env_args.extend(["--setenv", f"XDG_CONFIG_HOME={home}/.config"])
    if uhd_config_file:
        env_args.extend(["--setenv", f"UHD_CONFIG_FILE={uhd_config_file}"])
    if ld_library_path:
        env_args.extend(["--setenv", f"LD_LIBRARY_PATH={ld_library_path}"])

    log_since = time.time()
    launcher = subprocess.Popen(
        [
            "systemd-run",
            "--slice=rt-workload.slice",
            "--unit",
            unit_name,
            "--collect",
            "-p",
            f"AllowedCPUs={isolated_cpu_spec}",
            "-p",
            "Type=exec",
            "-p",
            f"WorkingDirectory={str(run_dir)}",
            *env_args,
            str((build_dir / "OFDMModulator").resolve()),
        ],
        cwd=run_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    main_pid = 0
    deadline = time.time() + 10.0
    while time.time() < deadline:
        result = subprocess.run(
            ["systemctl", "show", unit_name, "--property", "MainPID", "--value"],
            cwd=run_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        text = result.stdout.strip()
        if text.isdigit():
            main_pid = int(text)
            if main_pid > 0:
                break
        time.sleep(0.1)

    if main_pid <= 0:
        launcher_output = b""
        try:
            launcher_output = launcher.communicate(timeout=1.0)[0]
        except subprocess.TimeoutExpired:
            launcher.terminate()
            pass
        except ProcessLookupError:
            pass

        status_result = subprocess.run(
            ["systemctl", "status", unit_name, "--no-pager"],
            cwd=run_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        raise RuntimeError(
            f"Failed to resolve MainPID for transient unit {unit_name}\n"
            f"systemd-run output:\n{launcher_output.decode(errors='ignore')}\n"
            f"systemctl status:\n{status_result.stdout}\n{status_result.stderr}"
        )

    return launcher, main_pid, log_since


def collect_unit_logs(unit_name: str, since_epoch: float) -> bytes:
    since_text = dt.datetime.fromtimestamp(since_epoch).strftime("%Y-%m-%d %H:%M:%S.%f")
    result = subprocess.run(
        ["journalctl", "-u", unit_name, "--since", since_text, "--no-pager", "--output", "cat"],
        capture_output=True,
        check=False,
    )
    return result.stdout


def stop_unit(unit_name: str) -> None:
    subprocess.run(["systemctl", "stop", unit_name], capture_output=True, check=False)


def main() -> None:
    args = parse_args()
    sample_rates = [float(item) for item in args.sample_rates.split(",") if item.strip()]
    fft_sizes = [int(item) for item in args.fft_sizes.split(",") if item.strip()]
    base_mod_cfg = load_yaml(args.mod_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    dynamic_role_fields: set[str] = set()
    for sample_rate in sample_rates:
        for fft_size in fft_sizes:
            run_id = f"mod_sr{safe_stem(str(int(sample_rate)))}_fft{fft_size}"
            run_dir = args.output_dir / run_id

            mod_cfg = dict(base_mod_cfg)
            apply_fft_sample_rate_sweep(mod_cfg, mod_cfg, sample_rate=sample_rate, fft_size=fft_size)
            mod_cfg["range_fft_size"] = fft_size

            mod_proc = None
            mod_pid = 0
            log_since = 0.0
            cpu_rows: list[dict[str, float]] = []
            thread_rows: list[dict[str, object]] = []
            unit_name = f"bench-mod-{run_id}".replace("_", "-")
            try:
                run_dir.mkdir(parents=True, exist_ok=True)
                from bench_utils import save_yaml

                save_yaml(run_dir / "Modulator.yaml", mod_cfg)
                isolated_cpu_spec = build_isolated_cpu_spec(mod_cfg)
                mod_proc, mod_pid, log_since = launch_modulator_with_isolation(
                    args.build_dir,
                    run_dir,
                    args.isolate_script,
                    isolated_cpu_spec,
                    unit_name,
                )
                time.sleep(args.warmup)
                cpu_rows, thread_rows = sample_cpu_usage_with_threads(
                    [mod_pid],
                    args.duration,
                    args.interval,
                )
            finally:
                stop_unit(unit_name)
                mod_log = collect_unit_logs(unit_name, log_since) if mod_proc is not None else b""
                _ = terminate_process_tree(mod_proc) if mod_proc is not None else b""
                (run_dir / "modulator.log").write_bytes(mod_log)

            if mod_proc is None or mod_pid <= 0:
                raise RuntimeError(f"Failed to launch OFDMModulator for run {run_id}")

            role_map = build_mod_role_map(mod_cfg, mod_pid)
            role_rows = summarize_thread_rows(thread_rows, role_map)
            for row in thread_rows:
                row["role"] = role_map.get((int(row["pid"]), str(row["allowed_cpus"])), "")
                row["run_id"] = run_id
            for row in role_rows:
                row["run_id"] = run_id

            write_csv(
                run_dir / "thread_cpu_summary.csv",
                ["run_id", "pid", "tid", "comm", "allowed_cpus", "last_processor", "role", "avg_cpu_pct", "peak_cpu_pct", "samples"],
                thread_rows,
            )
            write_csv(
                run_dir / "thread_role_cpu_summary.csv",
                ["run_id", "pid", "allowed_cpus", "role", "avg_cpu_pct", "peak_cpu_pct", "thread_count", "threads"],
                role_rows,
            )

            role_metrics: dict[str, object] = {}
            for row in role_rows:
                role = str(row.get("role", "")).strip()
                if not role:
                    continue
                role_key = safe_stem(role)
                avg_field = f"{role_key}_avg_cpu_pct"
                peak_field = f"{role_key}_peak_cpu_pct"
                role_metrics[avg_field] = row["avg_cpu_pct"]
                role_metrics[peak_field] = row["peak_cpu_pct"]
                dynamic_role_fields.add(avg_field)
                dynamic_role_fields.add(peak_field)

            summary_rows.append(
                {
                    "run_id": run_id,
                    "sample_rate": sample_rate,
                    "fft_size": fft_size,
                    "avg_cpu_pct": mean_of(cpu_rows, "proc_cpu_pct"),
                    "peak_cpu_pct": peak_of(cpu_rows, "proc_cpu_pct"),
                    "avg_host_cpu_pct": mean_of(cpu_rows, "host_cpu_pct"),
                    "peak_host_cpu_pct": peak_of(cpu_rows, "host_cpu_pct"),
                    "thread_cpu_summary": str(run_dir / "thread_cpu_summary.csv"),
                    "thread_role_cpu_summary": str(run_dir / "thread_role_cpu_summary.csv"),
                    "isolated_cpu_spec": isolated_cpu_spec,
                    "run_dir": str(run_dir),
                    **role_metrics,
                }
            )

    base_fields = [
        "run_id",
        "sample_rate",
        "fft_size",
        "avg_cpu_pct",
        "peak_cpu_pct",
        "avg_host_cpu_pct",
        "peak_host_cpu_pct",
        "thread_cpu_summary",
        "thread_role_cpu_summary",
        "isolated_cpu_spec",
        "run_dir",
    ]
    write_csv(args.output_dir / "modulator_cpu_summary.csv", base_fields + sorted(dynamic_role_fields), summary_rows)


if __name__ == "__main__":
    main()
