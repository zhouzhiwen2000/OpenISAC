from __future__ import annotations

import argparse
import datetime as dt
import os
import resource
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from bench_utils import (
    apply_fft_sample_rate_sweep,
    load_yaml,
    mean_of,
    peak_of,
    safe_stem,
    sample_cpu_usage_with_threads,
    save_yaml,
    summarize_thread_rows,
    terminate_process_tree,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CPU load for CPU OFDMDemodulator.")
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument(
        "--mod-config",
        type=Path,
        default=Path("scripts/bench_demodulator_cpu_modulator_template.yaml"),
        help="Base Modulator YAML template used for reproducible CPU benchmark runs.",
    )
    parser.add_argument(
        "--demod-config",
        type=Path,
        default=Path("scripts/bench_demodulator_cpu_demodulator_template.yaml"),
        help="Base Demodulator YAML template used for reproducible CPU benchmark runs.",
    )
    parser.add_argument("--isolate-script", type=Path, default=Path("scripts/isolate_cpus.bash"))
    parser.add_argument("--sample-rates", type=str, default="50e6,100e6,200e6")
    parser.add_argument("--fft-sizes", type=str, default="256,512,1024,2048")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--startup-gap", type=float, default=1.0)
    parser.add_argument("--warmup", type=float, default=30.0)
    parser.add_argument(
        "--save-core-dumps",
        action="store_true",
        help="Temporarily enable local core dump files in each run directory for crash analysis.",
    )
    parser.add_argument(
        "--core-pattern",
        type=str,
        default="core.%e.%p.%t",
        help="Kernel core_pattern to use while --save-core-dumps is enabled.",
    )
    parser.add_argument(
        "--coredump-retries",
        type=int,
        default=1,
        help="How many times to retry a run automatically if the demodulator unit exits with core-dump.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("measurement/demodulator_cpu_bench"))
    return parser.parse_args()


def build_demod_role_map(demod_cfg: dict, pid: int) -> dict[tuple[int, str], str]:
    role_map: dict[tuple[int, str], str] = {}
    demod_cores = [str(core) for core in demod_cfg.get("cpu_cores", [])]
    if demod_cores:
        base_roles = [
            "demod:rx_proc",
            "demod:process_proc",
            "demod:sensing_process_proc",
            "demod:bit_processing_proc",
        ]
        for idx, role in enumerate(base_roles):
            if idx < len(demod_cores):
                role_map[(pid, demod_cores[idx])] = role
        role_map[(pid, demod_cores[-1])] = "demod:main"
    return role_map


def build_isolated_cpu_spec(*cfgs: dict) -> str:
    cpus: set[int] = set()
    for cfg in cfgs:
        cpus.update(int(cpu) for cpu in cfg.get("cpu_cores", []))
    if not cpus:
        raise RuntimeError("Benchmark configs must define cpu_cores for isolate_cpus.bash")
    return ",".join(str(cpu) for cpu in sorted(cpus))


def prepare_isolated_cpus(
    run_dir: Path,
    isolate_script: Path,
    isolated_cpu_spec: str,
) -> None:
    if os.geteuid() != 0:
        raise RuntimeError("bench_demodulator_cpu.py must run as root to use isolate_cpus.bash")

    run_dir = run_dir.resolve()
    isolate_script = isolate_script.resolve()
    subprocess.run(
        [str(isolate_script), isolated_cpu_spec],
        cwd=run_dir,
        check=True,
    )


def set_core_limit_unlimited() -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))


def setsid_with_unlimited_core() -> None:
    os.setsid()
    set_core_limit_unlimited()


def read_core_pattern() -> str:
    return Path("/proc/sys/kernel/core_pattern").read_text(encoding="utf-8").strip()


def write_core_pattern(pattern: str) -> None:
    Path("/proc/sys/kernel/core_pattern").write_text(pattern, encoding="utf-8")


@contextmanager
def local_core_dump_mode(enabled: bool, core_pattern: str):
    if not enabled:
        yield
        return

    original_pattern = read_core_pattern()
    set_core_limit_unlimited()
    write_core_pattern(core_pattern)
    try:
        yield
    finally:
        write_core_pattern(original_pattern)


def launch_demodulator_with_isolation(
    build_dir: Path,
    run_dir: Path,
    isolated_cpu_spec: str,
    unit_name: str,
    save_core_dumps: bool = False,
) -> tuple[subprocess.Popen[bytes], int, float]:
    if os.geteuid() != 0:
        raise RuntimeError("bench_demodulator_cpu.py must run as root to use isolate_cpus.bash")

    build_dir = build_dir.resolve()
    run_dir = run_dir.resolve()

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
    unit_props = [
        "-p",
        f"AllowedCPUs={isolated_cpu_spec}",
        "-p",
        "Type=exec",
        "-p",
        f"WorkingDirectory={str(run_dir)}",
    ]
    if save_core_dumps:
        unit_props.extend(["-p", "LimitCORE=infinity"])
    launcher = subprocess.Popen(
        [
            "systemd-run",
            "--slice=rt-workload.slice",
            "--unit",
            unit_name,
            "--collect",
            *unit_props,
            *env_args,
            str((build_dir / "OFDMDemodulator").resolve()),
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


def read_unit_status(unit_name: str, run_dir: Path) -> dict[str, str]:
    properties = [
        "Id",
        "LoadState",
        "ActiveState",
        "SubState",
        "Result",
        "MainPID",
        "ExecMainCode",
        "ExecMainStatus",
    ]
    result = subprocess.run(
        ["systemctl", "show", unit_name, "--property", ",".join(properties)],
        cwd=run_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    status: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        status[key] = value
    return status


def unit_status_text(status: dict[str, str]) -> str:
    ordered_keys = [
        "Id",
        "LoadState",
        "ActiveState",
        "SubState",
        "Result",
        "MainPID",
        "ExecMainCode",
        "ExecMainStatus",
    ]
    return "\n".join(f"{key}={status.get(key, '')}" for key in ordered_keys) + "\n"


def unit_hit_coredump(status: dict[str, str]) -> bool:
    result = status.get("Result", "").strip()
    exec_main_status = status.get("ExecMainStatus", "").strip()
    return result == "core-dump" or exec_main_status == "11"


def main() -> None:
    args = parse_args()
    sample_rates = [float(item) for item in args.sample_rates.split(",") if item.strip()]
    fft_sizes = [int(item) for item in args.fft_sizes.split(",") if item.strip()]
    base_mod_cfg = load_yaml(args.mod_config)
    base_demod_cfg = load_yaml(args.demod_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    dynamic_role_fields: set[str] = set()
    for sample_rate in sample_rates:
        for fft_size in fft_sizes:
            run_id = f"demod_sr{safe_stem(str(int(sample_rate)))}_fft{fft_size}"
            run_dir = args.output_dir / run_id

            mod_cfg = dict(base_mod_cfg)
            demod_cfg = dict(base_demod_cfg)
            apply_fft_sample_rate_sweep(mod_cfg, demod_cfg, sample_rate=sample_rate, fft_size=fft_size)
            mod_cfg["range_fft_size"] = fft_size
            demod_cfg["range_fft_size"] = fft_size

            cpu_rows: list[dict[str, float]] = []
            thread_rows: list[dict[str, object]] = []
            isolated_cpu_spec = build_isolated_cpu_spec(mod_cfg, demod_cfg)
            attempt_count = max(0, args.coredump_retries) + 1
            coredump_detected = False
            last_unit_status: dict[str, str] = {}
            unit_name = f"bench-demod-{run_id}".replace("_", "-")

            for attempt_idx in range(1, attempt_count + 1):
                mod_proc = None
                demod_proc = None
                demod_pid = 0
                log_since = 0.0
                unit_status: dict[str, str] = {}
                cpu_rows = []
                thread_rows = []
                coredump_detected = False
                try:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    save_yaml(run_dir / "Modulator.yaml", mod_cfg)
                    save_yaml(run_dir / "Demodulator.yaml", demod_cfg)

                    with local_core_dump_mode(args.save_core_dumps, args.core_pattern):
                        prepare_isolated_cpus(run_dir, args.isolate_script, isolated_cpu_spec)
                        mod_proc = subprocess.Popen(
                            [str((args.build_dir / "OFDMModulator").resolve())],
                            cwd=run_dir,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            preexec_fn=setsid_with_unlimited_core if args.save_core_dumps else os.setsid,
                        )
                        time.sleep(args.startup_gap)
                        demod_proc, demod_pid, log_since = launch_demodulator_with_isolation(
                            args.build_dir,
                            run_dir,
                            isolated_cpu_spec,
                            unit_name,
                            save_core_dumps=args.save_core_dumps,
                        )
                        time.sleep(args.warmup)
                        cpu_rows, thread_rows = sample_cpu_usage_with_threads(
                            [demod_pid],
                            args.duration,
                            args.interval,
                        )
                finally:
                    if demod_proc is not None:
                        unit_status = read_unit_status(unit_name, run_dir)
                    mod_log = terminate_process_tree(mod_proc) if mod_proc is not None else b""
                    stop_unit(unit_name)
                    demod_log = collect_unit_logs(unit_name, log_since) if demod_proc is not None else b""
                    _ = terminate_process_tree(demod_proc) if demod_proc is not None else b""

                    mod_attempt_log = run_dir / f"modulator.attempt{attempt_idx}.log"
                    demod_attempt_log = run_dir / f"demodulator.attempt{attempt_idx}.log"
                    status_attempt_path = run_dir / f"demodulator_unit_status.attempt{attempt_idx}.txt"
                    mod_attempt_log.write_bytes(mod_log)
                    demod_attempt_log.write_bytes(demod_log)
                    status_attempt_path.write_text(unit_status_text(unit_status), encoding="utf-8")
                    (run_dir / "modulator.log").write_bytes(mod_log)
                    (run_dir / "demodulator.log").write_bytes(demod_log)
                    (run_dir / "demodulator_unit_status.txt").write_text(
                        unit_status_text(unit_status),
                        encoding="utf-8",
                    )

                    last_unit_status = unit_status
                    coredump_detected = unit_hit_coredump(unit_status)

                if coredump_detected and attempt_idx < attempt_count:
                    continue
                break

            if coredump_detected:
                raise RuntimeError(
                    f"OFDMDemodulator hit core-dump for run {run_id} after {attempt_count} attempts.\n"
                    f"{unit_status_text(last_unit_status)}"
                )

            if demod_pid <= 0:
                raise RuntimeError(f"Failed to launch OFDMDemodulator for run {run_id}")

            role_map = build_demod_role_map(demod_cfg, demod_pid)
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
    write_csv(args.output_dir / "demodulator_cpu_summary.csv", base_fields + sorted(dynamic_role_fields), summary_rows)


if __name__ == "__main__":
    main()
