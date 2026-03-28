from __future__ import annotations

import csv
import math
import os
import signal
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import yaml


def generate_uniform_pilot_positions(fft_size: int, num_pilots: int = 16) -> list[int]:
    if fft_size <= 0 or num_pilots <= 0:
        return []
    positions: list[int] = []
    for idx in range(1, num_pilots + 1):
        pilot_before_shift = round(fft_size / (num_pilots + 1) * idx - 1)
        positions.append((pilot_before_shift + fft_size // 2) % fft_size)
    return positions


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def save_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def configure_measurement(
    cfg: dict,
    run_id: str,
    output_dir: Path,
    payload_bytes: int,
    prbs_seed: int,
) -> None:
    cfg["measurement_enable"] = True
    cfg["measurement_mode"] = "internal_prbs"
    cfg["measurement_run_id"] = run_id
    cfg["measurement_output_dir"] = str(output_dir)
    cfg["measurement_payload_bytes"] = int(payload_bytes)
    cfg["measurement_prbs_seed"] = int(prbs_seed)


def apply_fft_sample_rate_sweep(
    mod_cfg: dict,
    demod_cfg: dict,
    sample_rate: float,
    fft_size: int,
    num_pilots: int = 16,
) -> None:
    pilot_positions = generate_uniform_pilot_positions(fft_size, num_pilots=num_pilots)
    seen_cfg_ids: set[int] = set()
    for cfg in (mod_cfg, demod_cfg):
        cfg_id = id(cfg)
        if cfg_id in seen_cfg_ids:
            continue
        seen_cfg_ids.add(cfg_id)

        base_fft_size = int(cfg.get("fft_size", 1024))
        base_cp_length = int(cfg.get("cp_length", 128))
        base_num_symbols = int(cfg.get("num_symbols", 100))
        base_symbol_samples = base_fft_size + base_cp_length
        base_frame_samples = max(1, base_num_symbols * base_symbol_samples)
        cp_ratio = (base_cp_length / base_fft_size) if base_fft_size > 0 else 0.125

        new_cp_length = max(1, int(round(fft_size * cp_ratio)))
        new_symbol_samples = fft_size + new_cp_length
        new_num_symbols = max(1, int(round(base_frame_samples / new_symbol_samples)))

        cfg["sample_rate"] = float(sample_rate)
        cfg["bandwidth"] = float(sample_rate)
        cfg["fft_size"] = int(fft_size)
        cfg["cp_length"] = new_cp_length
        cfg["num_symbols"] = new_num_symbols
        if "desired_peak_pos" in cfg:
            base_desired_peak_pos = int(cfg.get("desired_peak_pos", 0))
            if base_cp_length > 0:
                scaled_desired_peak_pos = int(round(base_desired_peak_pos * new_cp_length / base_cp_length))
            else:
                scaled_desired_peak_pos = base_desired_peak_pos
            cfg["desired_peak_pos"] = max(0, min(new_cp_length - 1, scaled_desired_peak_pos))
        if "sensing_symbol_num" in cfg:
            cfg["sensing_symbol_num"] = new_num_symbols
            if "doppler_fft_size" in cfg:
                cfg["doppler_fft_size"] = max(int(cfg.get("doppler_fft_size", 0)), new_num_symbols)
        cfg["pilot_positions"] = pilot_positions


def launch_openisac_pair(
    build_dir: Path,
    run_dir: Path,
    mod_cfg: dict,
    demod_cfg: dict,
    demod_bin: str = "OFDMDemodulator",
    mod_bin: str = "OFDMModulator",
    startup_gap_s: float = 1.0,
) -> tuple[subprocess.Popen[bytes], subprocess.Popen[bytes]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(run_dir / "Modulator.yaml", mod_cfg)
    save_yaml(run_dir / "Demodulator.yaml", demod_cfg)

    demod_proc = subprocess.Popen(
        [str(build_dir / demod_bin)],
        cwd=run_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    time.sleep(startup_gap_s)
    mod_proc = subprocess.Popen(
        [str(build_dir / mod_bin)],
        cwd=run_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return mod_proc, demod_proc


def launch_openisac_process(
    build_dir: Path,
    run_dir: Path,
    binary_name: str,
    *,
    mod_cfg: dict | None = None,
    demod_cfg: dict | None = None,
) -> subprocess.Popen[bytes]:
    run_dir.mkdir(parents=True, exist_ok=True)
    if mod_cfg is not None:
        save_yaml(run_dir / "Modulator.yaml", mod_cfg)
    if demod_cfg is not None:
        save_yaml(run_dir / "Demodulator.yaml", demod_cfg)

    return subprocess.Popen(
        [str(build_dir / binary_name)],
        cwd=run_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def terminate_process_tree(proc: subprocess.Popen[bytes], timeout_s: float = 10.0) -> bytes:
    if proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        try:
            return proc.communicate(timeout=timeout_s)[0]
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            return proc.communicate(timeout=timeout_s)[0]
    return proc.communicate()[0]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: str | None, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def mean_of(rows: Iterable[dict[str, str]], key: str) -> float:
    values = [parse_float(row.get(key)) for row in rows]
    values = [value for value in values if not math.isnan(value)]
    return sum(values) / len(values) if values else math.nan


def peak_of(rows: Iterable[dict[str, str]], key: str) -> float:
    values = [parse_float(row.get(key)) for row in rows]
    values = [value for value in values if not math.isnan(value)]
    return max(values) if values else math.nan


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_total_cpu_ticks() -> tuple[int, int]:
    with Path("/proc/stat").open("r", encoding="utf-8") as handle:
        first = handle.readline().strip().split()
    values = [int(x) for x in first[1:]]
    total = sum(values)
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return total, idle


def read_process_ticks(pid: int) -> int | None:
    stat_path = Path("/proc") / str(pid) / "stat"
    parsed = read_proc_stat(stat_path)
    if parsed is None:
        return None
    return parsed["ticks"]


def read_proc_stat(stat_path: Path) -> dict[str, object] | None:
    if not stat_path.exists():
        return None
    content = stat_path.read_text(encoding="utf-8").strip()
    after_paren = content[content.rfind(")") + 2 :]
    fields = after_paren.split()
    if len(fields) < 39:
        return None
    utime = int(fields[11])
    stime = int(fields[12])
    processor = int(fields[36])
    return {
        "ticks": utime + stime,
        "processor": processor,
    }


def read_task_comm(pid: int, tid: int) -> str:
    comm_path = Path("/proc") / str(pid) / "task" / str(tid) / "comm"
    try:
        return comm_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def read_task_allowed_cpus(pid: int, tid: int) -> str:
    status_path = Path("/proc") / str(pid) / "task" / str(tid) / "status"
    try:
        with status_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("Cpus_allowed_list:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return ""
    return ""


def read_thread_snapshot(pids: list[int]) -> dict[tuple[int, int], dict[str, object]]:
    snapshot: dict[tuple[int, int], dict[str, object]] = {}
    for pid in pids:
        task_dir = Path("/proc") / str(pid) / "task"
        if not task_dir.exists():
            continue
        try:
            tids = [int(entry.name) for entry in task_dir.iterdir() if entry.name.isdigit()]
        except OSError:
            continue
        for tid in tids:
            stat_path = task_dir / str(tid) / "stat"
            parsed = read_proc_stat(stat_path)
            if parsed is None:
                continue
            snapshot[(pid, tid)] = {
                "ticks": int(parsed["ticks"]),
                "processor": int(parsed["processor"]),
                "comm": read_task_comm(pid, tid),
                "allowed_cpus": read_task_allowed_cpus(pid, tid),
            }
    return snapshot


def sample_cpu_usage(
    pids: list[int],
    duration_s: float,
    interval_s: float,
) -> list[dict[str, float]]:
    cpu_count = os.cpu_count() or 1
    samples: list[dict[str, float]] = []
    prev_total, prev_idle = read_total_cpu_ticks()
    prev_proc = sum(read_process_ticks(pid) or 0 for pid in pids)
    deadline = time.time() + duration_s

    while time.time() < deadline:
        time.sleep(interval_s)
        curr_total, curr_idle = read_total_cpu_ticks()
        curr_proc = sum(read_process_ticks(pid) or 0 for pid in pids)
        delta_total = curr_total - prev_total
        delta_idle = curr_idle - prev_idle
        delta_proc = curr_proc - prev_proc
        prev_total, prev_idle, prev_proc = curr_total, curr_idle, curr_proc
        if delta_total <= 0:
            continue
        proc_cpu_pct = 100.0 * cpu_count * delta_proc / delta_total
        host_cpu_pct = 100.0 * (1.0 - delta_idle / delta_total)
        samples.append(
            {
                "proc_cpu_pct": proc_cpu_pct,
                "host_cpu_pct": host_cpu_pct,
            }
        )
    return samples


def sample_cpu_usage_with_threads(
    pids: list[int],
    duration_s: float,
    interval_s: float,
) -> tuple[list[dict[str, float]], list[dict[str, object]]]:
    cpu_count = os.cpu_count() or 1
    samples: list[dict[str, float]] = []
    thread_acc: dict[tuple[int, int], dict[str, object]] = {}
    prev_total, prev_idle = read_total_cpu_ticks()
    prev_proc = sum(read_process_ticks(pid) or 0 for pid in pids)
    prev_threads = read_thread_snapshot(pids)
    deadline = time.time() + duration_s

    while time.time() < deadline:
        time.sleep(interval_s)
        curr_total, curr_idle = read_total_cpu_ticks()
        curr_proc = sum(read_process_ticks(pid) or 0 for pid in pids)
        curr_threads = read_thread_snapshot(pids)
        delta_total = curr_total - prev_total
        delta_idle = curr_idle - prev_idle
        delta_proc = curr_proc - prev_proc
        prev_total, prev_idle, prev_proc = curr_total, curr_idle, curr_proc
        if delta_total <= 0:
            prev_threads = curr_threads
            continue

        proc_cpu_pct = 100.0 * cpu_count * delta_proc / delta_total
        host_cpu_pct = 100.0 * (1.0 - delta_idle / delta_total)
        samples.append(
            {
                "proc_cpu_pct": proc_cpu_pct,
                "host_cpu_pct": host_cpu_pct,
            }
        )

        common_keys = set(prev_threads).intersection(curr_threads)
        for key in common_keys:
            prev_info = prev_threads[key]
            curr_info = curr_threads[key]
            delta_ticks = int(curr_info["ticks"]) - int(prev_info["ticks"])
            if delta_ticks < 0:
                continue
            thread_cpu_pct = 100.0 * cpu_count * delta_ticks / delta_total
            if key not in thread_acc:
                pid, tid = key
                thread_acc[key] = {
                    "pid": pid,
                    "tid": tid,
                    "comm": curr_info["comm"],
                    "allowed_cpus": curr_info["allowed_cpus"],
                    "last_processor": curr_info["processor"],
                    "sum_cpu_pct": 0.0,
                    "peak_cpu_pct": 0.0,
                    "samples": 0,
                }
            acc = thread_acc[key]
            acc["comm"] = curr_info["comm"]
            acc["allowed_cpus"] = curr_info["allowed_cpus"]
            acc["last_processor"] = curr_info["processor"]
            acc["sum_cpu_pct"] = float(acc["sum_cpu_pct"]) + thread_cpu_pct
            acc["peak_cpu_pct"] = max(float(acc["peak_cpu_pct"]), thread_cpu_pct)
            acc["samples"] = int(acc["samples"]) + 1

        prev_threads = curr_threads

    thread_rows: list[dict[str, object]] = []
    for key in sorted(thread_acc):
        acc = thread_acc[key]
        sample_count = int(acc["samples"])
        thread_rows.append(
            {
                "pid": int(acc["pid"]),
                "tid": int(acc["tid"]),
                "comm": str(acc["comm"]),
                "allowed_cpus": str(acc["allowed_cpus"]),
                "last_processor": int(acc["last_processor"]),
                "avg_cpu_pct": float(acc["sum_cpu_pct"]) / sample_count if sample_count else math.nan,
                "peak_cpu_pct": float(acc["peak_cpu_pct"]),
                "samples": sample_count,
            }
        )
    return samples, thread_rows


def summarize_thread_rows(
    thread_rows: list[dict[str, object]],
    role_map: dict[tuple[int, str], str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str, str], dict[str, object]] = {}
    for row in thread_rows:
        pid = int(row["pid"])
        allowed_cpus = str(row["allowed_cpus"])
        role = role_map.get((pid, allowed_cpus), "")
        key = (pid, allowed_cpus, role)
        if key not in grouped:
            grouped[key] = {
                "pid": pid,
                "allowed_cpus": allowed_cpus,
                "role": role,
                "avg_cpu_pct": 0.0,
                "peak_cpu_pct": 0.0,
                "thread_count": 0,
                "threads": [],
            }
        acc = grouped[key]
        acc["avg_cpu_pct"] = float(acc["avg_cpu_pct"]) + float(row["avg_cpu_pct"])
        acc["peak_cpu_pct"] = float(acc["peak_cpu_pct"]) + float(row["peak_cpu_pct"])
        acc["thread_count"] = int(acc["thread_count"]) + 1
        acc["threads"].append(f"{row['tid']}:{row['comm']}")

    summary_rows: list[dict[str, object]] = []
    for key in sorted(grouped):
        acc = grouped[key]
        summary_rows.append(
            {
                "pid": acc["pid"],
                "allowed_cpus": acc["allowed_cpus"],
                "role": acc["role"],
                "avg_cpu_pct": acc["avg_cpu_pct"],
                "peak_cpu_pct": acc["peak_cpu_pct"],
                "thread_count": acc["thread_count"],
                "threads": ";".join(acc["threads"]),
            }
        )
    return summary_rows


def safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
