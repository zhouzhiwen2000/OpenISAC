#!/usr/bin/env python3
"""CPU isolation helper for OpenISAC BS/UE real-time workloads.

Default isolation reads ``BS.yaml`` / ``UE.yaml`` (or paths passed with
``--yaml``) and reserves only the most scheduling-sensitive threads:

* USRP sample TX/RX (throughput) threads
* main thread
* BS monostatic sensing RX ingest (``rx_cpu_core``)

OFDM modulation/demodulation, LDPC, UDP, sensing DSP, and other workers are
**not** reserved by default so they can share noisier system cores.

System slices (``system.slice``, ``user.slice``, ``init.scope``) are
restricted to the complement of the reserved set.

``run`` launches the process with ``AllowedCPUs`` covering **all** logical
CPUs (reserved ∪ system). Critical threads stay pinned via YAML affinity to
the reserved cores; other threads may schedule on system cores.

This is soft isolation via systemd cgroup CPU affinity, not kernel
``isolcpus``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required for isolate_cpus.py. Install with: pip install pyyaml"
    ) from exc


APP_CPUS_CONF = Path("/tmp/isolate_cpus_app.conf")
STATE_JSON = Path("/tmp/isolate_cpus_state.json")
RT_SLICE = "rt-workload.slice"

# Role → list indices for cpu_cores.* vectors.
# Default isolate: only the most sensitive sample I/O + main (not OFDM/LDPC).
BS_DOWNLINK_ROLES = {
    0: "bs.tx_proc (USRP TX)",
    # 1 = OFDM modulation — intentionally not isolated by default
}
BS_UPLINK_ROLES = {
    0: "bs.rx_ingest (USRP RX)",
    # 1 = OFDM/LLR demod — intentionally not isolated by default
}
UE_DOWNLINK_ROLES = {
    0: "ue.rx_proc (USRP RX)",
    # 1 = process_proc (OFDM demod) — intentionally not isolated by default
}
UE_UPLINK_ROLES = {
    # 1 = OFDM modulation — intentionally not isolated by default
    2: "ue.tx_send (USRP TX)",
}


@dataclass(frozen=True)
class CoreAssignment:
    core: int
    role: str
    source: str


@dataclass
class IsolationPlan:
    reserved: list[int] = field(default_factory=list)
    system: list[int] = field(default_factory=list)
    process: list[int] = field(default_factory=list)
    assignments: list[CoreAssignment] = field(default_factory=list)
    yaml_files: list[str] = field(default_factory=list)
    mode: str = "yaml"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "reserved_cpus": list(self.reserved),
            "system_cpus": list(self.system),
            "process_cpus": list(self.process),
            "yaml_files": list(self.yaml_files),
            "assignments": [
                {"core": a.core, "role": a.role, "source": a.source}
                for a in self.assignments
            ],
        }


def total_cpu_count() -> int:
    try:
        out = subprocess.check_output(["nproc", "--all"], text=True).strip()
        return max(1, int(out))
    except Exception:
        return max(1, os.cpu_count() or 1)


def all_cpu_ids(total: int | None = None) -> list[int]:
    n = total_cpu_count() if total is None else total
    return list(range(n))


def format_cpu_list(cpus: Sequence[int]) -> str:
    """Compact CPU list: contiguous runs become ranges (e.g. 0-3,5,8-10)."""
    if not cpus:
        return ""
    ordered = sorted({int(c) for c in cpus})
    parts: list[str] = []
    start = prev = ordered[0]
    for cpu in ordered[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        parts.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = cpu
    parts.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(parts)


def parse_cpu_spec(spec: str, max_cpu: int) -> list[int]:
    """Parse ``4``, ``0-3``, ``0,2,4-6`` style CPU specs."""
    spec = spec.strip()
    if not spec:
        return []
    # Plain integer → first N cores
    if re.fullmatch(r"\d+", spec):
        count = int(spec)
        if count <= 0:
            return []
        if count >= max_cpu + 1:
            count = max_cpu  # leave at least one system core when possible
            if count < 0:
                count = 0
        return list(range(count))

    selected: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if re.fullmatch(r"\d+-\d+", token):
            lo_s, hi_s = token.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                lo, hi = hi, lo
            for cpu in range(lo, hi + 1):
                if 0 <= cpu <= max_cpu:
                    selected.add(cpu)
                else:
                    print(f"Warning: CPU {cpu} exceeds max CPU {max_cpu}, ignoring.", file=sys.stderr)
        elif re.fullmatch(r"\d+", token):
            cpu = int(token)
            if 0 <= cpu <= max_cpu:
                selected.add(cpu)
            else:
                print(f"Warning: CPU {cpu} exceeds max CPU {max_cpu}, ignoring.", file=sys.stderr)
        else:
            raise ValueError(f"Unrecognized CPU token '{token}' in '{spec}'")
    return sorted(selected)


def complement_cpus(reserved: Sequence[int], total: int | None = None) -> list[int]:
    all_ids = set(all_cpu_ids(total))
    return sorted(all_ids - {int(c) for c in reserved})


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        try:
            return [int(value)]
        except Exception:
            return []
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except Exception:
            continue
    return out


def _cpu_section(data: dict[str, Any]) -> dict[str, Any]:
    section = data.get("cpu_cores")
    if isinstance(section, dict):
        return section
    return data


def detect_side(path: Path | None, data: dict[str, Any]) -> str:
    if path is not None:
        name = path.name.upper()
        stem = path.stem.upper()
        if stem == "UE" or stem.startswith("UE_") or stem.endswith("_UE") or "UE_" in stem:
            return "ue"
        if stem == "BS" or stem.startswith("BS_") or stem.endswith("_BS") or "BS_" in stem:
            return "bs"
        if "CUDAUE" in name or name.startswith("UE"):
            return "ue"
        if "CUDABS" in name or name.startswith("BS"):
            return "bs"

    cpu = _cpu_section(data)
    if "demod_worker_cpu_cores" in cpu or "ldpc_worker_cpu_cores" in cpu:
        return "ue"
    ofdm = data.get("ofdm_frame")
    if isinstance(ofdm, dict):
        if "cuda_demod_pipeline_slots" in ofdm:
            return "ue"
        if "cuda_mod_pipeline_slots" in ofdm:
            return "bs"
    # Flattened web-editor keys
    if "demod_worker_cpu_cores" in data or "cuda_demod_pipeline_slots" in data:
        return "ue"
    return "bs"


def _pick_indexed_roles(
    cores: Sequence[int],
    roles: dict[int, str],
    source: str,
) -> list[CoreAssignment]:
    assignments: list[CoreAssignment] = []
    for idx, role in roles.items():
        if idx >= len(cores):
            continue
        core = int(cores[idx])
        if core < 0:
            continue
        assignments.append(CoreAssignment(core=core, role=role, source=source))
    return assignments


def extract_critical_assignments(
    data: dict[str, Any],
    *,
    side: str,
    source: str = "config",
) -> list[CoreAssignment]:
    """Extract critical core assignments from nested or flattened config data."""
    side = side.lower().strip()
    if side not in {"bs", "ue"}:
        raise ValueError(f"side must be 'bs' or 'ue', got {side!r}")

    cpu = _cpu_section(data)
    assignments: list[CoreAssignment] = []

    downlink = _as_int_list(cpu.get("downlink_cpu_cores", data.get("downlink_cpu_cores", [])))
    uplink = _as_int_list(cpu.get("uplink_cpu_cores", data.get("uplink_cpu_cores", [])))

    if side == "bs":
        assignments.extend(_pick_indexed_roles(downlink, BS_DOWNLINK_ROLES, source))
        assignments.extend(_pick_indexed_roles(uplink, BS_UPLINK_ROLES, source))
    else:
        assignments.extend(_pick_indexed_roles(downlink, UE_DOWNLINK_ROLES, source))
        assignments.extend(_pick_indexed_roles(uplink, UE_UPLINK_ROLES, source))
        # demod_worker_cpu_cores are OFDM demod helpers — not isolated by default

    main_core = cpu.get("main_cpu_core", data.get("main_cpu_core", -1))
    try:
        main_core_i = int(main_core)
    except Exception:
        main_core_i = -1
    if main_core_i >= 0:
        assignments.append(
            CoreAssignment(core=main_core_i, role=f"{side}.main", source=source)
        )

    # Monostatic sensing USRP RX loops (sample ingest), BS-side primarily.
    if side == "bs":
        sensing = data.get("sensing")
        channels: Any = None
        if isinstance(sensing, dict):
            channels = sensing.get("rx_channels")
        if channels is None:
            channels = data.get("sensing.rx_channels", data.get("rx_channels"))
        if isinstance(channels, list):
            for ch_idx, channel in enumerate(channels):
                if not isinstance(channel, dict):
                    continue
                try:
                    rx_core = int(channel.get("rx_cpu_core", -1))
                except Exception:
                    rx_core = -1
                if rx_core >= 0:
                    assignments.append(
                        CoreAssignment(
                            core=rx_core,
                            role=f"bs.sensing_rx[{ch_idx}] (USRP RX)",
                            source=source,
                        )
                    )

    return assignments


def extract_critical_cores(
    data: dict[str, Any],
    *,
    side: str,
    source: str = "config",
) -> list[int]:
    return sorted({a.core for a in extract_critical_assignments(data, side=side, source=source)})


def load_yaml_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return data


BS_YAML_NAMES = ("BS.yaml", "CUDABS.yaml")
UE_YAML_NAMES = ("UE.yaml", "CUDAUE.yaml")


def default_yaml_candidates(
    cwd: Path | None = None,
    *,
    role: str = "both",
) -> list[Path]:
    """Locate runtime YAML files in *cwd* for the selected machine role."""
    base = cwd or Path.cwd()
    role = normalize_machine_role(role)
    names: list[str] = []
    if role in {"bs", "both"}:
        names.extend(BS_YAML_NAMES)
    if role in {"ue", "both"}:
        names.extend(UE_YAML_NAMES)

    found: list[Path] = []
    seen: set[Path] = set()
    for name in names:
        path = (base / name).resolve()
        if path.is_file() and path not in seen:
            found.append(path)
            seen.add(path)
    return found


def normalize_machine_role(role: str) -> str:
    key = role.strip().lower().replace(" ", "")
    aliases = {
        "1": "bs",
        "bs": "bs",
        "b": "bs",
        "bs-only": "bs",
        "bsonly": "bs",
        "2": "ue",
        "ue": "ue",
        "u": "ue",
        "ue-only": "ue",
        "ueonly": "ue",
        "3": "both",
        "both": "both",
        "all": "both",
        "bs+ue": "both",
        "bsue": "both",
        "duo": "both",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown machine role {role!r}. Use bs | ue | both (or 1/2/3)."
        )
    return aliases[key]


def prompt_machine_role() -> str:
    """Ask whether this host runs BS, UE, or both.

    Interactive (TTY): print a menu and read the choice.
    Non-interactive: accept a single-line role from stdin, or fail and
    suggest ``--role bs|ue|both``.
    """
    labels = {"bs": "BS only", "ue": "UE only", "both": "BS + UE"}

    def _accept(answer: str) -> str:
        role = normalize_machine_role(answer)
        print(f"已选择 / Selected: {labels[role]}")
        return role

    if not sys.stdin.isatty():
        try:
            answer = sys.stdin.readline().strip()
        except Exception:
            answer = ""
        if not answer:
            raise RuntimeError(
                "Default isolation needs a machine role, but no TTY/stdin answer was provided.\n"
                "Pass --role bs|ue|both (recommended for scripts), or pipe 1/2/3 into stdin."
            )
        try:
            return _accept(answer)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    print()
    print("本机运行角色 / Machine role for default core isolation:")
    print("  1) BS only     — 只跑基站 (reads BS.yaml)")
    print("  2) UE only     — 只跑终端 (reads UE.yaml)")
    print("  3) BS + UE     — 同机双端 (reads BS.yaml + UE.yaml)")
    print()

    while True:
        try:
            answer = input("选择 [1/2/3] (bs/ue/both): ").strip()
        except EOFError as exc:
            raise RuntimeError(
                "No role selected. Re-run with --role bs|ue|both."
            ) from exc
        if not answer:
            print("请输入 1 / 2 / 3，或 bs / ue / both。")
            continue
        try:
            return _accept(answer)
        except ValueError as exc:
            print(f"无效输入: {exc}")
            continue


def resolve_role_yaml_paths(
    role: str,
    cwd: Path | None = None,
    *,
    explicit_yaml: Sequence[Path] | None = None,
) -> list[Path]:
    if explicit_yaml:
        paths = []
        for p in explicit_yaml:
            path = p if p.is_absolute() else (cwd or Path.cwd()) / p
            paths.append(path.resolve())
        return paths

    candidates = default_yaml_candidates(cwd, role=role)
    if candidates:
        return candidates

    role = normalize_machine_role(role)
    wanted = []
    if role in {"bs", "both"}:
        wanted.append("BS.yaml")
    if role in {"ue", "both"}:
        wanted.append("UE.yaml")
    raise RuntimeError(
        f"No YAML found in {(cwd or Path.cwd())} for role={role}.\n"
        f"Expected one of: {', '.join(wanted)} (or CUDABS.yaml / CUDAUE.yaml).\n"
        "Copy a preset into the working directory, e.g.:\n"
        "  cp ../config/BS_X310.yaml BS.yaml\n"
        "  cp ../config/UE_X310.yaml UE.yaml\n"
        "Or pass --yaml /path/to/file.yaml ..."
    )


def plan_from_yaml_files(
    yaml_paths: Sequence[Path],
    *,
    total: int | None = None,
    side_override: str | None = None,
    role: str | None = None,
) -> IsolationPlan:
    n = total_cpu_count() if total is None else total
    assignments: list[CoreAssignment] = []
    used_files: list[str] = []

    for path in yaml_paths:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"YAML not found: {path}")
        data = load_yaml_file(path)
        if side_override in {"bs", "ue"}:
            side = side_override
        else:
            side = detect_side(path, data)
        file_assignments = extract_critical_assignments(
            data, side=side, source=str(path)
        )
        if not file_assignments:
            print(
                f"Warning: no critical CPU cores found in {path} (side={side}).",
                file=sys.stderr,
            )
        assignments.extend(file_assignments)
        used_files.append(str(path))

    reserved = sorted({a.core for a in assignments if 0 <= a.core < n})
    out_of_range = sorted({a.core for a in assignments if a.core >= n})
    for core in out_of_range:
        print(
            f"Warning: configured core {core} is outside 0-{n - 1}, ignoring.",
            file=sys.stderr,
        )

    if not reserved:
        raise RuntimeError(
            "No sensitive CPU cores could be derived from the given YAML files. "
            "Set main_cpu_core and USRP TX/RX cores (downlink/uplink sample roles; "
            "optional BS sensing rx_cpu_core) to non-negative values, "
            "or pass an explicit CPU set (e.g. 4 or 0,2,4)."
        )

    system = complement_cpus(reserved, n)
    if not system:
        raise RuntimeError(
            f"Reserving cores {format_cpu_list(reserved)} leaves no CPUs for system slices. "
            "Reduce the critical core set so at least one core remains for the OS."
        )

    mode = "yaml"
    if role:
        mode = f"yaml:{normalize_machine_role(role)}"

    return IsolationPlan(
        reserved=reserved,
        system=system,
        process=all_cpu_ids(n),
        assignments=assignments,
        yaml_files=used_files,
        mode=mode,
    )


def plan_from_manual_spec(spec: str, *, total: int | None = None) -> IsolationPlan:
    n = total_cpu_count() if total is None else total
    max_cpu = n - 1
    reserved = parse_cpu_spec(spec, max_cpu)
    if not reserved:
        raise RuntimeError(f"Manual CPU spec '{spec}' selected no valid CPUs.")
    # If user asked for all cores as app, leave at least nothing for system is bad
    system = complement_cpus(reserved, n)
    if not system:
        # Match old bash behavior for "reserve almost all": shrink reserved by last core
        if len(reserved) > 1:
            reserved = reserved[:-1]
            system = complement_cpus(reserved, n)
        else:
            raise RuntimeError(
                "Manual spec leaves no system CPUs. Keep at least one core for the OS."
            )
    assignments = [
        CoreAssignment(core=c, role="manual", source=f"spec:{spec}") for c in reserved
    ]
    return IsolationPlan(
        reserved=reserved,
        system=system,
        process=all_cpu_ids(n),
        assignments=assignments,
        yaml_files=[],
        mode="manual",
    )


def require_root() -> None:
    if os.geteuid() != 0:
        raise SystemExit("Error: please run this script with sudo/root privileges.")


def run_systemctl_set_allowed(unit: str, cpus: Sequence[int]) -> None:
    spec = format_cpu_list(cpus)
    subprocess.run(
        ["systemctl", "set-property", "--now", unit, f"AllowedCPUs={spec}"],
        check=False,
    )


def apply_isolation(plan: IsolationPlan) -> None:
    print("------------------------------------------------")
    print("Configuring system isolation...")
    print(f"  Mode:            {plan.mode}")
    if plan.yaml_files:
        print(f"  YAML sources:    {', '.join(plan.yaml_files)}")
    print(f"  Reserved (app):  {format_cpu_list(plan.reserved)}")
    print(f"  System slices:   {format_cpu_list(plan.system)}")
    print(f"  Process (run):   {format_cpu_list(plan.process)}  (all CPUs)")
    if plan.assignments:
        print("  Critical roles:")
        # De-dupe display by (core, role)
        seen: set[tuple[int, str]] = set()
        for a in sorted(plan.assignments, key=lambda x: (x.core, x.role)):
            key = (a.core, a.role)
            if key in seen:
                continue
            seen.add(key)
            if a.core < 0:
                continue
            print(f"    CPU {a.core:>3}: {a.role}  [{a.source}]")
    print("------------------------------------------------")

    for unit in ("system.slice", "user.slice", "init.scope"):
        run_systemctl_set_allowed(unit, plan.system)

    APP_CPUS_CONF.write_text(format_cpu_list(plan.reserved) + "\n", encoding="utf-8")
    STATE_JSON.write_text(json.dumps(plan.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"Saved reserved CPU config to {APP_CPUS_CONF}")
    print(f"Saved full isolation state to {STATE_JSON}")
    print("Setup successful.")
    print()
    print("Launch BS/UE with process affinity spanning all CPUs:")
    print("  sudo ./scripts/isolate_cpus.py run ./BS")
    print("  sudo ./scripts/isolate_cpus.py run ./UE")
    print()
    print("Critical threads should remain pinned to reserved cores via YAML;")
    print("non-critical threads can schedule on system cores.")


def reset_isolation() -> None:
    n = total_cpu_count()
    all_cpus = all_cpu_ids(n)
    print("Resetting CPU affinity settings...")
    for unit in ("system.slice", "user.slice", "init.scope"):
        run_systemctl_set_allowed(unit, all_cpus)

    for conf in (
        Path("/etc/systemd/system/system.slice.d/50-AllowedCPUs.conf"),
        Path("/etc/systemd/system/user.slice.d/50-AllowedCPUs.conf"),
        Path("/etc/systemd/system/init.scope.d/50-AllowedCPUs.conf"),
    ):
        try:
            conf.unlink()
        except FileNotFoundError:
            pass

    for d in (
        Path("/etc/systemd/system/system.slice.d"),
        Path("/etc/systemd/system/user.slice.d"),
        Path("/etc/systemd/system/init.scope.d"),
    ):
        try:
            d.rmdir()
        except OSError:
            pass

    subprocess.run(["systemctl", "daemon-reload"], check=False)
    for path in (APP_CPUS_CONF, STATE_JSON):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    print(f"Reset complete. System can use all CPUs ({format_cpu_list(all_cpus)}).")
    print("Note: you may need to restart shells/sessions for affinity to fully refresh.")


def load_saved_plan() -> IsolationPlan | None:
    if STATE_JSON.is_file():
        try:
            raw = json.loads(STATE_JSON.read_text(encoding="utf-8"))
            return IsolationPlan(
                reserved=[int(x) for x in raw.get("reserved_cpus", [])],
                system=[int(x) for x in raw.get("system_cpus", [])],
                process=[int(x) for x in raw.get("process_cpus", [])] or all_cpu_ids(),
                assignments=[
                    CoreAssignment(
                        core=int(a["core"]),
                        role=str(a.get("role", "")),
                        source=str(a.get("source", "")),
                    )
                    for a in raw.get("assignments", [])
                    if isinstance(a, dict)
                ],
                yaml_files=[str(x) for x in raw.get("yaml_files", [])],
                mode=str(raw.get("mode", "saved")),
            )
        except Exception as exc:
            print(f"Warning: failed to parse {STATE_JSON}: {exc}", file=sys.stderr)

    if APP_CPUS_CONF.is_file():
        spec = APP_CPUS_CONF.read_text(encoding="utf-8").strip()
        if spec:
            try:
                return plan_from_manual_spec(spec)
            except Exception as exc:
                print(f"Warning: failed to parse {APP_CPUS_CONF}: {exc}", file=sys.stderr)
    return None


def run_application(
    command: Sequence[str],
    *,
    app_only: bool = False,
    cwd: Path | None = None,
) -> int:
    if not command:
        raise SystemExit("Error: run requires a command, e.g. run ./BS")

    plan = load_saved_plan()
    n = total_cpu_count()
    if plan is None:
        reserved_spec = format_cpu_list(list(range(min(8, max(0, n - 1)))))
        print(
            f"No saved isolation state; process will use all CPUs. "
            f"(Hint: run isolation first. Fallback reserved hint: {reserved_spec})",
            file=sys.stderr,
        )
        process_cpus = all_cpu_ids(n)
        reserved_hint = reserved_spec
    else:
        process_cpus = plan.process if not app_only else plan.reserved
        if not process_cpus:
            process_cpus = all_cpu_ids(n)
        reserved_hint = format_cpu_list(plan.reserved)

    if app_only:
        print(f"Starting process on reserved cores only ({format_cpu_list(process_cpus)})...")
    else:
        print(
            f"Starting process with AllowedCPUs={format_cpu_list(process_cpus)} "
            f"(all CPUs; reserved/critical set was {reserved_hint})..."
        )

    workdir = str((cwd or Path.cwd()).resolve())
    run_env_args: list[str] = []
    for key in ("HOME", "XDG_CONFIG_HOME", "UHD_CONFIG_FILE", "LD_LIBRARY_PATH"):
        val = os.environ.get(key)
        if val:
            run_env_args.extend([f"--setenv={key}={val}"])
    if "XDG_CONFIG_HOME" not in os.environ and os.environ.get("HOME"):
        run_env_args.append(f"--setenv=XDG_CONFIG_HOME={os.environ['HOME']}/.config")

    cmd = [
        "systemd-run",
        f"--slice={RT_SLICE}",
        "--pty",
        f"-p",
        f"AllowedCPUs={format_cpu_list(process_cpus)}",
        f"-p",
        f"WorkingDirectory={workdir}",
        *run_env_args,
        *command,
    ]
    print("+", " ".join(cmd))
    return subprocess.call(cmd)


def show_status() -> None:
    n = total_cpu_count()
    print(f"Total CPUs: {n} (0-{n - 1})")
    atom = Path("/sys/devices/cpu_atom/cpus")
    core = Path("/sys/devices/cpu_core/cpus")
    if atom.is_file() and core.is_file():
        print(f"P-Cores: {core.read_text().strip()}")
        print(f"E-Cores: {atom.read_text().strip()}")
    plan = load_saved_plan()
    if plan is None:
        print("No saved isolation state.")
        return
    print(f"Mode:          {plan.mode}")
    print(f"Reserved:      {format_cpu_list(plan.reserved)}")
    print(f"System:        {format_cpu_list(plan.system)}")
    print(f"Process(run):  {format_cpu_list(plan.process)}")
    if plan.yaml_files:
        print(f"YAML:          {', '.join(plan.yaml_files)}")
    if plan.assignments:
        print("Assignments:")
        seen: set[tuple[int, str]] = set()
        for a in sorted(plan.assignments, key=lambda x: (x.core, x.role)):
            key = (a.core, a.role)
            if key in seen:
                continue
            seen.add(key)
            print(f"  CPU {a.core}: {a.role} ({a.source})")


def show_plan_dry_run(plan: IsolationPlan) -> None:
    print(f"Mode:         {plan.mode}")
    if plan.yaml_files:
        print(f"YAML:         {', '.join(plan.yaml_files)}")
    print(f"Reserved:     {format_cpu_list(plan.reserved)}")
    print(f"System:       {format_cpu_list(plan.system)}")
    print(f"Process(run): {format_cpu_list(plan.process)}")
    print("Critical roles:")
    seen: set[tuple[int, str]] = set()
    for a in sorted(plan.assignments, key=lambda x: (x.core, x.role)):
        key = (a.core, a.role)
        if key in seen or a.core < 0:
            continue
        seen.add(key)
        print(f"  CPU {a.core:>3}: {a.role}  [{a.source}]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="isolate_cpus.py",
        description=(
            "Isolate system slices away from critical OpenISAC CPU cores, "
            "and run BS/UE with full-CPU process affinity."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sudo isolate_cpus.py
      # Default: ask BS / UE / BS+UE, then reserve critical cores from YAML

  sudo isolate_cpus.py --role bs
  sudo isolate_cpus.py --role ue
  sudo isolate_cpus.py --role both
  sudo isolate_cpus.py --yaml build/BS.yaml build/UE.yaml
  sudo isolate_cpus.py 4
  sudo isolate_cpus.py 0,2,4,6
  sudo isolate_cpus.py 8-15
  sudo isolate_cpus.py show-plan --role bs
  sudo isolate_cpus.py run ./BS
  sudo isolate_cpus.py run --app-only ./BS   # old behavior: process on reserved only
  sudo isolate_cpus.py reset
  isolate_cpus.py status
""",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default=None,
        help="Optional: reset | run | status | show-plan | or a manual CPU spec",
    )
    parser.add_argument(
        "rest",
        nargs="*",
        default=[],
        help="Arguments for 'run' (command to execute) or leftover tokens",
    )
    parser.add_argument(
        "--yaml",
        "-y",
        nargs="+",
        type=Path,
        default=None,
        help="YAML files to derive critical cores from (skips role prompt)",
    )
    parser.add_argument(
        "--role",
        choices=("bs", "ue", "both", "1", "2", "3"),
        default=None,
        help=(
            "Machine role for default YAML isolation: bs | ue | both "
            "(or 1/2/3). If omitted in interactive mode, the script asks."
        ),
    )
    parser.add_argument(
        "--side",
        choices=("bs", "ue", "auto"),
        default="auto",
        help="Force side interpretation when reading YAML (default: auto from filename)",
    )
    parser.add_argument(
        "--app-only",
        action="store_true",
        help="With 'run': restrict process AllowedCPUs to reserved cores only",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the isolation plan without applying it",
    )
    return parser


def _looks_like_cpu_spec(token: str) -> bool:
    return bool(
        re.fullmatch(r"\d+", token)
        or re.fullmatch(r"\d+(-\d+)?(,\d+(-\d+)?)*", token)
    )


def resolve_plan_from_args(
    args: argparse.Namespace,
    *,
    interactive: bool = True,
) -> IsolationPlan:
    side_override = None if args.side == "auto" else args.side

    if args.yaml:
        paths = resolve_role_yaml_paths("both", explicit_yaml=args.yaml)
        # If --role is also given with multiple yamls, just use files as-is.
        role = normalize_machine_role(args.role) if args.role else None
        return plan_from_yaml_files(
            paths,
            side_override=side_override,
            role=role,
        )

    # Positional manual CPU spec (legacy CLI): isolate_cpus.py 4 / 0,2,4 / 8-15
    if args.command and _looks_like_cpu_spec(args.command):
        return plan_from_manual_spec(args.command)

    # Default YAML path: ask which role this machine has, then bind from that YAML.
    if args.role:
        role = normalize_machine_role(args.role)
    elif interactive:
        role = prompt_machine_role()
    else:
        raise RuntimeError(
            "Machine role required. Pass --role bs|ue|both, or run interactively."
        )

    paths = resolve_role_yaml_paths(role)
    print(f"Using YAML for role={role}: {', '.join(str(p) for p in paths)}")
    return plan_from_yaml_files(paths, side_override=side_override, role=role)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    cmd = (args.command or "").strip()
    rest = list(args.rest or [])
    if rest and rest[0] == "--":
        rest = rest[1:]

    # Support legacy placement: run --app-only ./BS (flag in rest)
    if cmd == "run" and not args.app_only and "--app-only" in rest:
        args.app_only = True
        rest = [t for t in rest if t != "--app-only"]

    if cmd in {"-h", "--help", "help"}:
        parser.print_help()
        return 0

    if cmd == "status":
        show_status()
        return 0

    if cmd == "reset":
        require_root()
        reset_isolation()
        return 0

    if cmd == "run":
        require_root()
        return run_application(rest, app_only=args.app_only)

    if cmd == "show-plan":
        if rest and _looks_like_cpu_spec(rest[0]) and not args.yaml:
            plan = plan_from_manual_spec(rest[0])
        else:
            try:
                plan = resolve_plan_from_args(args, interactive=True)
            except Exception as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
        show_plan_dry_run(plan)
        return 0

    # Isolation setup (default, manual spec, or --yaml)
    try:
        if cmd and not _looks_like_cpu_spec(cmd) and cmd not in {"", "isolate"}:
            if not args.yaml:
                print(f"Error: unrecognized command '{cmd}'", file=sys.stderr)
                parser.print_help()
                return 2
        plan = resolve_plan_from_args(args, interactive=True)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        show_plan_dry_run(plan)
        return 0

    require_root()
    try:
        apply_isolation(plan)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
