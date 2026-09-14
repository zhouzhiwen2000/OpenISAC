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
restricted to the complement of the reserved set. When the kernel exposes
SMT siblings for reserved CPUs, interactive setup asks whether those siblings
should also be kept idle. When the kernel exposes any ``isolcpus`` CPU, setup
also offers a NIC IRQ isolation menu. The user chooses how many isolated
logical CPUs to use, and NIC IRQs are distributed round-robin across them.

``run`` launches the process with ``AllowedCPUs`` covering all application
CPUs. Dedicated NIC IRQ CPUs are excluded when NIC isolation is enabled.
Critical threads stay pinned via YAML affinity to the reserved cores; other
threads may schedule on system cores.

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
KERNEL_ISOLATED_CPUS = Path("/sys/devices/system/cpu/isolated")
DEFAULT_NIC_MIN_SPEED_MBPS = 10_000

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
class NicIrqPlan:
    mode: str = "off"  # off | isolate
    isolated_cpus: list[int] = field(default_factory=list)
    irq_cpus: list[int] = field(default_factory=list)
    excluded_cpus: list[int] = field(default_factory=list)
    interfaces: list[str] = field(default_factory=list)
    original_combined: dict[str, int] = field(default_factory=dict)
    # One-shot CLI override. Do not persist it in the saved isolation policy.
    combined_queues: int | None = None
    irqbalance_stopped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "isolated_cpus": list(self.isolated_cpus),
            "irq_cpus": list(self.irq_cpus),
            "excluded_cpus": list(self.excluded_cpus),
            "interfaces": list(self.interfaces),
            "original_combined": dict(self.original_combined),
            "irqbalance_stopped": bool(self.irqbalance_stopped),
        }


@dataclass
class IsolationPlan:
    reserved: list[int] = field(default_factory=list)
    system: list[int] = field(default_factory=list)
    process: list[int] = field(default_factory=list)
    smt_siblings_isolated: bool = False
    smt_sibling_cpus: list[int] = field(default_factory=list)
    assignments: list[CoreAssignment] = field(default_factory=list)
    yaml_files: list[str] = field(default_factory=list)
    mode: str = "yaml"
    nic_irq: NicIrqPlan = field(default_factory=NicIrqPlan)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "reserved_cpus": list(self.reserved),
            "system_cpus": list(self.system),
            "process_cpus": list(self.process),
            "smt_siblings_isolated": bool(self.smt_siblings_isolated),
            "smt_sibling_cpus": list(self.smt_sibling_cpus),
            "yaml_files": list(self.yaml_files),
            "nic_irq": self.nic_irq.to_dict(),
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


def parse_cpu_list_text(spec: str) -> list[int]:
    """Parse a Linux cpulist where a plain integer is a CPU id, not a count."""
    selected: set[int] = set()
    for token in spec.strip().split(","):
        token = token.strip()
        if not token:
            continue
        if re.fullmatch(r"\d+", token):
            selected.add(int(token))
            continue
        if re.fullmatch(r"\d+-\d+", token):
            lo_s, hi_s = token.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                lo, hi = hi, lo
            selected.update(range(lo, hi + 1))
            continue
        raise ValueError(f"Invalid Linux CPU-list token {token!r} in {spec!r}")
    return sorted(selected)


def kernel_isolated_cpus() -> list[int]:
    """Return CPUs selected by the kernel ``isolcpus`` boot parameter."""
    try:
        return parse_cpu_list_text(KERNEL_ISOLATED_CPUS.read_text().strip())
    except (FileNotFoundError, OSError, ValueError):
        return []


def cpu_topology_key(cpu: int) -> tuple[int, int]:
    topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    try:
        package = int((topology / "physical_package_id").read_text().strip())
        core = int((topology / "core_id").read_text().strip())
        return package, core
    except (FileNotFoundError, OSError, ValueError):
        return 0, cpu


def expand_physical_cpu_cores(cpus: Sequence[int]) -> list[int]:
    """Return all logical CPUs that share a physical core with *cpus*."""
    topology_keys = {cpu_topology_key(int(cpu)) for cpu in cpus}
    return [cpu for cpu in all_cpu_ids() if cpu_topology_key(cpu) in topology_keys]


def smt_sibling_candidates(cpus: Sequence[int]) -> list[int]:
    """Logical siblings of *cpus* that are not themselves explicitly reserved."""
    reserved = {int(cpu) for cpu in cpus}
    return sorted(set(expand_physical_cpu_cores(cpus)) - reserved)


def normalize_smt_sibling_mode(mode: str) -> str:
    key = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "ask": "ask",
        "1": "isolate",
        "yes": "isolate",
        "y": "isolate",
        "isolate": "isolate",
        "shield": "isolate",
        "2": "share",
        "no": "share",
        "n": "share",
        "share": "share",
        "off": "share",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown SMT sibling mode {mode!r}. Use ask | isolate | share."
        )
    return aliases[key]


def prompt_smt_sibling_mode(plan: IsolationPlan, siblings: Sequence[int]) -> str:
    print()
    print("关键业务 CPU 的 SMT 同胞 / SMT siblings of critical CPUs:")
    reserved_set = set(plan.reserved)
    for cpu in plan.reserved:
        peers = [
            peer
            for peer in expand_physical_cpu_cores([cpu])
            if peer != cpu and peer not in reserved_set
        ]
        if peers:
            print(f"  CPU {cpu} -> sibling {format_cpu_list(peers)}")
    print(f"  可隔离同胞 / Siblings available to isolate: {format_cpu_list(siblings)}")
    print("  1) Isolate siblings — system 和普通应用线程均不使用这些同胞")
    print("  2) Share siblings   — 保持这些同胞可调度")
    print()
    while True:
        try:
            answer = input("是否隔离对应的 SMT 同胞？[1/2]: ").strip()
        except EOFError as exc:
            raise RuntimeError(
                "No SMT sibling policy selected. Re-run with "
                "--smt-siblings isolate|share."
            ) from exc
        try:
            mode = normalize_smt_sibling_mode(answer)
        except ValueError as exc:
            print(f"无效输入: {exc}")
            continue
        if mode == "ask":
            print("请输入 1 / 2。")
            continue
        return mode


def prepare_smt_sibling_plan(
    plan: IsolationPlan,
    *,
    requested_mode: str = "ask",
    interactive: bool = True,
) -> IsolationPlan:
    mode = normalize_smt_sibling_mode(requested_mode)
    siblings = smt_sibling_candidates(plan.reserved)
    if not siblings:
        plan.smt_siblings_isolated = False
        plan.smt_sibling_cpus = []
        return plan

    if mode == "ask":
        if interactive and sys.stdin.isatty():
            mode = prompt_smt_sibling_mode(plan, siblings)
        else:
            mode = "share"
            print(
                "No interactive terminal for SMT sibling selection; keeping "
                "siblings shared. Pass --smt-siblings isolate to shield them."
            )

    plan.smt_siblings_isolated = mode == "isolate"
    plan.smt_sibling_cpus = siblings if plan.smt_siblings_isolated else []
    if plan.smt_siblings_isolated:
        sibling_set = set(siblings)
        plan.system = [cpu for cpu in plan.system if cpu not in sibling_set]
        plan.process = [cpu for cpu in plan.process if cpu not in sibling_set]
        if not plan.system or not plan.process:
            raise RuntimeError(
                "SMT sibling isolation leaves no CPUs for system slices or the "
                "application process. Choose --smt-siblings share."
            )
    return plan


def high_speed_interfaces(
    *,
    minimum_speed_mbps: int = DEFAULT_NIC_MIN_SPEED_MBPS,
    allowlist: Sequence[str] | None = None,
) -> list[str]:
    allowed = {item.strip() for item in (allowlist or []) if item.strip()}
    selected: list[str] = []
    for iface_path in sorted(Path("/sys/class/net").glob("*")):
        iface = iface_path.name
        if iface == "lo" or (allowed and iface not in allowed):
            continue
        try:
            speed = int((iface_path / "speed").read_text().strip())
        except (FileNotFoundError, OSError, ValueError):
            continue
        if speed >= minimum_speed_mbps:
            selected.append(iface)
    return selected


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


def normalize_nic_irq_mode(mode: str) -> str:
    key = mode.strip().lower().replace("_", "-")
    aliases = {
        "0": "off",
        "1": "off",
        "off": "off",
        "no": "off",
        "none": "off",
        "2": "isolate",
        "isolate": "isolate",
        "irq": "isolate",
        "single": "isolate",
        "ask": "ask",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown NIC IRQ mode {mode!r}. Use ask | off | isolate."
        )
    return aliases[key]


def prompt_nic_irq_mode(
    isolated_cpus: Sequence[int],
    available_cpus: Sequence[int],
) -> str:
    print()
    print("网卡 IRQ 隔离 / NIC IRQ isolation:")
    print(
        f"  Kernel-isolated logical CPUs ({len(isolated_cpus)}): "
        f"{format_cpu_list(isolated_cpus)}"
    )
    print(
        f"  Available IRQ logical CPUs ({len(available_cpus)}):   "
        f"{format_cpu_list(available_cpus)}"
    )
    print("  1) Disable / skip     — 不启用网卡 IRQ 隔离")
    print("  2) IRQ isolation      — 保留 combined 队列，IRQ 均匀分配到隔离逻辑 CPU")
    print()

    while True:
        try:
            answer = input("选择 [1/2]: ").strip()
        except EOFError as exc:
            raise RuntimeError(
                "No NIC IRQ choice selected. Re-run with "
                "--nic-irq off|isolate."
            ) from exc
        if not answer:
            print("请输入 1 / 2。")
            continue
        try:
            mode = normalize_nic_irq_mode(answer)
        except ValueError as exc:
            print(f"无效输入: {exc}")
            continue
        if mode == "isolate" and not available_cpus:
            print("无法启用：没有未被应用实时线程占用的隔离逻辑 CPU。")
            continue
        labels = {
            "off": "skip NIC IRQ isolation",
            "isolate": "distribute NIC IRQs across isolated logical CPUs",
        }
        print(f"已选择 / Selected: {labels[mode]}")
        return mode


def prompt_nic_irq_cpu_count(available_cpus: Sequence[int]) -> int:
    """Ask how many available isolated logical CPUs should handle NIC IRQs."""
    maximum = len(available_cpus)
    if maximum < 1:
        raise RuntimeError("No isolated logical CPUs are available for NIC IRQs.")
    print()
    print(
        f"可用隔离逻辑 CPU / Available isolated logical CPUs: "
        f"{format_cpu_list(available_cpus)} ({maximum})"
    )
    while True:
        try:
            answer = input(f"请输入用于网卡 IRQ 的逻辑 CPU 数量 [1-{maximum}]: ").strip()
        except EOFError as exc:
            raise RuntimeError(
                "No NIC IRQ logical CPU count selected. Re-run interactively or "
                "pass --nic-irq-cpu-count N."
            ) from exc
        if not re.fullmatch(r"\d+", answer):
            print(f"请输入 1 到 {maximum} 之间的整数。")
            continue
        count = int(answer)
        if not 1 <= count <= maximum:
            print(f"请输入 1 到 {maximum} 之间的整数。")
            continue
        return count


def prepare_nic_irq_plan(
    plan: IsolationPlan,
    *,
    requested_mode: str = "ask",
    requested_combined: int | None = None,
    requested_irq_cpu_count: int | None = None,
    interactive: bool = True,
    iface_allowlist: Sequence[str] | None = None,
) -> IsolationPlan:
    requested_mode = normalize_nic_irq_mode(requested_mode)
    if requested_combined is not None and requested_combined < 1:
        raise RuntimeError("--nic-combined must be a positive integer.")
    if requested_irq_cpu_count is not None and requested_irq_cpu_count < 1:
        raise RuntimeError("--nic-irq-cpu-count must be a positive integer.")
    isolated = kernel_isolated_cpus()
    application_topology = {cpu_topology_key(cpu) for cpu in plan.reserved}
    available = [
        cpu for cpu in isolated if cpu_topology_key(cpu) not in application_topology
    ]

    if requested_mode == "off":
        if requested_combined is not None or requested_irq_cpu_count is not None:
            raise RuntimeError(
                "--nic-combined and --nic-irq-cpu-count require NIC IRQ isolation."
            )
        return plan

    if not isolated:
        if (
            requested_mode not in {"ask", "off"}
            or requested_combined is not None
            or requested_irq_cpu_count is not None
        ):
            raise RuntimeError(
                "NIC IRQ isolation requires at least one kernel-isolated CPU; "
                "detected: none."
            )
        return plan

    interfaces = high_speed_interfaces(allowlist=iface_allowlist)
    if not interfaces:
        if (
            requested_mode not in {"ask", "off"}
            or requested_combined is not None
            or requested_irq_cpu_count is not None
        ):
            raise RuntimeError(
                "No active >=10 Gbps NIC found for NIC IRQ isolation."
            )
        return plan

    if requested_mode == "ask":
        if not interactive or not sys.stdin.isatty():
            saved = load_saved_plan()
            if saved is not None and saved.nic_irq.mode != "off":
                stale_irq_cpus = set(saved.nic_irq.irq_cpus) - set(isolated)
                if stale_irq_cpus:
                    raise RuntimeError(
                        "Saved NIC IRQ CPUs are no longer kernel-isolated: "
                        f"{format_cpu_list(sorted(stale_irq_cpus))}. Re-run "
                        "interactively or pass --nic-irq off to replace the "
                        "saved NIC policy."
                    )
                saved_excluded_set = set(
                    saved.nic_irq.excluded_cpus or saved.nic_irq.irq_cpus
                )
                overlap = saved_excluded_set.intersection(plan.reserved)
                if overlap:
                    raise RuntimeError(
                        "Saved NIC IRQ CPUs overlap the new application CPU plan: "
                        f"{format_cpu_list(sorted(overlap))}. Re-run interactively "
                        "or pass --nic-irq off after resetting isolation."
                    )
                plan.nic_irq = NicIrqPlan(
                    mode=saved.nic_irq.mode,
                    isolated_cpus=list(saved.nic_irq.isolated_cpus),
                    irq_cpus=list(saved.nic_irq.irq_cpus),
                    excluded_cpus=list(
                        saved.nic_irq.excluded_cpus or saved.nic_irq.irq_cpus
                    ),
                    interfaces=list(saved.nic_irq.interfaces),
                    original_combined=dict(saved.nic_irq.original_combined),
                    combined_queues=requested_combined,
                    irqbalance_stopped=saved.nic_irq.irqbalance_stopped,
                )
                if requested_irq_cpu_count is not None:
                    raise RuntimeError(
                        "--nic-irq-cpu-count requires --nic-irq isolate when "
                        "running non-interactively."
                    )
                excluded_set = set(plan.nic_irq.excluded_cpus)
                plan.system = [cpu for cpu in plan.system if cpu not in excluded_set]
                plan.process = [cpu for cpu in plan.process if cpu not in excluded_set]
                if not plan.system or not plan.process:
                    raise RuntimeError(
                        "Saved NIC IRQ physical-core exclusions leave no CPUs for "
                        "system slices or the application process. Reset isolation "
                        "and select a smaller CPU set."
                    )
            if (
                plan.nic_irq.mode == "off"
                and (
                    requested_combined is not None
                    or requested_irq_cpu_count is not None
                )
            ):
                raise RuntimeError(
                    "--nic-combined and --nic-irq-cpu-count require "
                    "--nic-irq isolate."
                )
            return plan
        mode = prompt_nic_irq_mode(isolated, available)
    else:
        mode = requested_mode

    if mode == "off":
        if requested_combined is not None or requested_irq_cpu_count is not None:
            raise RuntimeError(
                "--nic-combined and --nic-irq-cpu-count require NIC IRQ isolation."
            )
        return plan
    if not available:
        raise RuntimeError(
            "NIC IRQ isolation needs at least one isolated logical CPU that is "
            "not on a physical core used by an application real-time thread."
        )

    if requested_irq_cpu_count is None:
        if interactive and sys.stdin.isatty():
            irq_cpu_count = prompt_nic_irq_cpu_count(available)
        else:
            raise RuntimeError(
                "NIC IRQ isolation requires a logical CPU count in non-interactive "
                "mode. Pass --nic-irq-cpu-count N."
            )
    else:
        irq_cpu_count = requested_irq_cpu_count
    if irq_cpu_count > len(available):
        raise RuntimeError(
            f"Requested {irq_cpu_count} NIC IRQ logical CPUs, but only "
            f"{len(available)} are available: {format_cpu_list(available)}."
        )
    irq_cpus = list(available[:irq_cpu_count])
    excluded_cpus = expand_physical_cpu_cores(irq_cpus)

    previous = load_saved_plan()
    previous_nic = previous.nic_irq if previous is not None else NicIrqPlan()
    original_combined = {
        iface: previous_nic.original_combined[iface]
        for iface in interfaces
        if iface in previous_nic.original_combined
    }
    plan.nic_irq = NicIrqPlan(
        mode=mode,
        isolated_cpus=isolated,
        irq_cpus=irq_cpus,
        excluded_cpus=excluded_cpus,
        interfaces=interfaces,
        original_combined=original_combined,
        combined_queues=requested_combined,
        irqbalance_stopped=previous_nic.irqbalance_stopped,
    )
    excluded_set = set(excluded_cpus)
    plan.system = [cpu for cpu in plan.system if cpu not in excluded_set]
    plan.process = [cpu for cpu in plan.process if cpu not in excluded_set]
    if not plan.system or not plan.process:
        raise RuntimeError(
            "NIC IRQ physical-core exclusions leave no CPUs for system slices or "
            "the application process. Select fewer application/IRQ cores."
        )
    return plan


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


def run_checked_command(command: Sequence[str], *, action: str) -> str:
    result = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"{action} failed: {detail}")
    return result.stdout


def get_iface_combined_count(iface: str) -> int | None:
    result = subprocess.run(
        ["ethtool", "-l", iface],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    in_current = False
    for line in result.stdout.splitlines():
        if line.strip() == "Current hardware settings:":
            in_current = True
            continue
        if in_current:
            match = re.match(r"\s*Combined:\s*(\d+)\s*$", line)
            if match:
                return int(match.group(1))
    return None


def get_iface_irqs(iface: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    try:
        lines = Path("/proc/interrupts").read_text().splitlines()
    except OSError:
        return found
    for line in lines:
        match = re.match(r"\s*(\d+):", line)
        if not match:
            continue
        label = line.split()[-1] if line.split() else ""
        if label == iface or label.startswith(f"{iface}-"):
            found.append((int(match.group(1)), label))
    return found


def stop_irqbalance_for_nic(nic: NicIrqPlan) -> None:
    if shutil.which("systemctl") is None:
        return
    state = subprocess.run(
        ["systemctl", "is-active", "irqbalance"],
        capture_output=True,
        text=True,
        check=False,
    )
    if state.stdout.strip() != "active":
        return
    run_checked_command(
        ["systemctl", "stop", "irqbalance"],
        action="stopping irqbalance",
    )
    nic.irqbalance_stopped = True
    print("  Stopped irqbalance so NIC IRQ affinity remains stable.")


def configure_iface_nic_irq(
    iface: str,
    nic: NicIrqPlan,
    *,
    irq_offset: int = 0,
) -> int:
    current_combined = get_iface_combined_count(iface)
    target_combined = nic.combined_queues
    if target_combined is not None:
        if current_combined is None:
            raise RuntimeError(
                f"{iface} does not expose configurable combined channels via ethtool."
            )
        if current_combined != target_combined:
            nic.original_combined.setdefault(iface, current_combined)
            print(f"  {iface}: setting combined queues to {target_combined}")
            run_checked_command(
                [
                    "ethtool",
                    "-L",
                    iface,
                    "combined",
                    str(target_combined),
                ],
                action=f"setting {iface} combined queues",
            )
        else:
            print(f"  {iface}: combined queues already {target_combined}")
    else:
        if current_combined is None:
            print(f"  {iface}: preserving current combined queues")
        else:
            print(f"  {iface}: preserving {current_combined} combined queues")

    run_checked_command(
        ["ethtool", "-X", iface, "default"],
        action=f"resetting {iface} RSS steering",
    )
    for path in Path(f"/sys/class/net/{iface}/queues").glob("tx-*/xps_cpus"):
        path.write_text("0\n", encoding="ascii")

    irqs = get_iface_irqs(iface)
    if not irqs:
        raise RuntimeError(f"no IRQs found for {iface} after queue configuration")
    if not nic.irq_cpus:
        raise RuntimeError("NIC IRQ isolation selected without any IRQ logical CPUs")
    for index, (irq, label) in enumerate(irqs):
        cpu = nic.irq_cpus[(irq_offset + index) % len(nic.irq_cpus)]
        affinity = Path(f"/proc/irq/{irq}/smp_affinity_list")
        affinity.write_text(f"{cpu}\n", encoding="ascii")
        configured = parse_cpu_list_text(affinity.read_text(encoding="ascii"))
        effective_path = Path(f"/proc/irq/{irq}/effective_affinity_list")
        effective = (
            parse_cpu_list_text(effective_path.read_text(encoding="ascii"))
            if effective_path.is_file()
            else configured
        )
        if configured != [cpu] or effective != [cpu]:
            raise RuntimeError(
                f"{iface} IRQ {irq} affinity verification failed: requested CPU "
                f"{cpu}, configured={format_cpu_list(configured) or 'none'}, "
                f"effective={format_cpu_list(effective) or 'none'}"
            )
        print(f"  {iface}: IRQ {irq} ({label}) -> CPU {cpu}")
    return irq_offset + len(irqs)


def apply_nic_irq_isolation(plan: IsolationPlan) -> None:
    nic = plan.nic_irq
    if nic.mode == "off":
        return
    if not shutil.which("ethtool"):
        raise RuntimeError("ethtool is required for NIC IRQ isolation")

    print("Configuring NIC IRQ isolation...")
    print(f"  Mode:          {nic.mode}")
    print(f"  IRQ CPUs:      {format_cpu_list(nic.irq_cpus)}")
    print(f"  Excluded CPUs: {format_cpu_list(nic.excluded_cpus)}")
    print(f"  Interfaces:    {', '.join(nic.interfaces)}")
    stop_irqbalance_for_nic(nic)
    irq_offset = 0
    for iface in nic.interfaces:
        irq_offset = configure_iface_nic_irq(iface, nic, irq_offset=irq_offset)


def reset_nic_irq_isolation(plan: IsolationPlan) -> None:
    nic = plan.nic_irq
    if nic.mode == "off":
        return
    print("Resetting NIC IRQ isolation...")
    for iface in nic.interfaces:
        original = nic.original_combined.get(iface)
        if original is not None:
            result = subprocess.run(
                ["ethtool", "-L", iface, "combined", str(original)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                print(
                    f"Warning: failed to restore {iface} combined={original}: "
                    f"{result.stderr.strip() or result.stdout.strip()}",
                    file=sys.stderr,
                )
        subprocess.run(
            ["ethtool", "-X", iface, "default"],
            capture_output=True,
            text=True,
            check=False,
        )
        for path in Path(f"/sys/class/net/{iface}/queues").glob("tx-*/xps_cpus"):
            try:
                path.write_text("0\n", encoding="ascii")
            except OSError as exc:
                print(f"Warning: failed to clear {path}: {exc}", file=sys.stderr)
        all_spec = format_cpu_list(all_cpu_ids())
        for irq, _label in get_iface_irqs(iface):
            try:
                Path(f"/proc/irq/{irq}/smp_affinity_list").write_text(
                    all_spec + "\n", encoding="ascii"
                )
            except OSError as exc:
                print(f"Warning: failed to reset IRQ {irq}: {exc}", file=sys.stderr)

    if nic.irqbalance_stopped and shutil.which("systemctl"):
        subprocess.run(
            ["systemctl", "start", "irqbalance"],
            capture_output=True,
            text=True,
            check=False,
        )


def apply_isolation(plan: IsolationPlan) -> None:
    print("------------------------------------------------")
    print("Configuring system isolation...")
    print(f"  Mode:            {plan.mode}")
    if plan.yaml_files:
        print(f"  YAML sources:    {', '.join(plan.yaml_files)}")
    print(f"  Reserved (app):  {format_cpu_list(plan.reserved)}")
    if plan.smt_siblings_isolated:
        print(f"  SMT shielded:    {format_cpu_list(plan.smt_sibling_cpus)}")
    print(f"  System slices:   {format_cpu_list(plan.system)}")
    print(f"  Process (run):   {format_cpu_list(plan.process)}")
    if plan.nic_irq.mode != "off":
        print(f"  NIC IRQ mode:    {plan.nic_irq.mode}")
        print(f"  NIC IRQ CPUs:    {format_cpu_list(plan.nic_irq.irq_cpus)}")
        print(f"  NIC excluded:    {format_cpu_list(plan.nic_irq.excluded_cpus)}")
        print(f"  NIC interfaces:  {', '.join(plan.nic_irq.interfaces)}")
        if plan.nic_irq.combined_queues is not None:
            print(f"  NIC combined:    {plan.nic_irq.combined_queues} (explicit)")
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

    previous = load_saved_plan()
    if previous is not None and previous.nic_irq.mode != "off":
        removed_ifaces = set(previous.nic_irq.interfaces) - set(
            plan.nic_irq.interfaces
        )
        if plan.nic_irq.mode == "off" or removed_ifaces:
            reset_nic_irq_isolation(previous)

    apply_nic_irq_isolation(plan)

    for unit in ("system.slice", "user.slice", "init.scope"):
        run_systemctl_set_allowed(unit, plan.system)

    APP_CPUS_CONF.write_text(format_cpu_list(plan.reserved) + "\n", encoding="utf-8")
    STATE_JSON.write_text(json.dumps(plan.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"Saved reserved CPU config to {APP_CPUS_CONF}")
    print(f"Saved full isolation state to {STATE_JSON}")
    print("Setup successful.")
    print()
    print("Launch BS/UE with the saved application CPU affinity:")
    print("  sudo ../scripts/isolate_cpus.py run ./BS")
    print("  sudo ../scripts/isolate_cpus.py run ./UE")
    print()
    print("Critical threads should remain pinned to reserved cores via YAML;")
    print("non-critical threads can schedule on system cores.")


def reset_isolation() -> None:
    n = total_cpu_count()
    saved_plan = load_saved_plan()
    if saved_plan is not None:
        reset_nic_irq_isolation(saved_plan)
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
            nic_raw = raw.get("nic_irq", {})
            if not isinstance(nic_raw, dict):
                nic_raw = {}
            return IsolationPlan(
                reserved=[int(x) for x in raw.get("reserved_cpus", [])],
                system=[int(x) for x in raw.get("system_cpus", [])],
                process=[int(x) for x in raw.get("process_cpus", [])] or all_cpu_ids(),
                smt_siblings_isolated=bool(
                    raw.get("smt_siblings_isolated", False)
                ),
                smt_sibling_cpus=[
                    int(x) for x in raw.get("smt_sibling_cpus", [])
                ],
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
                nic_irq=NicIrqPlan(
                    mode=(
                        "isolate"
                        if str(nic_raw.get("mode", "off")) == "split"
                        else str(nic_raw.get("mode", "off"))
                    ),
                    isolated_cpus=[int(x) for x in nic_raw.get("isolated_cpus", [])],
                    irq_cpus=[int(x) for x in nic_raw.get("irq_cpus", [])],
                    excluded_cpus=[
                        int(x) for x in nic_raw.get("excluded_cpus", [])
                    ],
                    interfaces=[str(x) for x in nic_raw.get("interfaces", [])],
                    original_combined={
                        str(key): int(value)
                        for key, value in dict(
                            nic_raw.get("original_combined", {})
                        ).items()
                    },
                    irqbalance_stopped=bool(
                        nic_raw.get("irqbalance_stopped", False)
                    ),
                ),
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
        scope_note = "all application CPUs"
        exclusions: list[str] = []
        if plan is not None and plan.smt_siblings_isolated:
            exclusions.append(
                f"shielded SMT siblings {format_cpu_list(plan.smt_sibling_cpus)}"
            )
        if plan is not None and plan.nic_irq.mode != "off":
            exclusions.append(
                "NIC IRQ physical-core CPUs "
                f"{format_cpu_list(plan.nic_irq.excluded_cpus)}"
            )
        if exclusions:
            scope_note += "; " + ", ".join(exclusions) + " excluded"
        print(
            f"Starting process with AllowedCPUs={format_cpu_list(process_cpus)} "
            f"({scope_note}; reserved/critical set was {reserved_hint})..."
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
    print(
        "SMT siblings:  "
        + (
            f"isolated ({format_cpu_list(plan.smt_sibling_cpus)})"
            if plan.smt_siblings_isolated
            else "shared"
        )
    )
    print(f"System:        {format_cpu_list(plan.system)}")
    print(f"Process(run):  {format_cpu_list(plan.process)}")
    if plan.nic_irq.mode != "off":
        print(f"NIC IRQ mode:  {plan.nic_irq.mode}")
        print(f"NIC IRQ CPUs:  {format_cpu_list(plan.nic_irq.irq_cpus)}")
        print(f"NIC excluded:  {format_cpu_list(plan.nic_irq.excluded_cpus)}")
        print(f"NIC ifaces:    {', '.join(plan.nic_irq.interfaces)}")
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
    print(
        "SMT siblings: "
        + (
            f"isolated ({format_cpu_list(plan.smt_sibling_cpus)})"
            if plan.smt_siblings_isolated
            else "shared"
        )
    )
    print(f"System:       {format_cpu_list(plan.system)}")
    print(f"Process(run): {format_cpu_list(plan.process)}")
    if plan.nic_irq.mode != "off":
        print(f"NIC IRQ:      {plan.nic_irq.mode}")
        print(f"NIC IRQ CPUs: {format_cpu_list(plan.nic_irq.irq_cpus)}")
        print(f"NIC excluded: {format_cpu_list(plan.nic_irq.excluded_cpus)}")
        print(f"NIC ifaces:   {', '.join(plan.nic_irq.interfaces)}")
        if plan.nic_irq.combined_queues is not None:
            print(f"NIC combined: {plan.nic_irq.combined_queues} (explicit)")
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
            "optionally distribute NIC IRQs across isolated logical CPUs, and run BS/UE with "
            "application-wide process affinity."
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
  sudo isolate_cpus.py --role bs --nic-irq isolate --nic-irq-cpu-count 2
  sudo isolate_cpus.py --role both --nic-irq off
  sudo isolate_cpus.py --role bs --smt-siblings isolate
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
        "--smt-siblings",
        choices=("ask", "isolate", "share"),
        default="ask",
        help=(
            "Policy for logical SMT siblings of reserved critical CPUs: ask "
            "(TTY only), isolate them from system/application scheduling, or share"
        ),
    )
    parser.add_argument(
        "--nic-irq",
        choices=("ask", "off", "isolate"),
        default="ask",
        help=(
            "NIC IRQ policy when kernel-isolated CPUs exist: ask (TTY only), "
            "off, or isolate (preserve queues and distribute IRQs across the "
            "selected isolated logical CPUs)"
        ),
    )
    parser.add_argument(
        "--nic-irq-cpu-count",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of available kernel-isolated logical CPUs (including SMT "
            "threads) to use for NIC IRQs; interactive setup asks when omitted"
        ),
    )
    parser.add_argument(
        "--nic-ifaces",
        default="",
        help=(
            "Optional comma-separated >=10 Gbps NIC allowlist for IRQ isolation; "
            "default auto-detects active high-speed interfaces"
        ),
    )
    parser.add_argument(
        "--nic-combined",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Explicitly set the NIC combined queue count while applying IRQ "
            "isolation; omitted preserves the current count"
        ),
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
    iface_allowlist = [
        item.strip() for item in str(args.nic_ifaces or "").split(",") if item.strip()
    ]

    def finalize(plan: IsolationPlan) -> IsolationPlan:
        plan = prepare_smt_sibling_plan(
            plan,
            requested_mode=args.smt_siblings,
            interactive=interactive,
        )
        return prepare_nic_irq_plan(
            plan,
            requested_mode=args.nic_irq,
            requested_combined=args.nic_combined,
            requested_irq_cpu_count=args.nic_irq_cpu_count,
            interactive=interactive,
            iface_allowlist=iface_allowlist,
        )

    if args.yaml:
        paths = resolve_role_yaml_paths("both", explicit_yaml=args.yaml)
        # If --role is also given with multiple yamls, just use files as-is.
        role = normalize_machine_role(args.role) if args.role else None
        return finalize(
            plan_from_yaml_files(
                paths,
                side_override=side_override,
                role=role,
            )
        )

    # Positional manual CPU spec (legacy CLI): isolate_cpus.py 4 / 0,2,4 / 8-15
    if args.command and _looks_like_cpu_spec(args.command):
        return finalize(plan_from_manual_spec(args.command))

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
    return finalize(
        plan_from_yaml_files(paths, side_override=side_override, role=role)
    )


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
            plan = prepare_smt_sibling_plan(
                plan,
                requested_mode=args.smt_siblings,
                interactive=True,
            )
            iface_allowlist = [
                item.strip()
                for item in str(args.nic_ifaces or "").split(",")
                if item.strip()
            ]
            plan = prepare_nic_irq_plan(
                plan,
                requested_mode=args.nic_irq,
                requested_combined=args.nic_combined,
                requested_irq_cpu_count=args.nic_irq_cpu_count,
                interactive=True,
                iface_allowlist=iface_allowlist,
            )
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
