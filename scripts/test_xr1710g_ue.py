#!/usr/bin/env python3
"""CPU BS -> XR1710G ABI fixture -> CPU UE -> UDP, including RX-only recovery.

Uses only uniquely named local IPC, ephemeral ports and temporary YAML files.
Run after building BS, UE and Xr1710gBsBridge; no hardware is accessed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import socket
import subprocess
import time
import uuid

import yaml

ROOT = Path(__file__).resolve().parents[1]


def port(kind: int) -> int:
    with socket.socket(socket.AF_INET, kind) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-xr1710g")
    args = parser.parse_args()
    build = args.build_dir.resolve()
    session = "xr1710g_e2e_" + uuid.uuid4().hex[:12]
    work = build / session
    work.mkdir()
    bs = yaml.safe_load((ROOT / "config/BS_Sim.yaml").read_text())
    ue = yaml.safe_load((ROOT / "config/UE_XR1710G.yaml").read_text())
    bs["simulation"].update(session=session, enable_sensing_rx=False, enable_uplink=False)
    bs["rf_sampling"].update(sample_rate=40000000, bandwidth=20000000)
    bs["downlink"]["center_freq"] = 2442000000
    bs["sensing"]["rx_channel_count"] = 0
    bs["sensing"]["rx_channels"] = []
    bs["uplink"]["enabled"] = False
    bs["usrp_device"].update(clock_source="internal", time_source="internal")
    bs["cpu_cores"] = {"downlink_cpu_cores": [], "main_cpu_core": -1}
    bs["network_output"]["udp_input_ip"] = "127.0.0.1"
    bs["network_output"]["udp_input_port"] = port(socket.SOCK_DGRAM)
    ue["network_output"]["udp_output_port"] = port(socket.SOCK_DGRAM)
    # UE binds viewer publishers even when no viewer is attached.
    for key in ("channel_port", "pdf_port", "constellation_port"):
        ue["network_output"][key] = port(socket.SOCK_STREAM)
    for cfg in (bs, ue):
        cfg["network_output"]["control_port"] = port(socket.SOCK_STREAM)
        cfg["logging"] = {"default_level": "warn", "modules": {"sync": "info", "radio": "info"}}
    ue["radio"]["xr1710g_shm"] = "/" + session
    # Exercise the existing no-uplink reacquisition with a shorter hold in this
    # synthetic test. The production preset retains its normal 0.5 second hold.
    ue["sync_tracking"]["reset_hold_s"] = 0.05
    for name, cfg in (("BS", bs), ("UE", ue)):
        (work / (name + ".yaml")).write_text(yaml.safe_dump(cfg, sort_keys=False))
    processes: list[subprocess.Popen] = []
    logs = []
    result = {"directory": str(work), "cfo_hz": 5000, "uplink": False, "phases": {}}
    try:
        def start(name: str, argv: list[str]) -> None:
            log = (work / (name + ".log")).open("w")
            logs.append(log)
            processes.append(subprocess.Popen(argv, cwd=work, stdout=log, stderr=subprocess.STDOUT))
        start("bridge", [str(build / "Xr1710gBsBridge"), session, "/"+session, str(work)])
        time.sleep(.2)
        start("bs", [str(build / "BS")])
        start("ue", [str(build / "UE")])
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as tx, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as rx:
            rx.bind(("127.0.0.1", ue["network_output"]["udp_output_port"]))
            rx.settimeout(.1)
            for phase in ("initial", "loss", "restart"):
                if phase != "initial":
                    (work / phase).touch()
                    deadline = time.monotonic()+5
                    while not (work / (phase+".done")).exists():
                        if time.monotonic() > deadline:
                            raise RuntimeError("fixture did not inject " + phase)
                        time.sleep(.02)
                    time.sleep(.3)
                payload = (session + ":" + phase + ":").encode() + bytes(range(256))
                received = 0
                began = time.monotonic()
                while time.monotonic()-began < 35 and received < 5:
                    for process in processes:
                        if process.poll() is not None:
                            raise RuntimeError(f"child exited {process.returncode}; inspect {work}")
                    tx.sendto(payload, ("127.0.0.1", bs["network_output"]["udp_input_port"]))
                    try:
                        data = rx.recv(65536)
                    except socket.timeout:
                        continue
                    if data == payload:
                        received += 1
                if received < 5:
                    raise RuntimeError(f"{phase}: only {received} matching decoded payloads; inspect {work}")
                result["phases"][phase] = {"matching_payloads": received, "seconds": time.monotonic()-began}
                print(json.dumps({phase: result["phases"][phase]}), flush=True)
        log_text = (work / "ue.log").read_text()
        if "uplink DISABLED" not in log_text or log_text.count("Sync found at pos:") < 3:
            raise RuntimeError("expected receive-only initial sync and two reacquisitions")
        if "metadata error OVERFLOW" not in log_text or "metadata error BROKEN_CHAIN" not in log_text:
            raise RuntimeError("missing injected discontinuity metadata")
        result["sync_acquisitions"] = log_text.count("Sync found at pos:")
        result["passed"] = True
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for process in reversed(processes):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
        (work / "result.json").write_text(json.dumps(result, indent=2)+"\n")
        print("Evidence:", work, flush=True)


if __name__ == "__main__":
    main()
