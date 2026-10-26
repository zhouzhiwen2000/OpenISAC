#!/usr/bin/env python3
"""Verify UDP payloads through the local simulator and selectable BS/UE binaries.

Example: python3 scripts/test_modulation_sim.py --bs BS --ue UE
Creates isolated YAML files and logs; never edits build/BS.yaml or build/UE.yaml.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import tempfile
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
# Use a local loopback alias: WSL may route 127.0.0.1 through its Windows proxy,
# which can discard fragmented UDP datagrams before they reach the backend.
MODES = ("BPSK", "QPSK", "8PSK", "16QAM", "32QAM", "64QAM")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--bs", default="BS")
    parser.add_argument("--ue", default="UE")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--cpu-precision", choices=("q16", "float32"), default="q16")
    parser.add_argument("--payload-bytes", type=int, default=1316)
    parser.add_argument("--packets", type=int, default=40)
    args = parser.parse_args()
    if not 8 <= args.payload_bytes <= 65507 or args.packets < 1:
        parser.error("payload bytes must be 8..65507 and packets must be positive")
    build = args.build_dir.resolve()
    for binary in ("ChannelSimulator", args.bs, args.ue):
        if not (build / binary).is_file():
            parser.error(f"missing executable: {build / binary}")
    output = Path(tempfile.mkdtemp(prefix="openisac-modulation-"))
    print(f"Results: {output}", flush=True)
    used_ports: set[int] = set()

    def free_port() -> int:
        while True:
            with socket.socket() as sock:
                # Backend PUB sockets bind all interfaces; also exclude ports
                # occupied by services listening only on another loopback IP.
                sock.bind(("0.0.0.0", 0))
                number = sock.getsockname()[1]
            if all(abs(number - other) >= 5 for other in used_ports):
                used_ports.add(number)
                return number

    results = []
    for index, mode in enumerate(args.modes):
        run = output / f"{index}-{mode}"
        session = f"modulation_test_{os.getpid()}_{index}"
        udp_in, udp_out = free_port(), free_port()
        for role in ("BS", "UE"):
            directory = run / role.lower()
            directory.mkdir(parents=True)
            config = yaml.safe_load((ROOT / "config" / f"{role}_Sim.yaml").read_text())
            config["modulation"] = {"scheme": mode}
            config["rf_sampling"].update(sample_rate=2_000_000, bandwidth=2_000_000)
            config["simulation"].update(
                session=session, enable_sensing_rx=False, enable_uplink=False,
                cfo_hz=0, target_snr_db=40, targets=[], bistatic_targets=[],
                pacing_enabled=True, control_port=free_port(),
            )
            config["ofdm_frame"]["midframe_pilot_symbols"] = []
            config["uplink"].update(enabled=False, symbol_count=0)
            config["cpu_cores"] = {
                "downlink_cpu_cores": [-1] * (4 if role == "BS" else 3),
                "demod_worker_cpu_cores": [-1, -1], "ldpc_worker_cpu_cores": [-1, -1],
                "main_cpu_core": -1, "sensing_cpu_cores": [-1],
            }
            config["ldpc"] = {"fixed_point": args.cpu_precision == "q16", "fixed_point_scale": 16,
                              "decoder_iterations": 12, "nms_normalization_factor": 0.75}
            config["logging"] = {"default_level": "warn", "force_error": True}
            network = config["network_output"]
            for key in network:
                if key.endswith("_port"):
                    network[key] = free_port()
            network.update(udp_input_ip="127.0.0.2", udp_input_port=udp_in,
                           udp_output_ip="127.0.0.2", udp_output_port=udp_out)
            if role == "BS":
                config["sensing"].update(rx_channel_count=0, rx_channels=[])
            else:
                config["sync_tracking"].update(predictive_delay=False,
                                               sync_cfo_alias_search_range_hz=20000, desired_peak_pos=0)
            (directory / f"{role}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        processes: list[subprocess.Popen] = []
        logs = []
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        rx.bind(("127.0.0.2", udp_out))
        rx.setblocking(False)

        def launch(name: str, argv: list[str], directory: Path) -> None:
            log = (run / f"{name}.log").open("w")
            logs.append(log)
            processes.append(subprocess.Popen(argv, cwd=directory, stdout=log,
                                              stderr=subprocess.STDOUT, start_new_session=True))

        def collect() -> list[bytes]:
            if any(process.poll() is not None for process in processes):
                raise RuntimeError(f"backend exited; inspect {run}")
            packets = []
            while True:
                try:
                    packets.append(rx.recv(65536))
                except BlockingIOError:
                    return packets

        try:
            launch("sim", [str(build / "ChannelSimulator"), str(run / "bs/BS.yaml")], run / "bs")
            time.sleep(0.5)
            launch("ue", [str(build / args.ue)], run / "ue")
            time.sleep(0.5)
            launch("bs", [str(build / args.bs)], run / "bs")
            warmup = b"OPENISAC_MODULATION_PROBE" + bytes(range(256)) * 5
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                tx.sendto(warmup, ("127.0.0.2", udp_in))
                time.sleep(0.1)
                if warmup in collect():
                    break
            else:
                raise RuntimeError(f"no decoded warmup packet; inspect {run}")
            expected = {number: struct.pack("<Q", number) +
                        bytes((number + offset * 37) & 255 for offset in range(args.payload_bytes - 8))
                        for number in range(args.packets)}
            received: set[int] = set()
            corrupt = 0

            def record() -> None:
                nonlocal corrupt
                for data in collect():
                    if data == warmup:
                        continue
                    number = struct.unpack_from("<Q", data)[0] if len(data) >= 8 else -1
                    if number in expected and data == expected[number]:
                        received.add(number)
                    else:
                        corrupt += 1

            for data in expected.values():
                tx.sendto(data, ("127.0.0.2", udp_in))
                time.sleep(0.07)
                record()
            deadline = time.monotonic() + 4
            while time.monotonic() < deadline and len(received) < args.packets:
                time.sleep(0.05)
                record()
            result = {"modulation": mode, "sent": args.packets, "received": len(received),
                      "corrupt": corrupt, "logs": str(run)}
            results.append(result)
            (output / "summary.json").write_text(json.dumps({"options": vars(args) | {"build_dir": str(build)},
                                                            "results": results}, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            if len(received) != args.packets or corrupt:
                raise RuntimeError(f"UDP payload comparison failed; inspect {run}")
        finally:
            # Stop only this test's children; the simulator remains alive until the backends stop.
            for process in processes[1:]:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
            for process in processes[1:]:
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            if processes:
                simulator = processes[0]
                if simulator.poll() is None:
                    os.killpg(simulator.pid, signal.SIGINT)
                try:
                    simulator.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(simulator.pid, signal.SIGKILL)
                    simulator.wait()
            for log in logs:
                log.close()
            rx.close()
            tx.close()
            for path in Path("/dev/shm").glob(f"*{session}*"):
                if path.is_file():
                    path.unlink()


if __name__ == "__main__":
    main()
