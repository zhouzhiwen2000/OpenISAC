#!/usr/bin/env python3
"""Exercise the real UDP tester through deterministic loss/corruption/reordering."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time

import zmq


def port(kind=socket.SOCK_DGRAM):
    with socket.socket(socket.AF_INET, kind) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def endpoint():
    return f"tcp://127.0.0.1:{port(socket.SOCK_STREAM)}"


def snapshots(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.startswith("{")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path(__file__).resolve().parents[1] / "build/UdpLinkTest")
    args = parser.parse_args()
    binary = str(args.binary.resolve())
    with tempfile.TemporaryDirectory(prefix="openisac-udp-test-") as directory:
        root = Path(directory)
        ingress = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ingress.bind(("127.0.0.1", 0))
        ingress.settimeout(.1)
        tx_port = ingress.getsockname()[1]
        rx_port = port()
        tx_pub, rx_pub, control = endpoint(), endpoint(), endpoint()
        done = threading.Event()
        drop = {i for i in range(200) if i % 10 == 0} | {199}
        expected_errors = sum((2 if i % 7 == 0 else 0) + (1 if i % 11 == 0 else 0)
                              for i in range(200) if i not in drop)

        def proxy():
            held = None
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as output:
                while not done.is_set():
                    try:
                        raw = bytearray(ingress.recv(65536))
                    except socket.timeout:
                        continue
                    seq = int.from_bytes(raw[16:24], "little")
                    if seq in drop:
                        continue
                    if seq % 7 == 0:
                        raw[-1] ^= 3
                    if seq % 11 == 0:
                        raw[0] ^= 1
                    if seq == 151:
                        for offset in (0, 32, 64):
                            raw[offset] ^= 1
                    if seq % 13 == 0 and held is None:
                        held = bytes(raw)
                        continue
                    output.sendto(raw, ("127.0.0.1", rx_port))
                    if seq % 9 == 0:
                        output.sendto(raw, ("127.0.0.1", rx_port))
                    if held is not None:
                        output.sendto(held, ("127.0.0.1", rx_port))
                        held = None

        thread = threading.Thread(target=proxy, daemon=True)
        thread.start()
        processes = []
        try:
            with (root / "rx.jsonl").open("w") as rx_log, (root / "tx.jsonl").open("w") as tx_log:
                rx = subprocess.Popen([binary, "rx", "--port", str(rx_port), "--telemetry", rx_pub,
                    "--sender-telemetry", tx_pub, "--session", "1234", "--loss-grace", ".2", "--duration", "4"],
                    stdout=rx_log, stderr=subprocess.PIPE, text=True)
                processes.append(rx)
                time.sleep(.3)
                tx = subprocess.Popen([binary, "tx", "--port", str(tx_port), "--telemetry", tx_pub,
                    "--control", control, "--session", "1234", "--rate-mbps", "2", "--count", "200"],
                    stdout=tx_log, stderr=subprocess.PIPE, text=True)
                processes.append(tx)
                _, error = tx.communicate(timeout=8)
                assert tx.returncode == 0, error
                _, error = rx.communicate(timeout=8)
                assert rx.returncode == 0, error
            final = snapshots(root / "rx.jsonl")[-1]
            assert final["unique_packets"] == 200 - len(drop) - 1, final
            assert final["confirmed_sent_packets"] == 200, final
            assert final["lost_packets"] == len(drop) + 1, final
            assert final["bit_errors"] == expected_errors, final
            assert final["compared_bits"] == (200 - len(drop) - 1) * 1316 * 8, final
            assert final["received_bytes"] == (200 - len(drop)) * 1316, final
            assert final["duplicate_packets"] > 0 and final["reordered_packets"] > 0, final
            assert final["invalid_packets"] == 1, final
            assert any(s["rx_mbps"] > 0 for s in snapshots(root / "rx.jsonl")), final
            print("Known loss/corruption/reordering:", json.dumps(final))

            # Live rate control and pause are exercised independently of the GUI.
            tx_pub, control = endpoint(), endpoint()
            context = zmq.Context()
            sub = context.socket(zmq.SUB)
            sub.setsockopt(zmq.SUBSCRIBE, b"")
            sub.connect(tx_pub)
            request = context.socket(zmq.REQ)
            request.connect(control)
            with (root / "control.jsonl").open("w") as log:
                tx = subprocess.Popen([binary, "tx", "--port", str(tx_port), "--telemetry", tx_pub,
                    "--control", control, "--rate-mbps", "2", "--duration", "4"], stdout=log)
                processes.append(tx)
                time.sleep(.4)
                for command in ("RATE 8", "RATE 0"):
                    request.send_string(command)
                    assert request.poll(1500), "control response timeout"
                    assert request.recv_json()["ok"]
                    time.sleep(.6)
                tx.wait(timeout=6)
            data = snapshots(root / "control.jsonl")
            assert any(x["target_mbps"] == 8 and x["tx_mbps"] > 4 for x in data), data
            assert any(x["target_mbps"] == 0 and x["tx_mbps"] == 0 for x in data), data
            sub.close(0); request.close(0); context.term()
            print("Live rate change and pause passed")
        finally:
            done.set()
            thread.join(timeout=1)
            ingress.close()
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=3)


if __name__ == "__main__":
    main()
