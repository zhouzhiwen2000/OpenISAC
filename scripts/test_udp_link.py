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
import zlib

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
        captured = {}
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
                    captured[seq] = bytes(raw)
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
                    "--sender-telemetry", tx_pub, "--control", endpoint(),
                    "--session", "1234", "--loss-grace", ".2", "--duration", "4"],
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
            assert final["loss_started"] and final["loss_expected_packets"] == 199, final
            assert final["lost_packets"] == len(drop), final  # Sequence 0 precedes first RX.
            assert final["bit_errors"] == expected_errors, final
            assert final["compared_bits"] == (200 - len(drop) - 1) * 1316 * 8, final
            assert final["received_bytes"] == (200 - len(drop)) * 1316, final
            assert final["duplicate_packets"] > 0 and final["reordered_packets"] > 0, final
            assert final["invalid_packets"] == 1, final
            assert any(s["rx_mbps"] > 0 for s in snapshots(root / "rx.jsonl")), final
            print("Known loss/corruption/reordering:", json.dumps(final))

            # Independent resets, late packets across reset and delayed first RX.
            context = zmq.Context()
            fake_tx = context.socket(zmq.PUB)
            fake_tx.bind(endpoint_address := endpoint())
            rx_control = endpoint()
            request = context.socket(zmq.REQ)
            request.connect(rx_control)
            output = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            path = root / "reset.jsonl"
            try:
                with path.open("w") as log:
                    rx = subprocess.Popen([binary, "rx", "--port", str(rx_port),
                        "--telemetry", endpoint(), "--control", rx_control,
                        "--sender-telemetry", endpoint_address, "--session", "1234",
                        "--loss-grace", ".1"], stdout=log, stderr=subprocess.PIPE, text=True)
                    processes.append(rx)

                    def wait_for(predicate):
                        deadline = time.monotonic() + 3
                        while time.monotonic() < deadline:
                            data = snapshots(path)
                            if data and predicate(data[-1]):
                                return data[-1]
                            time.sleep(.02)
                        raise AssertionError(data[-1] if data else "No RX telemetry")

                    def count(sent):
                        for _ in range(5):
                            fake_tx.send_json(dict(role="tx", session="1234", sent_packets=sent))
                            time.sleep(.05)

                    def packet(seq, flip=0):
                        data = bytearray(captured[seq]); data[-1] ^= flip
                        output.sendto(data, ("127.0.0.1", rx_port))

                    def reset(command):
                        request.send_string(command)
                        assert request.poll(1500), "RX reset response timeout"
                        return request.recv_json()["ok"]

                    time.sleep(.3)
                    count(100)
                    foreign = bytearray(captured[0])
                    foreign[8:16] = (0x9999).to_bytes(8, "little")
                    foreign[28:32] = zlib.crc32(foreign[:28]).to_bytes(4, "little")
                    foreign[32:64] = foreign[64:96] = foreign[:32]
                    output.sendto(foreign, ("127.0.0.1", rx_port))
                    output.sendto(b"invalid", ("127.0.0.1", rx_port))
                    state = wait_for(lambda d: d["foreign_packets"] == 1 and d["invalid_packets"] == 1)
                    assert state["sender_seen"] and not state["loss_started"]
                    assert state["lost_packets"] == state["loss_expected_packets"] == 0
                    # No new sender telemetry: the first RX must use the saved count.
                    packet(99)
                    state = wait_for(lambda d: d["loss_expected_packets"] == 1)
                    assert state["lost_packets"] == 0
                    packet(100); packet(102, 3); count(104)
                    state = wait_for(lambda d: d["loss_expected_packets"] == 5)
                    assert state["lost_packets"] == 2 and state["bit_errors"] == 2
                    assert reset("RESET_LOSS")
                    state = wait_for(lambda d: d["loss_resets"] == 1)
                    assert state["lost_packets"] == state["loss_expected_packets"] == 0
                    assert state["unique_packets"] == 3 and state["bit_errors"] == 2
                    packet(101); packet(104); packet(106, 7); count(108)
                    state = wait_for(lambda d: d["loss_expected_packets"] == 4)
                    assert state["lost_packets"] == 2 and state["bit_errors"] == 5
                    assert state["loss_percent"] == 50
                    assert reset("RESET_ERRORS")
                    cleared = wait_for(lambda d: d["error_resets"] == 1)
                    assert cleared["bit_errors"] == cleared["compared_bits"] == 0
                    assert cleared["lost_packets"] == 2 and cleared["unique_packets"] == 6
                    packet(106, 7); packet(107, 1); packet(103); count(109)
                    state = wait_for(lambda d: d["loss_expected_packets"] == 5)
                    assert state["lost_packets"] == 2 and state["loss_percent"] == 40
                    assert state["bit_errors"] == 1 and state["compared_bits"] == 2 * 1316 * 8
                    assert state["unique_packets"] == 8
                    assert state["received_bytes"] == cleared["received_bytes"] + 2 * 1316
                    assert not reset("UNKNOWN")
                    rx.terminate(); _, error = rx.communicate(timeout=3)
                    assert rx.returncode == 0, error
                    print("First-RX start, independent resets and late recovery passed")
            finally:
                output.close(); fake_tx.close(0); request.close(0); context.term()

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
