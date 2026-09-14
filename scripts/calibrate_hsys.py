#!/usr/bin/env python3
"""Capture and verify a full-band Hsys calibration using the running backend.

Inspired by Henry Teall <henry.teall@paconsulting.com>, philippaPA/OpenISAC
commit 9b630f3744a2261b234d22f95bfe6e07b727d4d8.

Set suitable gains before starting the backend and connecting the attenuated
RF loopback. This tool never changes gains. Close other control clients during
capture: ALCH is backend-global. A timeout does not cancel backend capture;
stop the backend before changing the RF connection after an unsuccessful run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time
import uuid

import numpy as np
import yaml
import zmq

from sensing_runtime_protocol import (
    FLAG_BISTATIC, build_control_command, build_params_request,
    make_control_dealer, make_tcp_endpoint, parse_params_packet,
)

# The backend writes a packed, native-endian header and native complex<float>.
HSYS_HEADER = struct.Struct("=8s16s4I3dQ")
ROLE_IDS = {"monostatic": 1, "bistatic": 2}


def positive_int(text: str) -> int:
    value = int(text)
    if not 0 < value <= 2**31 - 1:
        raise argparse.ArgumentTypeError("must be in 1..2147483647")
    return value


def positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def inspect_response(path: Path) -> dict:
    """Validate a native-endian Hsys file and summarize its numerical quality."""
    raw = path.read_bytes()
    if len(raw) < HSYS_HEADER.size:
        raise ValueError("truncated Hsys header")
    magic, mode, version, role_id, channel, fft, rate, freq, bandwidth, symbols = (
        HSYS_HEADER.unpack_from(raw))
    role = mode.split(b"\0", 1)[0].decode("ascii")
    if magic != b"OISACH1\0" or version != 1 or ROLE_IDS.get(role) != role_id:
        raise ValueError("unsupported Hsys header, role or byte order")
    if fft == 0 or symbols == 0 or len(raw) != HSYS_HEADER.size + fft * 8:
        raise ValueError("invalid Hsys dimensions, symbol count or file size")
    if not all(math.isfinite(v) and v > 0 for v in (rate, freq, bandwidth)):
        raise ValueError("invalid Hsys RF metadata")
    response = np.frombuffer(raw, dtype=np.complex64, offset=HSYS_HEADER.size)
    power = np.abs(response.astype(np.complex128)) ** 2
    if (not np.all(np.isfinite(power)) or not np.any(power > 0)
            or np.any(power > np.finfo(np.float32).max)):
        raise ValueError("nonfinite or zero Hsys response")
    threshold = max(float(power.max()) * 1e-6, 1e-20)
    weak = int(np.count_nonzero(power < threshold))
    return dict(path=str(path), role=role, channel=channel, fft_size=fft,
                sample_rate_hz=rate, center_freq_hz=freq, bandwidth_hz=bandwidth,
                captured_symbols=symbols, weak_bins=weak, total_bins=fft,
                mean_power=float(power.mean()),
                magnitude_p05_p50_p95=np.percentile(np.sqrt(power), [5, 50, 95]).tolist(),
                sha256=hashlib.sha256(raw).hexdigest(), byte_order=sys.byteorder)


def load_profile(directory: Path, role: str, channel: int) -> dict:
    """Read the actual backend working-directory YAML; never rewrite it."""
    config = directory / ("BS.yaml" if role == "monostatic" else "UE.yaml")
    cfg = yaml.safe_load(config.read_text())
    sensing = cfg["sensing"]
    if channel < 0:
        raise ValueError("channel is a zero-based logical sensing channel, never -1/all")
    if role == "monostatic":
        channels = sensing["rx_channels"]
        count = int(sensing.get("rx_channel_count", len(channels)))
        if channel >= min(count, len(channels)):
            raise ValueError("channel is not present in the backend YAML")
        if channels[channel].get("enable_system_delay_estimation", False):
            raise ValueError("finish system-delay calibration and disable estimation first")
    elif channel != 0 or not sensing.get("bi_enabled", False):
        raise ValueError("bistatic calibration requires bi_enabled and logical channel 0")
    profile = dict(fft_size=int(cfg["ofdm"]["fft_size"]),
                sample_rate_hz=float(cfg["rf_sampling"]["sample_rate"]),
                center_freq_hz=float(cfg["downlink"]["center_freq"]),
                bandwidth_hz=float(cfg["rf_sampling"]["bandwidth"]),
                control_port=int(cfg["network_output"]["control_port"]))
    if profile["fft_size"] <= 0 or not all(
        math.isfinite(profile[key]) and profile[key] > 0
        for key in ("sample_rate_hz", "center_freq_hz", "bandwidth_hz")
    ):
        raise ValueError("backend YAML has invalid FFT or RF geometry")
    return profile


def check_params(params, profile: dict, role: str) -> None:
    if bool(params.flags & FLAG_BISTATIC) != (role == "bistatic"):
        raise ValueError("connected backend role does not match --role")
    if params.ofdm_fft_size != profile["fft_size"]:
        raise ValueError("backend FFT size differs from the working-directory YAML")
    # V9 transports RF values in units of 100 Hz.
    for key in ("sample_rate_hz", "center_freq_hz"):
        if abs(getattr(params, key) - profile[key]) > 100:
            raise ValueError(f"backend {key} differs from the working-directory YAML")
    if params.is_compact_mask() and (
        not params.compact_local_delay_doppler_supported()
        or params.active_cols != params.ofdm_fft_size
    ):
        raise ValueError("Hsys requires dense mode or a full-band regular compact mask")


def request_params(sock, timeout: float):
    """Wait for an exact V9 reply; readiness alone does not prove compatibility."""
    sock.send(build_params_request())
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not sock.poll(max(1, min(100, int((deadline - time.monotonic()) * 1000)))):
            continue
        parts = sock.recv_multipart()
        if len(parts) != 1 or parts[0][:8] != b"CTRLPARM":
            continue
        params = parse_params_packet(parts[0])
        if params is None:
            raise ValueError("backend PARM is incompatible; update backend and tools together")
        return params
    raise TimeoutError("no V9 PARM response from backend")


def file_stamp(path: Path):
    try:
        st = path.stat()
        return st.st_ino, st.st_mtime_ns, st.st_ctime_ns, st.st_size
    except FileNotFoundError:
        return None


def wait_for_response(path: Path, previous, profile: dict, role: str,
                      channel: int, symbols: int, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    reason = "no newly saved file"
    while time.monotonic() < deadline:
        stamp = file_stamp(path)
        if stamp is not None and stamp != previous:
            try:
                result = inspect_response(path)
                if file_stamp(path) != stamp:
                    raise ValueError("file is still being written")
                if result["role"] != role or result["channel"] != channel:
                    raise ValueError("saved file role/channel does not match request")
                if result["captured_symbols"] != symbols:
                    raise ValueError("saved symbol count does not match request")
                for key in ("fft_size", "sample_rate_hz", "center_freq_hz", "bandwidth_hz"):
                    if not math.isclose(result[key], profile[key], rel_tol=1e-9):
                        raise ValueError(f"saved {key} differs from backend YAML")
                return result
            except (ValueError, OSError) as exc:
                reason = str(exc)
        time.sleep(0.05)
    raise TimeoutError(f"calibration not verified: {reason}")


def capture(args) -> dict:
    directory = args.backend_dir.resolve(strict=True)
    profile = load_profile(directory, args.role, args.channel)
    port = args.control_port if args.control_port is not None else profile["control_port"]
    if not 1 <= port <= 65535:
        raise ValueError("control port must be in 1..65535")
    output = directory / f"sensing_system_response_{args.role}_ch{args.channel}.bin"
    endpoint = make_tcp_endpoint(args.host, port)
    sock = make_control_dealer(endpoint, identity=f"calibrate-hsys-{uuid.uuid4().hex}")
    sock.setsockopt(zmq.SNDTIMEO, int(args.connect_timeout * 1000))
    sock.setsockopt(zmq.IMMEDIATE, 1)
    try:
        check_params(request_params(sock, args.connect_timeout), profile, args.role)
        print(f"Backend: {endpoint}; {args.role} logical CH{args.channel}; output: {output}")
        print("Use the actual backend working directory. Close other control clients.\n"
              "Set appropriate gains before starting; connect a suitably attenuated loopback.\n"
              "This tool does not change or restore TX/RX gain. Gain 0 dB does not mute TX.")
        if not args.yes and input("Type CALIBRATE when this setup is ready: ").strip() != "CALIBRATE":
            raise ValueError("calibration not requested")
        # Recheck connectivity after an arbitrarily long interactive preparation.
        check_params(request_params(sock, args.connect_timeout), profile, args.role)
        if output.exists():
            backup = output.with_name(output.name + f".{uuid.uuid4().hex}.bak")
            with backup.open("xb") as target:
                target.write(output.read_bytes())
            print(f"Previous calibration backed up to {backup}")
        previous = file_stamp(output)
        if args.role == "monostatic":
            sock.send(build_control_command(b"ALCH", args.channel))
        sock.send(build_control_command(b"CALB", args.symbols))
        print("CALB requested; waiting for a new, complete matching file (not a command acknowledgement).")
        result = wait_for_response(output, previous, profile, args.role, args.channel,
                                   args.symbols, args.timeout)
        result["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["endpoint"] = endpoint
        print("Saved file verified. Stop the backend before changing RF connections or restoring gains.")
        return result
    finally:
        sock.close(linger=0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="action", required=True)
    run = modes.add_parser("capture", help="request calibration and verify the saved file")
    run.add_argument("--backend-dir", type=Path, required=True,
                     help="actual backend cwd with BS.yaml/UE.yaml and calibration files (shared mount allowed)")
    run.add_argument("--role", choices=ROLE_IDS, default="monostatic")
    run.add_argument("--channel", type=int, required=True, help="zero-based logical sensing channel")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--control-port", type=int, help="default: read network_output.control_port from YAML")
    run.add_argument("--symbols", type=positive_int, default=1000)
    run.add_argument("--timeout", type=positive_float, default=60.0)
    run.add_argument("--connect-timeout", type=positive_float, default=5.0)
    run.add_argument("--yes", action="store_true", help="confirm RF setup without an interactive prompt")
    inspect = modes.add_parser("inspect", help="validate an existing file without sending commands")
    inspect.add_argument("file", type=Path)
    args = parser.parse_args(argv)
    try:
        result = capture(args) if args.action == "capture" else inspect_response(args.file)
        print(json.dumps(result, indent=2, allow_nan=False))
        if result["weak_bins"]:
            print("Note: weak bins are left unequalized by the backend; check occupied-band coverage.")
        return 0
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError, TimeoutError, zmq.ZMQError) as exc:
        print(f"Calibration failed: {exc}", file=sys.stderr)
    except (KeyboardInterrupt, EOFError):
        print("Calibration interrupted.", file=sys.stderr)
    if args.action == "capture":
        print("Completion was not verified. Capture may still be active; stop the backend before changing RF wiring/gains.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
