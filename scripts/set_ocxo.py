#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import termios
import time


MAX_DAC_VALUE = (1 << 20) - 1
MID_DAC_VALUE = MAX_DAC_VALUE // 2
OCXO_FREQ_RANGE = 4.0e-7


def ppm_to_ocxo_control_word(ppm: float) -> int:
    ratio = ppm * 1e-6
    fraction = ratio / (2.0 * OCXO_FREQ_RANGE)
    dac_value = MID_DAC_VALUE + fraction * MAX_DAC_VALUE
    return max(0, min(MAX_DAC_VALUE, int(round(dac_value))))


def build_ocxo_set_frequency_command(control_word: int) -> bytes:
    control_word = max(0, min(MAX_DAC_VALUE, int(control_word)))
    payload = f"SF{control_word:07d}"
    checksum = 0
    for ch in payload:
        checksum ^= ord(ch)
    return f"{payload}*{checksum:03d}\n".encode("ascii")


def configure_serial(fd: int) -> None:
    attrs = termios.tcgetattr(fd)
    attrs[4] = termios.B57600
    attrs[5] = termios.B57600
    attrs[2] |= termios.CLOCAL | termios.CREAD
    attrs[2] &= ~termios.PARENB
    attrs[2] &= ~termios.CSTOPB
    attrs[2] &= ~termios.CSIZE
    attrs[2] |= termios.CS8
    attrs[3] &= ~(termios.ICANON | termios.ECHO | termios.ECHOE | termios.ISIG)
    attrs[1] &= ~termios.OPOST
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, attrs)


def set_ocxo_control_word(device_path: str, control_word: int, response_timeout_s: float) -> bytes:
    control_word = max(0, min(MAX_DAC_VALUE, int(control_word)))
    command = build_ocxo_set_frequency_command(control_word)
    fd = os.open(device_path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        configure_serial(fd)
        written = 0
        while written < len(command):
            try:
                n = os.write(fd, command[written:])
            except BlockingIOError:
                time.sleep(0.005)
                continue
            if n <= 0:
                time.sleep(0.005)
                continue
            written += n

        deadline = time.monotonic() + response_timeout_s
        response = bytearray()
        while time.monotonic() < deadline:
            try:
                chunk = os.read(fd, 31)
            except BlockingIOError:
                chunk = b""
            if chunk:
                response.extend(chunk)
                if b"\n" in response:
                    break
            else:
                time.sleep(0.01)
        return bytes(response)
    finally:
        os.close(fd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set the OCXO DAC control word over the serial interface."
    )
    parser.add_argument("--tty", default="/dev/ttyUSB0", help="OCXO serial device.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--word", type=int, help="Raw 20-bit OCXO DAC control word.")
    group.add_argument("--ppm", type=float, help="Nominal ppm converted with the built-in DAC formula.")
    parser.add_argument("--timeout", type=float, default=0.5, help="Serial response timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without writing serial data.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    control_word = args.word if args.word is not None else ppm_to_ocxo_control_word(args.ppm)
    control_word = max(0, min(MAX_DAC_VALUE, int(control_word)))
    command = build_ocxo_set_frequency_command(control_word)

    if args.dry_run:
        print(f"tty={args.tty}")
        print(f"control_word={control_word}")
        print(f"command={command.decode('ascii').rstrip()}")
        return 0

    response = set_ocxo_control_word(args.tty, control_word, args.timeout)
    print(f"control_word={control_word}")
    print(f"command={command.decode('ascii').rstrip()}")
    if response:
        print(f"response={response.decode('ascii', errors='replace').rstrip()}")
    else:
        print("response=<timeout>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
