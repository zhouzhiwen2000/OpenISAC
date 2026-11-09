#!/usr/bin/env python3
"""Check configured USRPs and repair X3xx compatibility using verified flash backups."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import threading

import yaml

MISMATCH = re.compile(
    r'Major compat number mismatch|Expected (?:FPGA|firmware) compatibility number'
    r'|FPGA component .*?revision .*?UHD supports'
    r'|RFNoC protocol mismatch between SW and HW', re.I | re.S)
MANIFEST = Path('/opt/uhd/share/uhd/openisac-image-sha256.json')


def configured_targets(config: dict, role: str) -> list[str]:
    """Resolve the same active device arguments as BS/UE and sensing RX."""
    if config.get('radio', {}).get('radio_backend', 'uhd').lower() != 'uhd':
        return []
    shared = config.get('usrp_device', {}).get('device_args', '') or ''
    if role == 'UE':
        return [shared]
    tx = config.get('downlink', {}).get('tx_device_args') or shared
    targets = [tx]
    uplink = config.get('uplink', {})
    frame_symbols = int(config.get('ofdm_frame', {}).get('num_symbols', 100))
    if str(uplink.get('duplex_mode', 'tdd')).lower() == 'fdd':
        ul_symbols = frame_symbols
    else:
        start = int(uplink.get('symbol_start', 90))
        count = min(int(uplink.get('symbol_count', 10)), max(0, frame_symbols - start))
        ul_symbols = count - int(uplink.get('guard_symbols', 1))
    if uplink.get('enabled', False) and ul_symbols >= 2:
        targets.append(uplink.get('rx_device_args') or tx)
    sensing = config.get('sensing', {})
    channels = sensing.get('rx_channels', [])
    count = int(sensing.get('rx_channel_count', len(channels) if 'rx_channels' in sensing else 1))
    for i in range(count):
        channel = channels[i] if i < len(channels) else (channels[0] if channels else {})
        targets.append(channel.get('device_args') or sensing.get('rx_device_args') or shared)
    return list(dict.fromkeys(targets))


def run(command: list[str], timeout: int = 90) -> subprocess.CompletedProcess:
    print('+ ' + ' '.join(command), flush=True)
    chunks = []
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
        def copy_output() -> None:
            while chunk := os.read(process.stdout.fileno(), 4096):
                chunks.append(chunk)
                sys.stdout.write(chunk.decode('utf-8', errors='replace'))
                sys.stdout.flush()
        reader = threading.Thread(target=copy_output, daemon=True)
        reader.start()
        try:
            process.wait(timeout=timeout)
        except BaseException:
            process.kill()
            process.wait()
            reader.join()
            raise
        reader.join()
    return subprocess.CompletedProcess(command, process.returncode,
                                       b''.join(chunks).decode('utf-8', errors='replace'))



def probe(args: str) -> subprocess.CompletedProcess:
    return run(['uhd_usrp_probe', '--args', args, '--init-only'])


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def identify(args: str) -> dict:
    result = run(['uhd_find_devices', '--args', args])
    devices = []
    for block in result.stdout.split('Device Address:')[1:]:
        fields = dict(re.findall(r'^[ \t]*(\w+):[ \t]*([^\r\n]*)', block, re.M))
        devices.append({k: v.strip() for k, v in fields.items()})
    if result.returncode or len(devices) != 1:
        raise RuntimeError('Automatic update requires exactly one discoverable device; use an explicit address/serial.')
    device = devices[0]
    if (device.get('type') != 'x300' or device.get('product') not in ('X300', 'X310')
            or device.get('fpga') not in ('HG', 'XG')
            or not re.fullmatch(r'[A-Za-z0-9_-]+', device.get('serial', ''))
            or not re.fullmatch(r'[0-9.]+', device.get('addr', ''))):
        raise RuntimeError('Automatic persistent update supports identified Ethernet X300/X310 HG/XG devices only.')
    return device


def image_for(device: dict) -> Path:
    name = f"usrp_{device['product'].lower()}_fpga_{device['fpga']}.bit"
    image = Path(os.environ.get('UHD_IMAGES_DIR', '/opt/uhd/share/uhd/images')) / name
    hashes = json.loads(MANIFEST.read_text())
    if not image.is_file() or hashes.get(name) != digest(image):
        raise RuntimeError(f'Image is missing or differs from the build-verified UHD release: {image}')
    return image


def read_flash(args: str, image: Path, prefix: Path) -> Path:
    # --download ALONE also writes! Both no-* flags are mandatory for read-only backups.
    result = run(['uhd_image_loader', '--args', args, '--fpga-path', str(image),
                  '--out-path', str(prefix), '--download', '--no-fpga', '--no-fw'], 600)
    prefix.with_suffix('.log').write_text(result.stdout)
    candidates = [p for p in (prefix.with_suffix('.bit'), prefix.with_suffix('.bin')) if p.is_file()]
    if result.returncode or len(candidates) != 1:
        raise RuntimeError('Flash backup failed; no update attempted.')
    path = candidates[0]
    expected_size = 15_878_042 if path.suffix == '.bit' else 15_877_916
    if path.stat().st_size != expected_size:
        raise RuntimeError('Flash backup size is invalid; no update attempted.')
    with path.open('rb') as stream:
        os.fsync(stream.fileno())
    return path


def prepare(args: str, state: Path, auto: bool) -> None:
    initial = probe(args)
    if initial.returncode == 0:
        return
    if not MISMATCH.search(initial.stdout):
        raise RuntimeError('USRP probe failed without a recognized compatibility mismatch; firmware unchanged.')
    if not auto:
        raise RuntimeError('USRP is incompatible; automatic update is disabled (OPENISAC_AUTO_UPDATE_FIRMWARE=0).')
    if re.search(r'(?:^|,)\s*(?:addr|serial)\d+=', args):
        raise RuntimeError('For automatic update, check each motherboard separately using usrp-check --args addr=IP.')
    device = identify(args)
    image = image_for(device)
    if not state.is_dir() or not os.path.ismount(state):
        raise RuntimeError('Flash updates require a persistent bind mount at OPENISAC_FIRMWARE_STATE; use docker_run.sh.')
    root = state / device['serial']
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (root / 'update.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another firmware check/update owns this USRP; retry after it finishes.') from error
        journal = root / 'transaction.json'
        if journal.exists():
            previous = json.loads(journal.read_text())
            if previous.get('status') != 'complete':
                raise RuntimeError(f'An earlier update needs recovery; see {journal}. No repeated flash write attempted.')
        # Another launcher may have repaired the device before we acquired the lock.
        current = probe(args)
        if current.returncode == 0:
            return
        if not MISMATCH.search(current.stdout):
            raise RuntimeError('USRP state changed during preparation; firmware unchanged.')
        directory = root / time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        directory.mkdir()
        (directory / 'probe-before.log').write_text(current.stdout)
        # UHD 4.11 serial filtering accepts only up to eight hexadecimal characters.
        # Some lab radios have non-hex EEPROM serials; address them explicitly and
        # recheck the full discovered identity immediately before any write.
        keys = ['type', 'addr', 'fpga']
        if re.fullmatch(r'[0-9a-fA-F]{1,8}', device['serial']):
            keys.append('serial')
        loader_args = ','.join(f'{key}={device[key]}' for key in keys)
        first = read_flash(loader_args, image, directory / 'before-a')
        second = read_flash(loader_args, image, directory / 'before-b')
        if first.stat().st_size != second.stat().st_size or digest(first) != digest(second):
            raise RuntimeError('Independent flash backups differ; no update attempted.')
        if identify(f"type=x300,addr={device['addr']}") != device:
            raise RuntimeError('Device identity changed after backup; no update attempted.')
        transaction = {'status': 'update_in_progress', 'device': device,
                       'image': str(image), 'image_sha256': digest(image),
                       'backup': str(first), 'backup_sha256': digest(first),
                       'backup_verified_twice': True,
                       'restore_command': ['uhd_image_loader', '--args', loader_args + ',verify,configure',
                                           '--fpga-path', str(first), '--no-fw']}
        save_json(directory / 'backup.json', transaction)
        save_json(journal, transaction)
        # A timeout or interruption leaves the durable journal pending for manual recovery.
        handlers = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            result = run(['uhd_image_loader', '--args', loader_args + ',verify,configure',
                          '--fpga-path', str(image), '--no-fw'], 900)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        (directory / 'update.log').write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(f'Image loader failed; verified backups and recovery command are in {journal}.')
        transaction['status'] = 'awaiting_probe'
        save_json(journal, transaction)
        for _ in range(12):
            time.sleep(5)
            try:
                after = probe(args)
            except subprocess.TimeoutExpired:
                break
            (directory / 'probe-after.log').write_text(after.stdout)
            if after.returncode == 0:
                transaction['status'] = 'complete'
                save_json(journal, transaction)
                print(f"USRP {device['serial']}: update verified; compatible probe passed.", flush=True)
                return
        raise RuntimeError(f'Flash written, but the USRP needs recovery or a power cycle; see {journal}. No repeat write.')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?')
    parser.add_argument('--role', choices=('BS', 'UE'), default='BS')
    parser.add_argument('--args', action='append', dest='device_args')
    opts = parser.parse_args()
    try:
        if opts.device_args is not None:
            targets = opts.device_args
        else:
            config = opts.config or os.environ.get('OPENISAC_CONFIG')
            if not config:
                raise RuntimeError('Supply a YAML file or --args addr=IP.')
            targets = configured_targets(yaml.safe_load(Path(config).read_text()), opts.role)
        state = Path(os.environ.get('OPENISAC_FIRMWARE_STATE', '/work/firmware'))
        auto = os.environ.get('OPENISAC_AUTO_UPDATE_FIRMWARE', '1') != '0'
        for target in targets:
            prepare(target, state, auto)
        return 0
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired, yaml.YAMLError) as error:
        print(f'USRP preparation: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
