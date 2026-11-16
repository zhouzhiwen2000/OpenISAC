#!/usr/bin/env python3
"""Check configured USRPs and repair image compatibility with UHD's device loaders."""
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
import tempfile
import time
import threading

import yaml

MISMATCH = re.compile(
    r'(?:Major|Minor) compat number mismatch|Expected (?:FPGA|firmware|protocol) compatibility number'
    r'|(?:FPGA|firmware) compatibility number mismatch'
    r'|FPGA component .*?revision .*?UHD supports'
    r'|RFNoC protocol mismatch between SW and HW'
    r'|This MPM version does not support the loaded FPGA image', re.I | re.S)
MPM_MISMATCH = re.compile(
    r'\bMPM (?:major|minor) compat number mismatch[^\r\n]*'
    r'|(?:major|minor) compat number mismatch for (?:component )?[`\x27\"]?MPM\b[^\r\n]*'
    r'|Please update (?:the version of )?MPM\b[^\r\n]*'
    r'|This MPM version is not compatible[^\r\n]*'
    r'|MPM compatibility infos suggest[^\r\n]*', re.I)
MANIFEST = Path('/opt/uhd/share/uhd/openisac-image-sha256.json')


def parse_args(args: str) -> dict[str, str]:
    """Parse UHD's comma-separated device address without invoking a shell."""
    fields = {}
    for token in args.split(','):
        if not token.strip():
            continue
        key, _, value = token.partition('=')
        key, value = key.strip(), value.strip()
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key) or key in fields:
            raise RuntimeError(f'Invalid or duplicate UHD device argument: {key}')
        fields[key] = value
    return fields


def format_args(fields: dict[str, str]) -> str:
    return ','.join(f'{key}={value}' for key, value in fields.items())


def separate_targets(args: str) -> list[str]:
    """Apply UHD's numbered-motherboard argument rules, including shared defaults."""
    fields = parse_args(args)
    if len(fields.get('addr', '').split()) > 1:
        for index, address in enumerate(fields.pop('addr').split()):
            fields[f'addr{index}'] = address
    shared, boards = {}, {}
    for key, value in fields.items():
        match = re.fullmatch(r'(\D+\d*\D+)(\d+)', key)
        if match:
            base, index = match.groups()
            boards.setdefault(int(index), {})[base] = value
        else:
            shared[key] = value
    if not boards:
        return [args]
    # Missing indices would create a wildcard discovery target in UHD.
    if sorted(boards) != list(range(len(boards))):
        raise RuntimeError('Numbered UHD device arguments must start at 0 and have no gaps.')
    return list(dict.fromkeys(format_args(shared | boards[index]) for index in sorted(boards)))


def warn_mpm(result: subprocess.CompletedProcess, args: str) -> bool:
    """Surface MPM warnings even when UHD permits a newer minor version."""
    details = list(dict.fromkeys(MPM_MISMATCH.findall(result.stdout)))
    if not details:
        return False
    print_mpm_warning(args, details)
    return True


def print_mpm_warning(args: str, details: list[str]) -> None:
    print('\n' + '!' * 72 + '\nWARNING: USRP MPM VERSION MISMATCH / MPM 版本不一致\n'
          f'Device: {args or "configured UHD device"}\n'
          + '\n'.join(details)
          + '\nAlign the device-side MPM/system image with the container UHD release.'
          '\n请将设备端 MPM/系统镜像与容器 UHD 版本配套。'
          '\nFPGA/firmware loading does not upgrade the device OS or MPM.'
          + '\n' + '!' * 72 + '\n', file=sys.stderr, flush=True)


def software_version(output: str) -> tuple[tuple[int, ...], str] | None:
    """Read the standalone software version, never a two-component MPM API number."""
    for line in reversed(output.splitlines()):
        value = re.sub(r'^UHD\s+', '', line.strip())
        match = re.fullmatch(r'v?(\d+)\.(\d+)\.(\d+)(?:\.(\d+))?([^\s]*)', value)
        if match:
            return tuple(int(part or 0) for part in match.groups()[:4]), value
    return None


def warn_mpm_software(args: str) -> None:
    """Compare MPM and host release versions even when their RPC APIs are compatible."""
    try:
        mpm = run(['uhd_usrp_probe', '--args', args, '--string', '/mboards/0/mpm_sw_version'])
        host = run(['uhd_config_info', '--version'])
        device_version = software_version(mpm.stdout) if mpm.returncode == 0 else None
        host_version = software_version(host.stdout) if host.returncode == 0 else None
        if device_version and host_version:
            if device_version[0] != host_version[0]:
                print_mpm_warning(args, [f'MPM software: {device_version[1]}',
                                         f'Container UHD: {host_version[1]}',
                                         'UHD initialization passed; continuing with this warning.'])
            return
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f'MPM software-version query failed: {error}', file=sys.stderr)
    print('WARNING: Could not compare MPM software and host UHD versions; '
          'only the UHD compatibility check is available.', file=sys.stderr, flush=True)


def needs_update(result: subprocess.CompletedProcess, args: str) -> bool:
    warn_mpm(result, args)
    if result.returncode == 0:
        return False
    if incompatible_mpm(result.stdout):
        raise RuntimeError('MPM is incompatible; update the device-side MPM/system image before starting OpenISAC.')
    if not MISMATCH.search(MPM_MISMATCH.sub('', result.stdout)):
        raise RuntimeError('USRP probe failed without a recognized compatibility mismatch; firmware unchanged.')
    return True


def incompatible_mpm(output: str) -> bool:
    """A newer MPM minor API is a warning, not a reason to suppress FPGA repair."""
    for detail in MPM_MISMATCH.findall(output):
        versions = re.search(r'Expected:\s*(\d+)\.(\d+)\s+Actual:\s*(\d+)\.(\d+)', detail, re.I)
        if 'minor compat' in detail.lower() and versions:
            expected_major, expected_minor, actual_major, actual_minor = map(int, versions.groups())
            if expected_major == actual_major and actual_minor >= expected_minor:
                continue
        return True
    return False


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
    result = run(['uhd_usrp_probe', '--args', args, '--init-only'])
    device_type = parse_args(args).get('type', '').lower()
    if (result.returncode == 0 and len(separate_targets(args)) == 1
            and (device_type in ('x4xx', 'n3xx', 'e3xx')
                 or re.search(r'\[MPMD(?:\]|\b)', result.stdout))):
        warn_mpm_software(args)
    return result


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
    if (not re.fullmatch(r'[A-Za-z0-9_-]+', device.get('type', ''))
            or not any(device.get(key) for key in ('serial', 'addr', 'mgmt_addr', 'resource'))
            or any(',' in value or '\n' in value for value in device.values())):
        raise RuntimeError('Automatic update requires a device type and an unambiguous address, serial or resource.')
    return device


def image_for(device: dict) -> Path:
    if not all(re.fullmatch(r'[A-Za-z0-9_-]+', device.get(key, '')) for key in ('product', 'fpga')):
        raise RuntimeError('Cannot determine the X3xx product/FPGA image; specify fpga= in device_args.')
    name = f"usrp_{device['product'].lower()}_fpga_{device['fpga']}.bit"
    image = Path(os.environ.get('UHD_IMAGES_DIR', '/opt/uhd/share/uhd/images')) / name
    hashes = json.loads(MANIFEST.read_text())
    if not image.is_file() or hashes.get(name) != digest(image):
        raise RuntimeError(f'Image is missing or differs from the build-verified UHD release: {image}')
    return image


def verified_images() -> dict[str, str]:
    """Validate the complete image set, including MPM DTS and checksum sidecars."""
    directory = Path(os.environ.get('UHD_IMAGES_DIR', '/opt/uhd/share/uhd/images'))
    hashes = json.loads(MANIFEST.read_text())
    verified = {}
    for path in sorted(directory.rglob('*')):
        if not path.is_file():
            continue
        name = path.relative_to(directory).as_posix()
        # Downloader bookkeeping is never loaded onto a device.
        if name == 'inventory.json':
            continue
        checksum = digest(path)
        if hashes.get(name) != checksum:
            raise RuntimeError(f'Image differs from the build-verified UHD release: {path}')
        verified[name] = checksum
    if not verified:
        raise RuntimeError(f'No build-verified UHD images found in {directory}.')
    missing = set(hashes) - {'inventory.json'} - set(verified)
    if missing:
        # Some loaders write firmware before opening the FPGA file. Require the
        # entire release set before invoking them so a partial mount cannot cause
        # a half-completed update.
        raise RuntimeError(f'UHD image set is incomplete; missing {sorted(missing)[0]}.')
    return verified


def loader_args_for(device: dict, configured: str) -> str:
    """Pin the loader to the discovered unit and retain its configured FPGA flavor."""
    fields = {key: device[key] for key in ('type', 'addr', 'mgmt_addr', 'resource', 'serial')
              if device.get(key)}
    # X3xx serial filtering only accepts up to eight hexadecimal characters.
    if device['type'] == 'x300' and not re.fullmatch(r'[0-9a-fA-F]{1,8}', fields.get('serial', '')):
        fields.pop('serial', None)
        if not any(fields.get(key) for key in ('addr', 'resource')):
            raise RuntimeError('This X3xx serial needs an explicit address or PCIe resource.')
    options = parse_args(configured)
    for key in ('fpga', 'rpc_port'):
        if options.get(key) or device.get(key):
            fields[key] = options.get(key) or device[key]
    return format_args(fields)


def state_key(device: dict) -> str:
    serial = device.get('serial', '')
    if re.fullmatch(r'[A-Za-z0-9_-]+', serial):
        return serial  # Preserve existing X3xx recovery journals.
    identity = {key: device[key] for key in ('type', 'addr', 'mgmt_addr', 'resource', 'serial') if key in device}
    return device['type'] + '-' + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]


def same_device(before: dict, after: dict) -> bool:
    # Discovery also reports transient claimed/reachable fields on MPM devices.
    return all(before.get(key) == after.get(key)
               for key in ('type', 'product', 'serial', 'addr', 'mgmt_addr', 'resource'))


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
    targets = separate_targets(args)
    for target in targets:
        prepare_one(target, state, auto)
    if len(targets) > 1 and needs_update(probe(args), args):
        raise RuntimeError('The combined USRP configuration is still incompatible after individual device checks.')


def prepare_one(args: str, state: Path, auto: bool) -> None:
    initial = probe(args)
    if not needs_update(initial, args):
        return
    if not auto:
        raise RuntimeError('USRP is incompatible; automatic update is disabled (OPENISAC_AUTO_UPDATE_FIRMWARE=0).')
    device = identify(args)
    loader_args = loader_args_for(device, args)
    x3xx = device['type'] == 'x300'
    image = image_for(device | {'fpga': parse_args(loader_args).get('fpga', '')}) if x3xx else None
    if not state.is_dir() or not os.path.ismount(state):
        raise RuntimeError('Image updates require a persistent bind mount at OPENISAC_FIRMWARE_STATE; use docker_run.sh.')
    root = state / state_key(device)
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
        if not needs_update(current, args):
            return
        directory = Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%dT%H%M%SZ-', time.gmtime()), dir=root))
        (directory / 'probe-before.log').write_text(current.stdout)
        transaction = {'status': 'update_in_progress', 'device': device,
                       'configured_args': args, 'directory': str(directory)}
        if x3xx:
            first = read_flash(loader_args, image, directory / 'before-a')
            second = read_flash(loader_args, image, directory / 'before-b')
            if first.stat().st_size != second.stat().st_size or digest(first) != digest(second):
                raise RuntimeError('Independent flash backups differ; no update attempted.')
            command = ['uhd_image_loader', '--args', loader_args + ',verify,configure',
                       '--fpga-path', str(image), '--no-fw']
            transaction.update(image=str(image), image_sha256=digest(image),
                               backup=str(first), backup_sha256=digest(first),
                               backup_verified_twice=True,
                               restore_command=['uhd_image_loader', '--args', loader_args + ',verify,configure',
                                                '--fpga-path', str(first), '--no-fw'])
            save_json(directory / 'backup.json', transaction)
        else:
            # Model/revision-specific image selection and verification belong to UHD.
            # --download is NOT a generic readback API: some loaders ignore it and write.
            transaction.update(images=verified_images(), backup_verified_twice=False,
                               recovery='Use the device-specific UHD recovery procedure; no flash readback is available.')
            if device['type'] == 'usrp2':
                loader_args += ',reset'
            command = ['uhd_image_loader', '--args', loader_args]
        if not same_device(device, identify(loader_args)):
            raise RuntimeError('Device identity changed before update; no update attempted.')
        transaction['update_command'] = command
        save_json(journal, transaction)
        # A timeout or interruption leaves the durable journal pending for manual recovery.
        handlers = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            result = run(command, 900)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        (directory / 'update.log').write_text(result.stdout)
        warn_mpm(result, args)
        if result.returncode:
            raise RuntimeError(f'Image loader failed; see update logs and recovery information in {journal}.')
        transaction['status'] = 'awaiting_probe'
        save_json(journal, transaction)
        for _ in range(12):
            time.sleep(5)
            try:
                after = probe(args)
            except subprocess.TimeoutExpired:
                break
            (directory / 'probe-after.log').write_text(after.stdout)
            warn_mpm(after, args)
            if after.returncode == 0:
                transaction['status'] = 'complete'
                save_json(journal, transaction)
                print(f"USRP {state_key(device)}: update verified; compatible probe passed.", flush=True)
                return
            if incompatible_mpm(after.stdout):
                break
        raise RuntimeError(f'Image loaded, but the USRP needs recovery or a power cycle; see {journal}. No repeat write.')


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
