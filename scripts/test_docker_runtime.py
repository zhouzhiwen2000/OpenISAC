#!/usr/bin/env python3
"""Exercise container BS/UE via the UDP simulator, recording affinity and scheduling state.

Run with access to a rootful Docker daemon (sudo is supported). Uses only unique test
containers/configs and never opens a USRP. Example:
  sudo python3 scripts/test_docker_runtime.py --pin-cpus 2 3 4 5 8 9 10 11
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def available_memory_bytes() -> int:
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('cannot read host available memory')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image')
    parser.add_argument('--pin-cpus', nargs='+', type=int)
    parser.add_argument('--direction', choices=('downlink', 'uplink'), default='downlink')
    parser.add_argument('--packets', type=int, default=40)
    parser.add_argument('--memory-gib', type=int, default=6,
                        help='hard RAM limit per BS/UE container; swap disabled (default: 6)')
    parser.add_argument('--reserve-gib', type=int, default=6,
                        help='stop if host available RAM falls below this reserve (default: 6)')
    args = parser.parse_args()
    if args.memory_gib < 1 or args.reserve_gib < 1:
        parser.error('memory limit and host reserve must be positive')
    reserve = args.reserve_gib * 1024 ** 3
    minimum_available = available_memory_bytes()
    # Two backend ceilings plus the simulator's 512 MiB must fit above the reserve.
    if minimum_available < reserve + (2 * args.memory_gib + 0.5) * 1024 ** 3:
        parser.error('insufficient available RAM for both backend limits and the host reserve')
    cpus = args.pin_cpus or sorted(os.sched_getaffinity(0))[2:14]
    if len(cpus) < 4:
        parser.error('at least four available CPUs are required')
    image = args.image or 'openisac:cpu'
    binaries = ('BS', 'UE')
    output = Path(tempfile.mkdtemp(prefix='openisac-docker-cpu-'))
    wrapper_dir = output / 'wrappers'
    wrapper_dir.mkdir()
    # The launcher runs with the invoking UID; avoid permission differences when the
    # test orchestrator itself uses sudo and creates the temporary config directories.
    env = os.environ.copy()
    env.pop('SUDO_UID', None)
    env.pop('SUDO_GID', None)
    names = []
    for binary in ('ChannelSimulator', *binaries):
        name = f'openisac-validation-{os.getpid()}-{binary.lower()}'
        names.append(name)
        command = [str(ROOT / 'scripts/docker_run.sh'), '--image', image,
                   '--name', name, '--cpuset-cpus', ','.join(map(str, cpus)),
                   '--memory', '512m' if binary == 'ChannelSimulator' else f'{args.memory_gib}g']
        script = '#!/usr/bin/env bash\nset -euo pipefail\n'
        if binary == 'ChannelSimulator':
            # test_modulation_sim supplies an absolute host YAML path; use mounted cwd.
            script += 'exec ' + shlex.join(command) + ' --work-dir "$PWD" ChannelSimulator /work/build/BS.yaml\n'
        else:
            role = 'BS' if binary.endswith('BS') else 'UE'
            script += 'exec ' + shlex.join(command) + f' --config "$PWD/{role}.yaml" --work-dir "$PWD/runtime" {binary}\n'
        path = wrapper_dir / binary
        path.write_text(script)
        path.chmod(0o755)
    cmd = ['python3', str(ROOT / 'scripts/test_modulation_sim.py'), '--build-dir', str(wrapper_dir),
           '--bs', binaries[0], '--ue', binaries[1], '--modes', 'QPSK', '16QAM',
           '--direction', args.direction, '--packets', str(args.packets), '--pin-cpus', *map(str, cpus)]
    print(f'Docker validation evidence: {output}', flush=True)
    samples = []
    try:
        with (output / 'traffic.log').open('w') as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            deadline = time.monotonic() + 240
            while proc.poll() is None and time.monotonic() < deadline:
                minimum_available = min(minimum_available, available_memory_bytes())
                if minimum_available < reserve:
                    raise RuntimeError('host memory reserve reached; stopping validation containers')
                for name in names[1:]:
                    result = subprocess.run(['docker', 'inspect', name], text=True, capture_output=True)
                    if result.returncode:
                        continue
                    info = json.loads(result.stdout)[0]
                    pid = info['State']['Pid']
                    if not pid:
                        continue
                    threads = []
                    # --init is PID 1. Read its direct application's task state.
                    children = Path(f'/proc/{pid}/task/{pid}/children')
                    try:
                        app_pids = children.read_text().split()
                        for app_pid in app_pids:
                            for task in Path(f'/proc/{app_pid}/task').iterdir():
                                status = task.joinpath('status').read_text()
                                fields = dict(line.split(':', 1) for line in status.splitlines() if ':' in line)
                                tid = int(task.name)
                                threads.append({'tid': tid, 'name': fields['Name'].strip(),
                                                'cpus': fields['Cpus_allowed_list'].strip(),
                                                'policy': os.sched_getscheduler(tid),
                                                'priority': os.sched_getparam(tid).sched_priority,
                                                'locked_kb': fields.get('VmLck', '').strip()})
                    except (FileNotFoundError, ProcessLookupError):
                        continue
                    memory_state = {}
                    try:
                        group = Path(f'/proc/{pid}/cgroup').read_text().split('0::', 1)[1].strip()
                        for metric in ('memory.current', 'memory.peak'):
                            memory_state[metric] = int((Path('/sys/fs/cgroup') / group.lstrip('/') / metric).read_text())
                    except (FileNotFoundError, IndexError, ValueError):
                        pass
                    samples.append({'name': name, 'network': info['HostConfig']['NetworkMode'],
                                    'ipc': info['HostConfig']['IpcMode'],
                                    'cpuset': info['HostConfig']['CpusetCpus'],
                                    'memory_limit': info['HostConfig']['Memory'],
                                    'memory': memory_state, 'threads': threads})
                time.sleep(1)
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=20)
                raise RuntimeError('container traffic test timed out')
        (output / 'threads.json').write_text(json.dumps(samples, indent=2) + '\n')
        print((output / 'traffic.log').read_text(), end='', flush=True)
        if proc.returncode:
            raise RuntimeError(f'traffic test failed: {proc.returncode}; see {output}')
        for name in names[1:]:
            matches = [s for s in samples if s['name'] == name]
            if not any(t['policy'] in (os.SCHED_FIFO, os.SCHED_RR) and t['priority'] > 0
                       for s in matches for t in s['threads']):
                raise RuntimeError(f'no real-time application thread observed for {name}')
            if not any(',' not in t['cpus'] and '-' not in t['cpus']
                       for s in matches for t in s['threads']):
                raise RuntimeError(f'no pinned application thread observed for {name}')
            if any(s['network'] != 'host' or s['ipc'] != 'host' for s in matches):
                raise RuntimeError('container did not share host network/IPC')
        print('PASS: UDP payloads, real-time threads, CPU pinning, host network and IPC', flush=True)
    finally:
        (output / 'threads.json').write_text(json.dumps(samples, indent=2) + '\n')
        (output / 'memory.json').write_text(json.dumps({
            'backend_limit_gib': args.memory_gib, 'simulator_limit_mib': 512,
            'host_reserve_gib': args.reserve_gib,
            'minimum_host_available_bytes': minimum_available}, indent=2) + '\n')
        if 'proc' in locals() and proc.poll() is None:
            proc.terminate()
        for name in names:
            subprocess.run(['docker', 'stop', '-t', '5', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if 'proc' in locals() and proc.poll() is None:
            proc.wait(timeout=20)


if __name__ == '__main__':
    main()
