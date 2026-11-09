#!/usr/bin/env python3
"""Check that Docker accepts the CPU isolation tool's actual saved-plan format."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from isolate_cpus import IsolationPlan, NicIrqPlan


class LauncherTest(unittest.TestCase):
    def test_saved_plan_preserves_process_and_irq_cpu_separation(self):
        plan = IsolationPlan(reserved=[4, 5], system=[0, 1], process=[0, 1, 4, 5],
                             nic_irq=NicIrqPlan(mode='isolate', excluded_cpus=[2, 3]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.json'
            path.write_text(json.dumps(plan.to_dict()))
            result = subprocess.run([
                str(ROOT / 'scripts/docker_run.sh'), '--dry-run',
                '--cgroup-parent', 'rt-workload.slice', '--work-dir', directory,
                '--isolation-plan', str(path), 'BS'], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            command = shlex.split(result.stdout)
            self.assertEqual(command[command.index('--cpuset-cpus') + 1], '0,1,4,5')
            self.assertEqual(command[command.index('--network') + 1], 'host')
            self.assertEqual(command[command.index('--ipc') + 1], 'host')
            self.assertNotIn('--cpus', command)


if __name__ == '__main__':
    unittest.main()
