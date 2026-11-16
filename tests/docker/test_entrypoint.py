#!/usr/bin/env python3
"""Entrypoint contract: executable selection, fresh external config, failure and signals."""
from pathlib import Path
import os
import signal
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]


class EntrypointTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.app = self.root / 'app'
        (self.app / 'bin').mkdir(parents=True)
        (self.app / 'config').mkdir()
        # The preflight itself is exercised against actual container capabilities.
        (self.app / 'docker_preflight.py').write_text('')
        (self.app / 'docker_uhd_prepare.py').write_text('')
        self.config = self.root / 'external.yaml'
        self.config.write_text('generation: 1\n')
        self.run = self.root / 'run'
        self.env = os.environ | {'OPENISAC_HOME': str(self.app),
                                'OPENISAC_RUN_DIR': str(self.run),
                                'OPENISAC_CONFIG': str(self.config)}
        for binary in ('BS', 'UE'):
            path = self.app / 'bin' / binary
            role = 'BS' if binary.endswith('BS') else 'UE'
            path.write_text(f'#!/bin/sh\necho {binary}\ncat {role}.yaml\n')
            path.chmod(0o755)

    def launch(self, binary):
        return subprocess.run([str(ROOT / 'docker-entrypoint.sh'), binary],
                              env=self.env, capture_output=True, text=True)

    def test_both_programs_reload_external_yaml(self):
        for binary in ('BS', 'UE'):
            with self.subTest(binary=binary):
                self.config.write_text('generation: 1\n')
                first = self.launch(binary)
                self.assertEqual(first.returncode, 0, first.stderr)
                self.config.write_text('generation: 2\n')
                second = self.launch(binary)
                self.assertEqual(second.returncode, 0, second.stderr)
                self.assertIn('generation: 2', second.stdout)
                self.assertEqual(self.config.read_text(), 'generation: 2\n')

    def test_missing_external_config_does_not_reuse_old_copy(self):
        self.assertEqual(self.launch('BS').returncode, 0)
        self.config.unlink()
        self.assertEqual(self.launch('BS').returncode, 2)

    def test_failed_usrp_preparation_prevents_application_start(self):
        (self.app / 'docker_uhd_prepare.py').write_text('raise SystemExit(2)\n')
        result = self.launch('BS')
        self.assertEqual(result.returncode, 2)
        self.assertNotIn('generation:', result.stdout)

    def test_real_mpm_warning_is_visible_and_obeys_uhd_compatibility(self):
        (self.app / 'docker_uhd_prepare.py').write_text((ROOT / 'scripts/docker_uhd_prepare.py').read_text())
        self.config.write_text('usrp_device:\n  device_args: type=x4xx,addr=192.0.2.1\n'
                               'sensing:\n  rx_channel_count: 0\n')
        self.env['PATH'] = str(self.app / 'bin') + ':' + self.env['PATH']
        probe = self.app / 'bin/uhd_usrp_probe'
        config_info = self.app / 'bin/uhd_config_info'
        config_info.write_text('#!/bin/sh\necho "UHD 4.11.0.0"\n')
        config_info.chmod(0o755)
        for code, actual in ((0, '5.4'), (1, '5.2')):
            probe.write_text('#!/bin/sh\n'
                             'if [ "$3" = --string ]; then echo 4.10.0.0; exit 0; fi\n'
                             f'echo "MPM minor compat number mismatch. Expected: 5.3 Actual: {actual}"\n'
                             f'exit {code}\n')
            probe.chmod(0o755)
            for binary in ('BS', 'UE'):
                with self.subTest(code=code, binary=binary):
                    result = self.launch(binary)
                    self.assertEqual(result.returncode, 0 if code == 0 else 2, result.stderr)
                    self.assertIn('WARNING: USRP MPM VERSION MISMATCH', result.stderr)
                    self.assertIn('Expected: 5.3', result.stderr)
                    if code == 0:
                        self.assertIn('MPM software: 4.10.0.0', result.stderr)
                        self.assertIn('Container UHD: 4.11.0.0', result.stderr)
                    self.assertEqual('OpenISAC:' in result.stdout, code == 0)

    def test_missing_program_is_clear(self):
        (self.app / 'bin/UE').unlink()
        result = self.launch('UE')
        self.assertEqual(result.returncode, 127)
        self.assertIn('scripts/docker_build.sh cpu', result.stderr)

    def test_exec_preserves_signal_delivery(self):
        path = self.app / 'bin/BS'
        path.write_text('#!/usr/bin/env python3\nimport os,signal,time\n'
                        'from pathlib import Path\n'
                        'signal.signal(signal.SIGINT, lambda *a: exit(0))\n'
                        'Path("ready").write_text(str(os.getpid()))\n'
                        'while True: time.sleep(0.1)\n')
        proc = subprocess.Popen([str(ROOT / 'docker-entrypoint.sh'), 'BS'], env=self.env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            for _ in range(100):
                if (self.run / 'ready').exists(): break
                time.sleep(0.02)
            self.assertTrue((self.run / 'ready').exists())
            self.assertEqual(int((self.run / 'ready').read_text()), proc.pid)
            proc.send_signal(signal.SIGINT)
            self.assertEqual(proc.wait(timeout=3), 0)
        finally:
            if proc.poll() is None: proc.kill(); proc.wait()
            proc.stderr.close()


if __name__ == '__main__':
    unittest.main()
