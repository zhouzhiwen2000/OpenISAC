#!/usr/bin/env python3
"""Exercise Ubuntu source rewriting and build-script argument forwarding."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class AptMirrorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'sources.list.d').mkdir()

    def select(self, mirror):
        return subprocess.run(
            ['bash', str(ROOT / 'scripts/docker_apt_mirror.sh'), mirror, str(self.root)],
            capture_output=True, text=True)

    def test_official_preserves_base_sources(self):
        source = self.root / 'sources.list.d/ubuntu.sources'
        original = 'URIs: http://archive.ubuntu.com/ubuntu/\nSigned-By: /keys/ubuntu.gpg\n'
        source.write_text(original)
        self.assertEqual(self.select('official').returncode, 0)
        self.assertEqual(source.read_text(), original)

    def test_deb822_both_mirrors_preserve_metadata(self):
        source = self.root / 'sources.list.d/ubuntu.sources'
        original = ('Types: deb\nURIs: http://archive.ubuntu.com/ubuntu/\n'
                    'Suites: noble noble-updates noble-backports\nComponents: main universe\n'
                    'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg\n\n'
                    'Types: deb\nURIs: https://security.ubuntu.com/ubuntu\n'
                    'Suites: noble-security\nComponents: main universe\n'
                    'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg\n')
        for mirror, host in (('ustc', 'mirrors.ustc.edu.cn'),
                             ('tuna', 'mirrors.tuna.tsinghua.edu.cn')):
            with self.subTest(mirror=mirror):
                source.write_text(original)
                result = self.select(mirror)
                self.assertEqual(result.returncode, 0, result.stderr)
                expected = original.replace('http://archive.ubuntu.com', f'http://{host}')
                expected = expected.replace('https://security.ubuntu.com', f'http://{host}')
                self.assertEqual(source.read_text(), expected)

    def test_legacy_country_mirror_and_vendor_sources(self):
        source = self.root / 'sources.list'
        original = ('deb [arch=amd64 signed-by=/keys/ubuntu.gpg] '
                    'http://cn.archive.ubuntu.com/ubuntu noble main\n'
                    'deb-src https://security.ubuntu.com/ubuntu/ noble-security main\n')
        source.write_text(original)
        vendor = self.root / 'sources.list.d/vendor.list'
        vendor_text = ('deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /\n'
                       'deb [signed-by=/keys/rocm.gpg] https://repo.radeon.com/rocm/apt/7.1 noble main\n'
                       'deb https://archive.ubuntu.com/ubuntu-custom noble main\n')
        vendor.write_text(vendor_text)
        result = self.select('ustc')
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = original.replace('http://cn.archive.ubuntu.com', 'http://mirrors.ustc.edu.cn')
        expected = expected.replace('https://security.ubuntu.com', 'http://mirrors.ustc.edu.cn')
        self.assertEqual(source.read_text(), expected)
        self.assertEqual(vendor.read_text(), vendor_text)

    def test_ports_idempotency_and_reselection(self):
        source = self.root / 'sources.list.d/ubuntu.sources'
        source.write_text('URIs: http://ports.ubuntu.com/ubuntu-ports/\nSuites: noble\n')
        for mirror, host in (('ustc', 'mirrors.ustc.edu.cn'),
                             ('ustc', 'mirrors.ustc.edu.cn'),
                             ('tuna', 'mirrors.tuna.tsinghua.edu.cn')):
            result = self.select(mirror)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(source.read_text(), f'URIs: http://{host}/ubuntu-ports/\nSuites: noble\n')

    def test_invalid_selector_does_not_write(self):
        source = self.root / 'sources.list'
        original = 'deb http://archive.ubuntu.com/ubuntu noble main\n'
        source.write_text(original)
        result = self.select('unknown')
        self.assertEqual(result.returncode, 2)
        self.assertIn('APT_MIRROR', result.stderr)
        self.assertEqual(source.read_text(), original)

    def test_missing_supported_sources_fails(self):
        result = self.select('ustc')
        self.assertEqual(result.returncode, 2)
        self.assertIn('No supported Ubuntu', result.stderr)

    def test_build_script_forwards_default_and_selected_mirror(self):
        docker = self.root / 'docker'
        docker.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
        docker.chmod(0o755)
        env = os.environ.copy()
        env['PATH'] = str(self.root) + ':' + env['PATH']
        env.pop('APT_MIRROR', None)
        for mirror in (None, 'ustc', 'tuna'):
            with self.subTest(mirror=mirror):
                if mirror is not None:
                    env['APT_MIRROR'] = mirror
                result = subprocess.run(['bash', str(ROOT / 'scripts/docker_build.sh'), 'cpu'],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = result.stdout.splitlines()
                index = args.index(f'APT_MIRROR={mirror or "official"}')
                self.assertEqual(args[index - 1], '--build-arg')


if __name__ == '__main__':
    unittest.main()
