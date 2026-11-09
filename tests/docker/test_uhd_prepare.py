"""Fault injection for the firmware transaction; no hardware commands are executed."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/docker_uhd_prepare.py'
spec = importlib.util.spec_from_file_location('prepare', SCRIPT)
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)
BAD = subprocess.CompletedProcess([], 1, 'RFNoC protocol mismatch between SW and HW (SW: 2.0, HW: 1.0)')
OK = subprocess.CompletedProcess([], 0, 'ok')
DEVICE = dict(type='x300', addr='192.168.30.2', serial='test123', product='X310', fpga='XG')


class FirmwareTest(unittest.TestCase):
    def test_streamed_command_timeout_terminates_child(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            u.run(['python3', '-c', 'import time; time.sleep(10)'], timeout=0.05)

    def test_active_devices_and_simulation(self):
        cfg = {'usrp_device': {'device_args': 'shared'}, 'downlink': {'tx_device_args': 'tx'},
               'uplink': {'enabled': False, 'rx_device_args': 'unused'},
               'sensing': {'rx_channel_count': 2, 'rx_device_args': 'sense',
                           'rx_channels': [{'device_args': 'separate'}, {}]}}
        self.assertEqual(u.configured_targets(cfg, 'BS'), ['tx', 'separate', 'sense'])
        self.assertEqual(u.configured_targets(cfg, 'UE'), ['shared'])
        cfg['uplink']['enabled'] = True
        self.assertEqual(u.configured_targets(cfg, 'BS'), ['tx', 'unused', 'separate', 'sense'])
        cfg['radio'] = {'radio_backend': 'sim'}
        self.assertEqual(u.configured_targets(cfg, 'BS'), [])

    def test_inactive_uplink_and_implicit_sensing_count(self):
        cfg = {'usrp_device': {'device_args': 'main'},
               'uplink': {'enabled': True, 'symbol_count': 0, 'rx_device_args': 'unused'},
               'sensing': {'rx_channels': [{'device_args': 'a'}, {'device_args': 'b'}]}}
        self.assertEqual(u.configured_targets(cfg, 'BS'), ['main', 'a', 'b'])
        cfg['sensing']['rx_channel_count'] = 3
        self.assertEqual(u.configured_targets(cfg, 'BS'), ['main', 'a', 'b'])
        cfg['sensing']['rx_channel_count'] = 0
        self.assertEqual(u.configured_targets(cfg, 'BS'), ['main'])

    def test_noncompatibility_failure_and_optout_never_flash(self):
        for result, auto in [(subprocess.CompletedProcess([], 1, 'Timeout'), True), (BAD, False)]:
            with patch.object(u, 'probe', return_value=result), patch.object(u, 'run') as run:
                with self.assertRaises(RuntimeError): u.prepare('addr=x', Path('/missing'), auto)
                run.assert_not_called()

    def test_discovery_with_empty_device_name(self):
        output = 'Device Address:\n serial: test123\n addr: 192.168.30.2\n fpga: XG\n name: \n product: X310\n type: x300\n'
        with patch.object(u, 'run', return_value=subprocess.CompletedProcess([], 0, output)):
            self.assertEqual(u.identify('addr=192.168.30.2')['product'], 'X310')

    def test_ambiguous_discovery_and_untrusted_image(self):
        output = 'Device Address:\n serial: A\nDevice Address:\n serial: B\n'
        with patch.object(u, 'run', return_value=subprocess.CompletedProcess([], 0, output)):
            with self.assertRaises(RuntimeError): u.identify('')
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / 'manifest.json'
            manifest.write_text('{}')
            with patch.object(u, 'MANIFEST', manifest):
                with self.assertRaises(RuntimeError): u.image_for(DEVICE)

    def transaction(self, bad_backup=False, failed_probe=False):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        image = root / 'image.bit'; image.write_bytes(b'x' * 1000000)
        calls = []
        def command(argv, timeout=90):
            calls.append(argv)
            if '--download' in argv:
                self.assertIn('--no-fpga', argv)
                self.assertIn('--no-fw', argv)
                prefix = Path(argv[argv.index('--out-path') + 1])
                data = b'y' if bad_backup and prefix.name == 'before-b' else b'x'
                prefix.with_suffix('.bit').write_bytes(data * 15_878_042)
            else:
                self.assertEqual(len(calls), 3)  # Two backups precede the sole write.
                self.assertIn(',verify,configure', argv[2])
                journal = json.loads((root / 'test123/transaction.json').read_text())
                self.assertEqual(journal['status'], 'update_in_progress')
                self.assertTrue(journal['backup_verified_twice'])
            return OK
        with patch.object(u, 'probe', side_effect=[BAD, BAD] + [BAD if failed_probe else OK] * 12), \
             patch.object(u, 'identify', return_value=DEVICE), \
             patch.object(u, 'image_for', return_value=image), \
             patch.object(u.os.path, 'ismount', return_value=True), \
             patch.object(u.time, 'sleep'), patch.object(u, 'run', side_effect=command):
            if bad_backup or failed_probe:
                with self.assertRaises(RuntimeError): u.prepare('addr=x', root, True)
            else:
                u.prepare('addr=x', root, True)
                self.assertEqual(json.loads((root / 'test123/transaction.json').read_text())['status'], 'complete')
        return root, image, calls

    def test_successful_transaction(self):
        _, _, calls = self.transaction()
        self.assertEqual(len(calls), 3)

    def test_backup_mismatch_aborts_before_write(self):
        root, _, calls = self.transaction(bad_backup=True)
        self.assertEqual(len(calls), 2)
        self.assertFalse((root / 'test123/transaction.json').exists())

    def test_failed_reprobe_prevents_repeat_flash(self):
        root, image, _ = self.transaction(failed_probe=True)
        with patch.object(u, 'probe', return_value=BAD), \
             patch.object(u, 'identify', return_value=DEVICE), \
             patch.object(u, 'image_for', return_value=image), \
             patch.object(u.os.path, 'ismount', return_value=True), patch.object(u, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'earlier update'): u.prepare('addr=x', root, True)
            run.assert_not_called()

    def test_ephemeral_state_cannot_flash(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(u, 'probe', return_value=BAD), \
             patch.object(u, 'identify', return_value=DEVICE), \
             patch.object(u, 'image_for', return_value=Path('/image')), \
             patch.object(u.os.path, 'ismount', return_value=False), patch.object(u, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'persistent bind'): u.prepare('addr=x', Path(directory), True)
            run.assert_not_called()


if __name__ == '__main__': unittest.main()
