"""Fault injection for the firmware transaction; no hardware commands are executed."""
import importlib.util
import io
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
MPM_NEWER = '[WARNING] [MPMD] MPM minor compat number mismatch. Expected: 5.3 Actual: 5.4'
MPM_OLDER = ('MPM minor compat number mismatch. Expected: 5.3 Actual: 5.2. '
             'Please update the version of MPM on your USRP device.')


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

    def test_discovery_accepts_configured_device_families(self):
        for device in (
                dict(type='x4xx', product='x410', serial='1234', mgmt_addr='192.168.1.2'),
                dict(type='n3xx', product='n310', serial='5678', addr='192.168.2.2'),
                dict(type='e3xx', product='e320', addr='192.168.3.2'),
                dict(type='b200', product='B210', serial='B123'),
                dict(type='usrp2', serial='N123', addr='192.168.4.2'),
                dict(type='future_type', serial='new-radio')):
            output = 'Device Address:\n' + ''.join(f' {k}: {v}\n' for k, v in device.items())
            with self.subTest(device=device), patch.object(u, 'run', return_value=subprocess.CompletedProcess([], 0, output)):
                self.assertEqual(u.identify(''), device)

    def test_numbered_devices_preserve_per_board_and_shared_options(self):
        targets = u.separate_targets('type=x4xx,addr0=10.0.0.2,addr1=10.0.1.2,'
                                     'mgmt_addr0=192.168.1.2,mgmt_addr1=192.168.1.3,'
                                     'serial0=A,serial1=B,fpga=CG_400,fpga1=X4_200')
        self.assertEqual([u.parse_args(x) for x in targets], [
            dict(type='x4xx', addr='10.0.0.2', mgmt_addr='192.168.1.2', serial='A', fpga='CG_400'),
            dict(type='x4xx', addr='10.0.1.2', mgmt_addr='192.168.1.3', serial='B', fpga='X4_200')])
        self.assertEqual(u.separate_targets('type=usrp2,addr=10.0.0.2 10.0.1.2'),
                         ['type=usrp2,addr=10.0.0.2', 'type=usrp2,addr=10.0.1.2'])
        with self.assertRaisesRegex(RuntimeError, 'no gaps'):
            u.separate_targets('addr1=10.0.0.2')

    def test_combined_configuration_is_checked_after_individual_updates(self):
        args = 'type=x300,addr0=10.0.0.2,addr1=10.0.1.2'
        with patch.object(u, 'prepare_one') as single, patch.object(u, 'probe', return_value=OK) as probe:
            u.prepare(args, Path('/state'), True)
            self.assertEqual([call.args[0] for call in single.call_args_list],
                             ['type=x300,addr=10.0.0.2', 'type=x300,addr=10.0.1.2'])
            probe.assert_called_once_with(args)

    def test_loader_pins_network_usb_and_custom_profiles(self):
        x410 = dict(type='x4xx', addr='10.0.0.2', mgmt_addr='192.168.1.2', serial='ABC', fpga='X4_200')
        self.assertEqual(u.parse_args(u.loader_args_for(x410, 'fpga=CG_400,master_clock_rate=245760000')),
                         dict(type='x4xx', addr='10.0.0.2', mgmt_addr='192.168.1.2', serial='ABC', fpga='CG_400'))
        self.assertEqual(u.loader_args_for(dict(type='b200', serial='ABC'), ''), 'type=b200,serial=ABC')
        self.assertNotIn('serial=', u.loader_args_for(DEVICE, ''))

    def test_mpm_warning_on_success_and_failure_never_triggers_image_loader(self):
        for code, output in ((0, MPM_NEWER), (1, MPM_OLDER),
                             (1, 'MPM major compat number mismatch. Expected: 5.3 Actual: 4.4.')):
            for auto in (True, False):
                with self.subTest(code=code, auto=auto), patch.object(u, 'probe', return_value=subprocess.CompletedProcess([], code, output)), \
                     patch.object(u, 'run') as run, patch.object(u.sys, 'stderr', new_callable=io.StringIO) as stderr:
                    if code:
                        with self.assertRaisesRegex(RuntimeError, 'MPM is incompatible'):
                            u.prepare('type=x4xx,addr=10.0.0.2', Path('/unused'), auto)
                    else:
                        u.prepare('type=x4xx,addr=10.0.0.2', Path('/unused'), auto)
                    self.assertIn('WARNING: USRP MPM VERSION MISMATCH', stderr.getvalue())
                    self.assertIn('Expected:', stderr.getvalue())
                    run.assert_not_called()

    def test_newer_mpm_warning_does_not_hide_fpga_failure_or_turn_timeout_into_update(self):
        with patch.object(u.sys, 'stderr', new_callable=io.StringIO):
            self.assertTrue(u.needs_update(subprocess.CompletedProcess([], 1, MPM_NEWER + '\n' + BAD.stdout), 'type=x4xx'))
            with self.assertRaisesRegex(RuntimeError, 'without a recognized'):
                u.needs_update(subprocess.CompletedProcess([], 1, MPM_NEWER + '\nTimeout'), 'type=x4xx')

    def test_mpm_software_release_difference_warns_without_failing_probe(self):
        for args in ('type=x4xx,addr=192.0.2.1', 'addr=192.0.2.1'):
            with self.subTest(args=args), patch.object(u, 'run', side_effect=[
                    subprocess.CompletedProcess([], 0, '[INFO] [MPMD] Initialized'),
                    subprocess.CompletedProcess([], 0, '4.10.0.0-g1234567\n'),
                    subprocess.CompletedProcess([], 0, 'UHD 4.11.0.0\n')]) as run, \
                 patch.object(u.sys, 'stderr', new_callable=io.StringIO) as stderr:
                self.assertEqual(u.probe(args).returncode, 0)
                self.assertIn('MPM software: 4.10.0.0-g1234567', stderr.getvalue())
                self.assertIn('Container UHD: 4.11.0.0', stderr.getvalue())
                self.assertIn('/mboards/0/mpm_sw_version', run.call_args_list[1].args[0])
                self.assertEqual(run.call_count, 3)

    def test_mpm_matching_release_ignores_build_suffix(self):
        with patch.object(u, 'run', side_effect=[
                subprocess.CompletedProcess([], 0, '4.11.0-g1234567\n'),
                subprocess.CompletedProcess([], 0, 'UHD 4.11.0.0-release\n')]), \
             patch.object(u.sys, 'stderr', new_callable=io.StringIO) as stderr:
            u.warn_mpm_software('type=x4xx,serial=ABC')
            self.assertEqual(stderr.getvalue(), '')
        self.assertIsNone(u.software_version('5.3'))

    def test_failed_mpm_release_query_is_not_claimed_to_match(self):
        with patch.object(u, 'run', side_effect=[subprocess.CompletedProcess([], 1, 'missing property'),
                                               subprocess.CompletedProcess([], 0, 'UHD 4.11.0.0')]), \
             patch.object(u.sys, 'stderr', new_callable=io.StringIO) as stderr:
            u.warn_mpm_software('type=x4xx')
            self.assertIn('Could not compare MPM software', stderr.getvalue())

    def test_failed_or_non_mpm_probe_never_queries_mpm_software(self):
        for args, result in (('type=x4xx', BAD), ('type=x300', OK), ('serial=1234', OK)):
            with self.subTest(args=args), patch.object(u, 'run', return_value=result) as run:
                self.assertIs(u.probe(args), result)
                self.assertEqual(run.call_count, 1)

    def test_protocol_and_minor_fpga_mismatches(self):
        for output in ('Expected protocol compatibility number 12, but got 11',
                       'FPGA compatibility number mismatch. Expected: 1.2 Actual: 1.1',
                       "Minor compat number mismatch for component 'FPGA': Expected: 8.3 Actual: 8.1",
                       "[MPM.PeriphManager] Major compat number mismatch for component 'FPGA': Expected: 8.0 Actual: 7.0"):
            self.assertTrue(u.needs_update(subprocess.CompletedProcess([], 1, output), 'addr=10.0.0.2'))

    def test_mpm_sidecar_integrity_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / 'images'; images.mkdir()
            files = {'usrp_x410_fpga_CG_400.bin': b'fpga', 'usrp_x410_fpga_CG_400.dts': b'dts',
                     'usrp_x410_fpga_CG_400.bin.md5': b'checksum'}
            for name, data in files.items(): (images / name).write_bytes(data)
            hashes = {name: u.digest(images / name) for name in files}
            manifest = root / 'manifest.json'; manifest.write_text(json.dumps(hashes))
            with patch.object(u, 'MANIFEST', manifest), patch.dict(u.os.environ, UHD_IMAGES_DIR=str(images)):
                self.assertEqual(u.verified_images(), hashes)
                (images / 'usrp_x410_fpga_CG_400.dts').write_bytes(b'changed')
                with self.assertRaisesRegex(RuntimeError, 'build-verified'):
                    u.verified_images()

    def test_build_manifest_covers_companion_files_and_rejects_partial_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / 'images'; images.mkdir()
            (images / 'fpga.bin').write_bytes(b'fpga')
            (images / 'fpga.dts').write_bytes(b'overlay')
            manifest = root / 'manifest.json'
            result = subprocess.run(['python3', str(SCRIPT.with_name('docker_uhd_manifest.py')),
                                     str(images), str(manifest)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with patch.object(u, 'MANIFEST', manifest), patch.dict(u.os.environ, UHD_IMAGES_DIR=str(images)):
                self.assertEqual(set(u.verified_images()), {'fpga.bin', 'fpga.dts'})
                (images / 'fpga.dts').unlink()
                with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                    u.verified_images()

    def test_non_x3xx_image_failure_prevents_loader(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(u, 'probe', return_value=BAD), \
             patch.object(u, 'identify', return_value=dict(type='x4xx', serial='ABC')), \
             patch.object(u.os.path, 'ismount', return_value=True), \
             patch.object(u, 'verified_images', side_effect=RuntimeError('incomplete')), patch.object(u, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                u.prepare('type=x4xx,serial=ABC', Path(directory), True)
            run.assert_not_called()
            self.assertFalse((Path(directory) / 'ABC/transaction.json').exists())

    def test_parallel_launch_cannot_enter_existing_transaction(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(u, 'probe', return_value=BAD), \
             patch.object(u, 'identify', return_value=dict(type='x4xx', serial='ABC')), \
             patch.object(u.os.path, 'ismount', return_value=True), \
             patch.object(u.fcntl, 'flock', side_effect=BlockingIOError), patch.object(u, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'Another firmware'):
                u.prepare('type=x4xx,serial=ABC', Path(directory), True)
            run.assert_not_called()

    def test_recheck_avoids_write_after_another_launcher_repaired_the_device(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(u, 'probe', side_effect=[BAD, OK]), \
             patch.object(u, 'identify', return_value=dict(type='x4xx', serial='ABC')), \
             patch.object(u.os.path, 'ismount', return_value=True), patch.object(u, 'run') as run:
            u.prepare('type=x4xx,serial=ABC', Path(directory), True)
            run.assert_not_called()

    def generic_transaction(self, device_type, *, loader_result=OK, after=OK, identity_changed=False):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        state = Path(tmp.name)
        device = dict(type=device_type, serial='unit123', product='device', addr='192.168.1.2')
        identities = [device, device | {'serial': 'other'} if identity_changed else device]
        def loader(command, timeout=90):
            self.assertEqual(command[0], 'uhd_image_loader')
            self.assertNotIn('--download', command)
            self.assertNotIn('--no-fw', command)
            self.assertNotIn('verify,configure', command[2])
            self.assertIn(f'type={device_type}', command[2])
            self.assertIn('serial=unit123', command[2])
            self.assertEqual(command[2].endswith(',reset'), device_type == 'usrp2')
            journal = json.loads((state / 'unit123/transaction.json').read_text())
            self.assertEqual(journal['status'], 'update_in_progress')
            self.assertFalse(journal['backup_verified_twice'])
            return loader_result
        with patch.object(u, 'probe', side_effect=[BAD, BAD] + [after] * 12), \
             patch.object(u, 'identify', side_effect=identities), \
             patch.object(u, 'verified_images', return_value={'release.bin': 'sha256'}), \
             patch.object(u.os.path, 'ismount', return_value=True), patch.object(u.time, 'sleep'), \
             patch.object(u, 'run', side_effect=loader) as run, patch.object(u, 'read_flash') as read_flash:
            if loader_result.returncode or after.returncode or identity_changed:
                with self.assertRaises(RuntimeError): u.prepare(f'type={device_type},addr=192.168.1.2', state, True)
            else:
                u.prepare(f'type={device_type},addr=192.168.1.2', state, True)
            self.assertEqual(run.call_count, 0 if identity_changed else 1)
            read_flash.assert_not_called()
        return state

    def test_each_family_uses_its_uhd_loader_and_completes_only_after_probe(self):
        for family in ('x4xx', 'n3xx', 'e3xx', 'b200', 'b3xx', 'usrp2', 'future_type'):
            with self.subTest(family=family):
                state = self.generic_transaction(family)
                journal = json.loads((state / 'unit123/transaction.json').read_text())
                self.assertEqual(journal['status'], 'complete')

    def test_generic_failure_keeps_pending_journal_and_blocks_repeat_write(self):
        for loader_result, after in ((BAD, OK), (OK, BAD)):
            state = self.generic_transaction('x4xx', loader_result=loader_result, after=after)
            journal = json.loads((state / 'unit123/transaction.json').read_text())
            self.assertNotEqual(journal['status'], 'complete')
            with patch.object(u, 'probe', return_value=BAD), \
                 patch.object(u, 'identify', return_value=journal['device']), \
                 patch.object(u.os.path, 'ismount', return_value=True), patch.object(u, 'run') as run:
                with self.assertRaisesRegex(RuntimeError, 'earlier update'):
                    u.prepare('type=x4xx,addr=192.168.1.2', state, True)
                run.assert_not_called()

    def test_identity_change_prevents_generic_write(self):
        state = self.generic_transaction('x4xx', identity_changed=True)
        self.assertFalse((state / 'unit123/transaction.json').exists())

    def test_serial_cannot_escape_state_directory(self):
        device = dict(type='x4xx', serial='../../other', mgmt_addr='192.168.1.2')
        self.assertNotIn('/', u.state_key(device))

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
