from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from config_web_editor import ConfigEditorApp


class ModulationWebConfigTest(unittest.TestCase):
    def test_round_trip_and_reject_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            app = ConfigEditorApp(Path(__file__).resolve().parents[1], build)
            for role in ('bs', 'ue'):
                path = build / f'{role.upper()}.yaml'
                path.write_text('ofdm_frame:\n  fft_size: 1024\n')
                fields = {field['key']: field for section in app.load_config(role)['sections']
                          for field in section['fields']}
                self.assertEqual(fields['scheme']['default_text'], 'QPSK')
                for scheme in ('BPSK', 'QPSK', '8PSK', '16QAM', '32QAM', '64QAM'):
                    app.save_config(role, {'scalars': {'scheme': scheme}})
                    self.assertEqual(yaml.safe_load(path.read_text())['modulation']['scheme'], scheme)
                before = path.read_bytes()
                with self.assertRaises(RuntimeError):
                    app.save_config(role, {'scalars': {'scheme': '256PSK'}})
                self.assertEqual(path.read_bytes(), before)
                app.save_config(role, {'scalars': {'scheme': ''}})
                self.assertNotIn('scheme', yaml.safe_load(path.read_text()).get('modulation', {}))


if __name__ == '__main__':
    unittest.main()
