from pathlib import Path
import tempfile
import unittest
import yaml
from config_web_editor import ConfigEditorApp


class Xr1710gWebConfigTest(unittest.TestCase):
    def test_round_trip_source_fields(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            path = build / 'UE.yaml'
            path.write_bytes((root / 'config/UE_XR1710G.yaml').read_bytes())
            app = ConfigEditorApp(root, build)
            fields = {field['key']: field for section in app.load_config('ue')['sections']
                      for field in section['fields']}
            self.assertIn('xr1710g', fields['radio_backend']['options'])
            for key in ('xr1710g_shm', 'xr1710g_center_freq', 'xr1710g_attach_timeout_s', 'xr1710g_pll_command'):
                self.assertEqual(fields[key]['enabled_if'], {'key': 'radio_backend', 'value': 'xr1710g'})
            app.save_config('ue', {'scalars': {'xr1710g_shm': '/test-rx',
                                              'xr1710g_center_freq': '6100000005',
                                              'xr1710g_attach_timeout_s': '10',
                                              'xr1710g_pll_command': 'python3 /path/pll.py --config /path/unit.json'}})
            saved = yaml.safe_load(path.read_text())
            self.assertEqual(saved['radio']['radio_backend'], 'xr1710g')
            self.assertEqual(saved['radio']['xr1710g_shm'], '/test-rx')
            self.assertEqual(saved['radio']['xr1710g_center_freq'], 6100000005)
            self.assertEqual(saved['radio']['xr1710g_attach_timeout_s'], 10)
            self.assertEqual(saved['radio']['xr1710g_pll_command'], 'python3 /path/pll.py --config /path/unit.json')
            self.assertFalse(saved['uplink']['enabled'])


if __name__ == '__main__':
    unittest.main()
