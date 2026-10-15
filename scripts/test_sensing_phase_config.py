from pathlib import Path
import tempfile
import unittest
import yaml
from config_web_editor import ConfigEditorApp


class SensingPhaseConfigTest(unittest.TestCase):
    def test_round_trip(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            path = build / 'UE.yaml'
            path.write_bytes((root / 'config/UE_XR1710G.yaml').read_bytes())
            app = ConfigEditorApp(root, build)
            fields = {field['key']: field for section in app.load_config('ue')['sections']
                      for field in section['fields']}
            self.assertEqual(fields['sensing.phase_reference_min_coherence']['enabled_if'],
                             {'key': 'sensing.bistatic_phase_reference', 'value': True})
            self.assertEqual(len(fields['sensing.bistatic_phase_reference']['enabled_if']['all']), 3)
            app.save_config('ue', {'scalars': {'sensing.bi_enabled': True,
                                             'sensing.output_mode': 'dense',
                                             'sensing.sensing_delay_correction_mode': 'los_tracking',
                                             'sensing.bistatic_phase_reference': True,
                                             'sensing.phase_reference_min_coherence': '0.12'}})
            saved = yaml.safe_load(path.read_text())
            self.assertTrue(saved['sensing']['bistatic_phase_reference'])
            self.assertEqual(saved['sensing']['phase_reference_min_coherence'], 0.12)
            self.assertFalse(saved['uplink']['enabled'])


if __name__ == '__main__':
    unittest.main()
