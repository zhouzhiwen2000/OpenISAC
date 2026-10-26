from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from config_web_editor import ConfigEditorApp


class LdpcWebConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.build_dir = Path(self.directory.name)
        self.app = ConfigEditorApp(Path(__file__).resolve().parents[1], self.build_dir)
        # Match existing configs that omit the newly exposed fields.
        for role in ("BS", "UE"):
            (self.build_dir / f"{role}.yaml").write_text("ldpc:\n  fixed_point: false\n")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_both_roles_round_trip(self) -> None:
        for role in ("bs", "ue"):
            form = self.app.load_config(role)
            fields = {f["key"]: f for section in form["sections"] for f in section["fields"]}
            self.assertEqual(fields["decoder_iterations"]["default_text"], "12")
            self.assertEqual(float(fields["nms_normalization_factor"]["default_text"]), 0.75)
            self.assertEqual(fields["fixed_point_scale"]["enabled_if"],
                             {"key": "fixed_point", "value": True})
            self.app.save_config(role, {"scalars": {
                "decoder_iterations": "6", "nms_normalization_factor": "1.0"}})
            loaded = yaml.safe_load((self.build_dir / f"{role.upper()}.yaml").read_text())
            self.assertEqual(loaded["ldpc"]["decoder_iterations"], 6)
            self.assertEqual(loaded["ldpc"]["nms_normalization_factor"], 1.0)
            self.app.save_config(role, {"scalars": {"decoder_iterations": "",
                                                    "nms_normalization_factor": ""}})
            loaded = yaml.safe_load((self.build_dir / f"{role.upper()}.yaml").read_text())
            self.assertNotIn("decoder_iterations", loaded["ldpc"])
            self.assertNotIn("nms_normalization_factor", loaded["ldpc"])

    def test_invalid_save_preserves_file(self) -> None:
        for role in ("bs", "ue"):
            path = self.build_dir / f"{role.upper()}.yaml"
            for values in (
                {"code_rate": "0.75"},
                {"code_rate": "4/3"},
                {"decoder_iterations": "0"},
                {"decoder_iterations": "2147483648"},
                {"nms_normalization_factor": "0"},
                {"nms_normalization_factor": "1.01"},
                {"nms_normalization_factor": "nan"},
                {"nms_normalization_factor": "inf"},
                {"fixed_point": True, "nms_normalization_factor": "0.8"},
            ):
                with self.subTest(role=role, values=values):
                    before = path.read_bytes()
                    with self.assertRaises(RuntimeError):
                        self.app.save_config(role, {"scalars": values})
                    self.assertEqual(path.read_bytes(), before)

    def test_code_rates(self) -> None:
        for role in ("bs", "ue"):
            fields = {f["key"]: f for section in self.app.load_config(role)["sections"]
                      for f in section["fields"]}
            self.assertEqual(fields["code_rate"]["default_text"], "1/2")
            for rate in ("1/3", "1/2", "2/3", "3/4"):
                self.app.save_config(role, {"scalars": {"code_rate": rate}})
                saved = yaml.safe_load((self.build_dir / f"{role.upper()}.yaml").read_text())
                self.assertEqual(saved["ldpc"]["code_rate"], rate)
            self.app.save_config(role, {"scalars": {"code_rate": ""}})
            saved = yaml.safe_load((self.build_dir / f"{role.upper()}.yaml").read_text())
            self.assertNotIn("code_rate", saved["ldpc"])

    def test_float_and_q16_factors(self) -> None:
        for role in ("bs", "ue"):
            self.app.save_config(role, {"scalars": {"nms_normalization_factor": "0.8"}})
            for factor in ("0.125", "0.25", "0.375", "0.5", "0.625", "0.75", "0.875", "1"):
                self.app.save_config(role, {"scalars": {"fixed_point": True,
                                                        "nms_normalization_factor": factor}})
                saved = yaml.safe_load((self.build_dir / f"{role.upper()}.yaml").read_text())
                self.assertEqual(saved["ldpc"]["nms_normalization_factor"], float(factor))


if __name__ == "__main__":
    unittest.main()
