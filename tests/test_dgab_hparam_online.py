import csv
import json
from pathlib import Path
import tempfile
import unittest

from run_dgab_hparam_online import configuration, selected_checkpoint, validate_results


class HparamOnlineTests(unittest.TestCase):
    def test_slot_mapping(self):
        self.assertEqual(configuration(0),
                         ("dense", "expectile", "tau_v_0.51_tau_c_0.49"))
        self.assertEqual(configuration(3),
                         ("dense", "expectile", "tau_v_0.9_tau_c_0.1"))
        self.assertEqual(configuration(9), ("dense", "alpha", "alpha_5"))
        self.assertEqual(configuration(10),
                         ("sparse", "expectile", "tau_v_0.51_tau_c_0.49"))
        self.assertEqual(configuration(19), ("sparse", "alpha", "alpha_5"))

    def test_selection_and_result_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model_dir = root / "model"
            model_dir.mkdir()
            checkpoint = model_dir / "step_4000.pt"
            checkpoint.touch()
            (model_dir / "normalize_dict.pkl").touch()
            selection_file = (root / "dense" / "expectile" /
                              "tau_v_0.51_tau_c_0.49" / "best_checkpoint.json")
            selection_file.parent.mkdir(parents=True)
            selection_file.write_text(json.dumps({
                "slot": 0, "environment": "dense", "sweep": "expectile",
                "name": "tau_v_0.51_tau_c_0.49",
                "evaluation": {"budget_rate": 1, "v_goal_multiplier": 1,
                               "seed": 44},
                "best": {"checkpoint": checkpoint.name,
                         "checkpoint_path": str(checkpoint)},
            }), encoding="utf-8")
            self.assertEqual(selected_checkpoint(root, 0)[4], checkpoint)

            result = root / "results.csv"
            with result.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["advertiser", "episode", "status"])
                writer.writeheader()
                for advertiser in range(48):
                    writer.writerow({"advertiser": advertiser, "episode": 0, "status": "ok"})
            self.assertEqual(len(validate_results(result, 0)), 48)
            with self.assertRaises(RuntimeError):
                validate_results(result, 1)


if __name__ == "__main__":
    unittest.main()
