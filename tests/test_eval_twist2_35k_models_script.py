import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class EvalTwist235kModelsScriptTests(unittest.TestCase):
    def test_dry_run_lists_expected_four_models(self):
        repo_root = Path("/home/huanghao/source/code/TWIST2")
        script_path = repo_root / "eval_twist2_35k_models.sh"
        motion_config = repo_root / "legged_gym/motion_data_configs/AMASS_numpy123_w1_EgoBody_numpy123_w1_OMOMO_numpy123_w1_interhuman_numpy123_w1_lafan1_numpy123_w1_pico_numpy123_w30_twist1_to_twist2_numpy123_w1_v1_v2_v3_g1_numpy123_w20_total49706.yaml"

        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    "bash",
                    str(script_path),
                    "0",
                    str(motion_config),
                    "8",
                    tmpdir,
                    "10",
                ],
                cwd=repo_root,
                env={**os.environ, "DRY_RUN": "1"},
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
        self.assertIn("实验数: 4", result.stdout)
        self.assertIn("sonic_35k_without_teacher/model_49999.pt", result.stdout)
        self.assertIn("TWIST2_35k_0_2_mlp_baseline/model_49999.pt", result.stdout)
        self.assertIn("TWIST2_35k_0_3_mlp_baseline/model_49999.pt", result.stdout)
        self.assertIn("TWIST2_35k_0_4_mlp_baseline/model_49999.pt", result.stdout)


if __name__ == "__main__":
    unittest.main()
