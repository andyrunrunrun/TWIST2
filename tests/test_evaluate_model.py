import unittest

from evaluate_model import detect_sonic_model, resolve_sonic_pd_mode


class EvaluateModelSonicPdTests(unittest.TestCase):
    def test_detect_sonic_model_is_case_insensitive(self):
        self.assertTrue(detect_sonic_model("/tmp/logs/g1/SonicPd_run/model_1000.pt"))
        self.assertFalse(detect_sonic_model("/tmp/logs/g1/plain_run/model_1000.pt"))

    def test_resolve_sonic_pd_mode_auto_enables_for_sonic_model(self):
        mode = resolve_sonic_pd_mode("/tmp/logs/g1/sonic_experiment/model_1000.pt", cli_override=None)

        self.assertTrue(mode["is_sonic_model"])
        self.assertTrue(mode["enable_sonic_pd"])
        self.assertTrue(mode["preserve_ankle_obs"])
        self.assertEqual(mode["source"], "path_auto")

    def test_resolve_sonic_pd_mode_cli_enable_overrides_non_sonic_path(self):
        mode = resolve_sonic_pd_mode("/tmp/logs/g1/plain_experiment/model_1000.pt", cli_override=True)

        self.assertFalse(mode["is_sonic_model"])
        self.assertTrue(mode["enable_sonic_pd"])
        self.assertTrue(mode["preserve_ankle_obs"])
        self.assertEqual(mode["source"], "cli_force_on")

    def test_resolve_sonic_pd_mode_cli_disable_overrides_sonic_path(self):
        mode = resolve_sonic_pd_mode("/tmp/logs/g1/sonic_experiment/model_1000.pt", cli_override=False)

        self.assertTrue(mode["is_sonic_model"])
        self.assertFalse(mode["enable_sonic_pd"])
        self.assertFalse(mode["preserve_ankle_obs"])
        self.assertEqual(mode["source"], "cli_force_off")


if __name__ == "__main__":
    unittest.main()
