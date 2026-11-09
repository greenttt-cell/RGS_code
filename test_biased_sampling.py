"""Check sampled curves against the original one-trajectory simulation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import grover_3_random_oracle as experiment


BLOCKS = [1, 12, 8, 79, 10, 177, 31, 2**40-318]


class SampledCurveTests(unittest.TestCase):
    def test_matches_original_random_sequences_and_query_counts(self):
        results = experiment.run_sampled_curves_three(BLOCKS, [.005, .15, .4],
                   n_trials=4, seed_base=1, points=61, T_steps=60)
        _, psi, operators = experiment.build_8d_system(BLOCKS)
        for result in results.values():
            for trial, seed in enumerate(result["used_seeds"]):
                original = experiment.run_one_random_trial_three(
                    psi, operators, [result["p0"], result["p1"], result["p2"]],
                    60, np.random.default_rng(seed))
                np.testing.assert_allclose(result["sampled_prob_curves"][trial],
                                           original["prob_curve"], rtol=1e-10, atol=1e-16)
                np.testing.assert_array_equal(result["all_counts"][trial], original["counts"])
            np.testing.assert_array_equal(result["summary"]["mean_prob_curve"],
                                          result["sampled_prob_curves"].mean(axis=0))
            self.assertLess(result["final_state_norm_error_max"], 1e-10)

    def test_same_seed_reproduces_curves(self):
        first = experiment.run_sampled_curves_three(BLOCKS, [.005], n_trials=4, points=15, T_steps=100)
        second = experiment.run_sampled_curves_three(BLOCKS, [.005], n_trials=4, points=15, T_steps=100)
        np.testing.assert_array_equal(first[.005]["sampled_prob_curves"], second[.005]["sampled_prob_curves"])
        self.assertEqual(first[.005]["used_seeds"], [1, 2, 3, 4])

    def test_zero_iterations_and_single_trial(self):
        result = experiment.run_sampled_curves_three(BLOCKS, [.15], n_trials=1, T_steps=0)[.15]
        self.assertEqual(result["steps"].tolist(), [0])
        self.assertAlmostEqual(result["summary"]["mean_final_prob"], 1/2**40)
        np.testing.assert_array_equal(result["summary"]["std_prob_curve"], [0.])

    def test_save_load_without_resampling(self):
        results = experiment.run_sampled_curves_three(BLOCKS, [.005, .4], n_trials=3, points=11, T_steps=20)
        with tempfile.TemporaryDirectory() as folder:
            experiment.save_sampled_curves_three(results, folder)
            with patch.object(experiment, "run_sampled_curves_three", side_effect=AssertionError("no resampling")):
                loaded = experiment.load_sampled_curves_three(folder)
            for delta in results:
                np.testing.assert_array_equal(loaded[delta]["summary"]["mean_prob_curve"],
                                              results[delta]["summary"]["mean_prob_curve"])
                self.assertEqual(loaded[delta]["used_seeds"], results[delta]["used_seeds"])
            path = Path(folder)/"parameters.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["method"] = "second_moment_matrix_powers"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Monte Carlo"):
                experiment.load_sampled_curves_three(folder)

    def test_style_is_preserved_and_label_restored(self):
        results = experiment.run_sampled_curves_three(BLOCKS, [.005], n_trials=2, T_steps=10)
        with patch.object(plt, "show", side_effect=AssertionError("no GUI")):
            figure = experiment.plot_multi_delta_mean_curves(results, show=False)
        axes = figure.axes[0]
        self.assertEqual(axes.get_ylabel(), r"$a_0^2(t)$")
        self.assertEqual(axes.lines[0].get_color(), "#0072B2")
        self.assertFalse(axes.spines["right"].get_visible())
        self.assertFalse(axes.spines["top"].get_visible())

    def test_invalid_parameters(self):
        for kwargs in ({"n_trials": 0}, {"points": 1}, {"seed_base": -1}, {"T_steps": -1}):
            with self.assertRaises(ValueError):
                experiment.run_sampled_curves_three(BLOCKS, [.005], **kwargs)


if __name__ == "__main__":
    unittest.main()
