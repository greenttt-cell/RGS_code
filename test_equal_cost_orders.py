"""Paired orders, shared timing components, persistence and CSV-only plots."""
import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import experiment_equal_cost as common
import experiment_equal_cost_orders as e
import nested_amplification as n


def fast_timer(function, repeats, min_seconds):
    return function(), {"seconds": .001, "seconds_per_call_samples": [.001],
                        "batch_size": 1, "repeats": 1, "pilot_seconds": .001}


class FiveMethodComparison(unittest.TestCase):
    def config(self, **overrides):
        return e.Config(k_min=2, k_max=3, samples=2, timing_repeats=1,
                        timing_min_seconds=.00001, bootstrap_resamples=30, **overrides)

    def test_default_range_and_independent_order_seed(self):
        self.assertEqual(e.Config().k_max, 15)
        config = e.Config()
        self.assertEqual(e.random_order(config, 1, 0), (0,))
        first = e.random_order(config, 7, 4)
        self.assertEqual(first, e.random_order(config, 7, 4))
        self.assertEqual(sorted(first), list(range(7)))
        self.assertNotEqual(first, e.random_order(replace(config, order_seed=config.order_seed+1), 7, 4))
        a = common.generate_instance(config, 5, 2)
        b = common.generate_instance(replace(config, order_seed=config.order_seed+1), 5, 2)
        self.assertEqual(a, b)

    def test_shared_orders_finite_costs_and_one_dp_per_case(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(common, "measure_strategy", side_effect=fast_timer), \
                    patch.object(n, "optimize_naive", wraps=n.optimize_naive) as dp, \
                    patch.object(n, "optimize_ss", side_effect=AssertionError("independent SS order search")):
                records = e.run_experiment(config, output, make_plots=False)
            self.assertEqual(dp.call_count, 1+4)  # One warmup plus one measured call per case.
            for record in records.values():
                self.assertEqual(set(record["methods"]), set(e.METHODS))
                random_order = tuple(record["random_order"])
                optimized_order = tuple(record["optimized_order"])
                for method in ("naive_random", "ss_random"):
                    self.assertEqual(tuple(record["methods"][method]["result"]["order"]), random_order)
                for method in ("naive_optimized", "ss_optimized"):
                    self.assertEqual(tuple(record["methods"][method]["result"]["order"]), optimized_order)
                full = n.SearchProblem(record["intersection_sizes"])
                self.assertEqual(record["methods"]["naive_random"]["result"]["query_counts"],
                                 n.evaluate_naive(full, random_order, config.target_success).query_counts)
                self.assertLessEqual(record["methods"]["naive_optimized"]["result"]["weighted_cost"],
                                     record["methods"]["naive_random"]["result"]["weighted_cost"])
                for method in ("ss_random", "ss_optimized"):
                    result = record["methods"][method]["result"]
                    self.assertEqual(result["completion"], "amplify")
                    self.assertGreaterEqual(result["success_probability"], config.target_success-1e-12)
                for method in e.METHODS:
                    entry = record["methods"][method]
                    expected = sum(record["components"][name]["seconds"] for name in e.PARTS[method])
                    self.assertEqual(entry["timing"]["seconds"], expected)
                    self.assertEqual(set(entry["timing"]["components"]), set(e.PARTS[method]))
                self.assertEqual(record["methods"]["naive_random"]["timing"]["seconds"], .002)
                self.assertEqual(record["methods"]["ss_optimized"]["timing"]["seconds"], .002)
                self.assertEqual(record["methods"]["naive_optimized"]["timing"]["seconds"], .001)
                self.assertEqual(record["components"]["naive_optimized"],
                                 record["methods"]["ss_optimized"]["timing"]["components"]["naive_optimized"])
            with (output/"summary.csv").open(encoding="utf-8-sig") as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual(len(summary), 10)
            self.assertTrue(all(row["complete"] == "1" and int(row["n"]) == 2 for row in summary))

    def test_calibration_draws_do_not_change_the_actual_random_order(self):
        config = replace(self.config(), k_min=3, k_max=3, samples=1)
        def many_calls(function, repeats, min_seconds):
            for _ in range(5):
                result = function()
            return result, {"seconds": .002, "seconds_per_call_samples": [.002],
                            "batch_size": 5, "repeats": 1, "pilot_seconds": .001}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(common, "measure_strategy", side_effect=fast_timer):
                first = e.run_experiment(config, Path(directory)/"first", make_plots=False)[3, 0]
            with patch.object(common, "measure_strategy", side_effect=many_calls):
                second = e.run_experiment(config, Path(directory)/"second", make_plots=False)[3, 0]
            self.assertEqual(first["random_order"], second["random_order"])
            self.assertEqual(first["optimized_order"], second["optimized_order"])
            for method in e.METHODS:
                self.assertEqual(first["methods"][method]["result"], second["methods"][method]["result"])

    def test_saved_inputs_are_reused_without_changes_and_need_full_sizes(self):
        config = replace(self.config(), k_min=2, k_max=2, samples=1)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"inputs"
            source.mkdir()
            original = common.generate_instance(config, 2, 0)
            original["status"] = "ok"
            common._append_record(source/"instances.jsonl", original)
            before = (source/"instances.jsonl").read_bytes()
            with patch.object(common, "generate_instance", side_effect=AssertionError("new input sampled")), \
                    patch.object(common, "measure_strategy", side_effect=fast_timer):
                records = e.run_experiment(replace(config, instances_from=str(source)), Path(directory)/"output", make_plots=False)
            self.assertEqual(records[2, 0]["block_sizes"], original["block_sizes"])
            self.assertEqual(before, (source/"instances.jsonl").read_bytes())
            prefix_only = {key: original[key] for key in ("k", "instance", "N", "r", "m")}
            with self.assertRaisesRegex(ValueError, "prefix-only"):
                e._input(config, 2, 0, {(2, 0): prefix_only}, n)
            full = n.SearchProblem.from_blocks(original["block_sizes"])
            subset_only = dict(prefix_only, intersection_sizes=list(full.intersection_sizes))
            restored, _ = e._input(config, 2, 0, {(2, 0): subset_only}, n)
            self.assertEqual(restored["m"], original["m"])

    def test_resume_skips_all_completed_strategy_calls(self):
        config = replace(self.config(), k_min=2, k_max=2, samples=1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(common, "measure_strategy", side_effect=fast_timer):
                e.run_experiment(config, output, make_plots=False)
            checkpoint = (output/"instances.jsonl").read_bytes()
            with patch.object(common, "measure_strategy", side_effect=AssertionError("resume reran strategies")), \
                    patch.object(n, "optimize_naive", side_effect=AssertionError("resume reran warmup")):
                e.run_experiment(config, output, resume=True, make_plots=False)
            self.assertEqual(checkpoint, (output/"instances.jsonl").read_bytes())
            with self.assertRaises(FileExistsError):
                e.run_experiment(config, output, make_plots=False)
            with self.assertRaisesRegex(ValueError, "Resume requires"):
                e.run_experiment(replace(config, order_seed=config.order_seed+1), output, resume=True, make_plots=False)

    def test_failure_is_checkpointed_then_retries_the_same_case(self):
        config = replace(self.config(), k_min=2, k_max=2, samples=1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(common, "measure_strategy", side_effect=RuntimeError("deliberate failure")):
                with self.assertRaisesRegex(RuntimeError, "failed and was saved"):
                    e.run_experiment(config, output, make_plots=False)
            first = common.load_records(output/"instances.jsonl")[2, 0]
            self.assertEqual(first["status"], "error")
            with patch.object(common, "measure_strategy", side_effect=fast_timer):
                second = e.run_experiment(config, output, resume=True, make_plots=False)[2, 0]
            self.assertEqual(first["block_sizes"], second["block_sizes"])
            self.assertEqual(first["random_order"], second["random_order"])
            self.assertEqual(second["status"], "ok")

    def test_one_oracle_has_no_random_order_selection_charge(self):
        config = replace(self.config(), k_min=1, k_max=1, samples=1)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(common, "measure_strategy", side_effect=fast_timer):
                record = e.run_experiment(config, Path(directory), make_plots=False)[1, 0]
            self.assertEqual(record["components"]["random_order"]["seconds"], 0)
            self.assertEqual(record["methods"]["naive_random"]["timing"]["seconds"], .001)
            self.assertEqual(record["methods"]["ss_random"]["timing"]["seconds"], .001)

    def test_csv_only_plot_has_five_labels_and_all_fifteen_ticks(self):
        from matplotlib.axes import Axes
        observed = []
        original = Axes.set_xticks
        def capture(axis, ticks, *args, **kwargs):
            observed.append(list(ticks))
            return original(axis, ticks, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            rows = []
            for k in range(1, 16):
                for index, method in enumerate(e.METHODS):
                    row = {"k": k, "method": method, "n": 50, "complete": 1}
                    for metric in ("oracle_cost", "normalized_cost", "planning_seconds"):
                        row.update({metric+"_mean": index+1, metric+"_ci_low": index+.8, metric+"_ci_high": index+1.2})
                    rows.append(row)
            common._write_csv(output/"summary.csv", rows)
            before = (output/"summary.csv").read_bytes()
            with patch.object(Axes, "set_xticks", capture), \
                    patch.object(common, "generate_instance", side_effect=AssertionError("plot sampled sets")), \
                    patch.object(n, "optimize_naive", side_effect=AssertionError("plot optimized order")), \
                    patch.object(n, "optimize_ss_for_order", side_effect=AssertionError("plot optimized SS")):
                paths = e.plot_saved(output)
            self.assertEqual(len(paths), 2)
            self.assertTrue(all(path.stat().st_size > 1000 for path in paths))
            self.assertEqual(observed, [list(range(1, 16))]*2)
            self.assertEqual(before, (output/"summary.csv").read_bytes())
            self.assertEqual(len(set(e.LABELS.values())), 5)
            self.assertIn("naïve-optimal", e.LABELS["ss_optimized"])


if __name__ == "__main__":
    unittest.main()
