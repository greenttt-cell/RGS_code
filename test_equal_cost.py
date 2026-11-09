"""Independent validity, success, persistence, and plotting checks."""
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import experiment_equal_cost as e
import nested_amplification as n


class EqualCostExperiment(unittest.TestCase):
    def test_reproducible_valid_region_sampling(self):
        config = e.Config()
        config.validate()
        for k in range(1, 11):
            for index in range(10):
                instance = e.generate_instance(config, k, index)
                self.assertEqual(instance, e.generate_instance(config, k, index))
                blocks = instance["block_sizes"]
                self.assertEqual(len(blocks), 2**k)
                self.assertEqual(sum(blocks), config.N)
                self.assertEqual(blocks[0], config.r)
                self.assertTrue(all(isinstance(x, int) and x >= 0 for x in blocks))
                problem = n.SearchProblem.from_blocks(blocks)
                self.assertEqual(problem.r, config.r)
                self.assertEqual(instance["m"], config.N-blocks[-1])
                result = e.rgs_strategy(config.N, config.r, instance["m"], k, config.target_success)
                self.assertGreaterEqual(result["success_lower_bound"], config.target_success)
                if k >= 2:
                    self.assertIsNone(result["success_probability"])
        self.assertNotEqual(e.generate_instance(config, 5, 0), e.generate_instance(config, 5, 1))

    def test_rgs_guarantee_against_exact_average_quantum_evolution(self):
        for blocks in ([1, 1, 0, 2**16-2], [1, 4, 7, 2**20-12]):
            N = sum(blocks)
            m = N-blocks[-1]
            result = e.rgs_strategy(N, blocks[0], m, 2, .8)
            psi = np.sqrt(np.asarray(blocks)/N)
            D = 2*np.outer(psi, psi)-np.eye(4)
            grovers = [D*np.array([-1, -1, 1, 1]), D*np.array([-1, 1, -1, 1])]
            density = np.outer(psi, psi)
            for _ in range(result["steps"]):
                density = sum(G@density@G.T for G in grovers)/2
            self.assertGreaterEqual(density[0, 0]+1e-12, result["success_lower_bound"])
            self.assertAlmostEqual(np.trace(density), 1, places=9)

    def test_invalid_success_regime_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "below target"):
            e.Config(N=2**30, union_max=1024).validate()
        for config in (e.Config(samples=0), e.Config(union_min=0), e.Config(target_success=1),
                       e.Config(seed=-1), e.Config(timing_min_seconds=0)):
            with self.assertRaises(ValueError):
                config.validate()

    def test_bootstrap_constant_case_and_reproducibility(self):
        self.assertEqual(e._mean_interval([3,3,3], np.random.default_rng(0), 100), (3,0,3,3))
        first = e._mean_interval([1,2,5,7], np.random.default_rng(5), 500)
        self.assertEqual(first, e._mean_interval([1,2,5,7], np.random.default_rng(5), 500))
        self.assertAlmostEqual(first[0], 3.75)
        self.assertLess(first[2], first[0])
        self.assertGreater(first[3], first[0])

    def test_timer_preserves_result_and_separate_samples(self):
        calls = []
        result, timing = e.measure_strategy(lambda: calls.append(1) or 7, 3, .0001)
        self.assertEqual(result, 7)
        self.assertEqual(len(calls), 1+3*timing["batch_size"])
        self.assertEqual(len(timing["seconds_per_call_samples"]), 3)
        self.assertGreater(timing["seconds"], 0)

    def test_ss_fixed_order_reuses_order_without_another_dp_search(self):
        problem = n.SearchProblem([1000, 100, 10, 1], [50, 1])
        standalone = n.optimize_ss(problem, .99)
        reference = n.optimize_naive(problem, .99)
        with patch.object(n, "optimize_naive", side_effect=AssertionError("duplicate order search")):
            extra_only = n.optimize_ss_for_order(problem, reference.order, .99)
        standalone_data = standalone.as_dict()
        extra_data = extra_only.as_dict()
        self.assertEqual(standalone_data.pop("naive_target_success"), .99)
        self.assertIsNone(extra_data.pop("naive_target_success"))
        self.assertEqual(standalone_data, extra_data)
        with self.assertRaises(ValueError):
            n.optimize_ss_for_order(problem, (0, 0))

    def test_component_sum_preserves_real_stage_samples(self):
        order = {"seconds": .02, "seconds_per_call_samples": [.019, .02, .021],
                 "batch_size": 1, "repeats": 3, "pilot_seconds": .03}
        extra = {"seconds": .003, "seconds_per_call_samples": [.002, .003, .004],
                 "batch_size": 2, "repeats": 3, "pilot_seconds": .004}
        combined = e.combine_ss_timing(order, extra)
        self.assertEqual(combined["seconds"], order["seconds"]+extra["seconds"])
        self.assertEqual(combined["order_seconds"], order["seconds"])
        self.assertEqual(combined["ss_extra_seconds"], extra["seconds"])
        self.assertEqual(combined["components"], {"order": order, "ss_extra": extra})
        self.assertIsNone(combined["batch_size"])
        self.assertNotIn("seconds_per_call_samples", combined)

    def test_end_to_end_checkpoints_tables_resume_and_plot_only(self):
        config = e.Config(k_max=3, samples=2, timing_repeats=1,
                          timing_min_seconds=.00001, bootstrap_resamples=50)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(n, "optimize_ss", side_effect=AssertionError("end-to-end SS was retimed")):
                records = e.run_experiment(config, output, make_plots=False)
            self.assertEqual(len(records), 6)
            self.assertEqual(len(e.load_records(output/"instances.jsonl")), 6)
            self.assertTrue(all(row["status"] == "ok" for row in records.values()))
            with (output/"per_instance.csv").open(encoding="utf-8-sig") as file:
                per_instance = list(csv.DictReader(file))
            with (output/"summary.csv").open(encoding="utf-8-sig") as file:
                summary = list(csv.DictReader(file))
            self.assertEqual(len(per_instance), 18)
            self.assertEqual(len(summary), 9)
            self.assertTrue(all(int(row["n"]) == 2 for row in summary))
            metadata = json.loads((output/"metadata.json").read_text(encoding="utf-8"))
            self.assertIn("source_sha256", metadata)
            self.assertEqual(metadata["config"]["samples"], 2)
            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(metadata["timing_protocol"], e.TIMING_PROTOCOL)
            for record in records.values():
                ss = record["methods"]["ss"]["result"]
                naive = record["methods"]["naive"]["result"]
                self.assertEqual(ss["order"], naive["order"])
                self.assertEqual(ss["completion"], "amplify")
                nt = record["methods"]["naive"]["timing"]
                st = record["methods"]["ss"]["timing"]
                self.assertEqual(st["components"]["order"], nt)
                self.assertEqual(st["seconds"], nt["seconds"]+st["ss_extra_seconds"])
                self.assertGreater(st["seconds"], nt["seconds"])
                self.assertLess(record["measurement_order"].index("naive"),
                                record["measurement_order"].index("ss_extra"))
            groups = {(int(row["k"]), row["method"]): row for row in summary}
            for k in range(1, 4):
                naive, ss = groups[k, "naive"], groups[k, "ss"]
                self.assertGreater(float(ss["planning_seconds_mean"]), float(naive["planning_seconds_mean"]))
                self.assertAlmostEqual(float(ss["planning_seconds_mean"]),
                                       float(ss["order_seconds_mean"])+float(ss["ss_extra_seconds_mean"]))
                for suffix in ("mean", "ci_low", "ci_high"):
                    self.assertGreater(float(ss["planning_seconds_"+suffix]), float(naive["planning_seconds_"+suffix]))
                    self.assertEqual(ss["order_seconds_"+suffix], naive["order_seconds_"+suffix])
            for row in per_instance:
                if row["method"] == "ss":
                    self.assertEqual(row["timing_mode"], "sum_of_component_medians")
                    self.assertEqual(row["timing_batch_size"], "")
                    self.assertAlmostEqual(float(row["planning_seconds"]),
                                           float(row["order_seconds"])+float(row["ss_extra_seconds"]))
            checkpoint = (output/"instances.jsonl").read_bytes()
            with patch.object(e, "measure_strategy", side_effect=AssertionError("resume reran data")):
                e.run_experiment(config, output, resume=True, make_plots=False)
            self.assertEqual((output/"instances.jsonl").read_bytes(), checkpoint)
            # Old standalone timings must not be mixed into a component run.
            metadata.pop("timing_protocol")
            (output/"metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different timing protocol"):
                e.run_experiment(config, output, resume=True, make_plots=False)
            with self.assertRaises(FileExistsError):
                e.run_experiment(config, output, make_plots=False)
            with patch.object(e, "generate_instance", side_effect=AssertionError("plot generated data")), \
                    patch.object(n, "optimize_ss", side_effect=AssertionError("plot called optimizer")):
                paths = e.plot_saved(output)
            self.assertEqual(len(paths), 4)
            self.assertTrue(all(path.stat().st_size > 1000 for path in paths))
            self.assertEqual((output/"instances.jsonl").read_bytes(), checkpoint)

    def test_failure_is_checkpointed_not_resampled(self):
        config = e.Config(k_min=2, k_max=2, samples=2, timing_repeats=1,
                          timing_min_seconds=.00001, bootstrap_resamples=50)
        real_measure = e.measure_strategy
        count = 0
        def fail_second_method(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError("deliberate solver error")
            return real_measure(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(e, "measure_strategy", side_effect=fail_second_method):
                with self.assertRaisesRegex(RuntimeError, "no instance was silently discarded"):
                    e.run_experiment(config, output, make_plots=False)
            records = e.load_records(output/"instances.jsonl")
            self.assertEqual(list(records), [(2,0)])
            self.assertEqual(records[(2,0)]["status"], "error")
            self.assertIn("deliberate solver error", records[(2,0)]["error"])
            e.run_experiment(config, output, resume=True, make_plots=False)
            records = e.load_records(output/"instances.jsonl")
            self.assertEqual(len(records), 2)
            self.assertTrue(all(row["status"] == "ok" for row in records.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
