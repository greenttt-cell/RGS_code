"""Validation of cost profiles, biased sampling, paired data, and restartability."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import experiment_cost_heterogeneity as exp
import nested_amplification as nested


class CostProfileTests(unittest.TestCase):
    def test_mean_min_and_heterogeneity(self):
        for k in (2, 3, 7, 15):
            permutation = list(reversed(range(k)))
            for kappa in (1, 1.0001, 3.2, 1000):
                for profile in exp.PROFILES:
                    costs = exp.cost_vector(k, kappa, profile, permutation)
                    stats = exp.cost_statistics(costs)
                    self.assertAlmostEqual(stats["mean_cost"], 1)
                    self.assertAlmostEqual(stats["kappa_actual"], kappa)
                    self.assertEqual(min(costs), 1/kappa)
                    self.assertEqual(costs[permutation[0]], min(costs))
                    self.assertEqual(sorted(costs), [costs[i] for i in permutation])

    def test_k2_profiles_and_equal_cost_limit(self):
        for kappa in (1, 2, 10, 1000):
            vectors = [exp.cost_vector(2, kappa, p, [1, 0]) for p in exp.PROFILES]
            for vector in vectors:
                np.testing.assert_allclose(vector, vectors[0], rtol=1e-15)
                # Multiplication by kappa gives the previous (1,lambda) family.
                self.assertAlmostEqual(vector[0]/vector[1], 2*kappa-1)
        for profile in exp.PROFILES:
            self.assertEqual(exp.cost_vector(10, 1, profile, list(range(10))), [1.0]*10)

    def test_shapes_differ_despite_equal_kappa(self):
        stats = [exp.cost_statistics(exp.cost_vector(8, 10, p, list(range(8)))) for p in exp.PROFILES]
        self.assertEqual(len({round(s["kappa_actual"], 10) for s in stats}), 1)
        self.assertEqual(len({round(s["cv"], 10) for s in stats}), 3)
        self.assertEqual(len({round(s["max_min_ratio"], 10) for s in stats}), 3)

    def test_rgs_identity_profile_invariance_and_lp_vertices(self):
        for k in (2, 5, 15):
            parameters = exp.rgs_parameters(2**48, 1, 777, k, .99)
            p, T = parameters["probability_lower_constraint"], parameters["steps"]
            for kappa in (1, 3, 1000):
                expected = T*(k*p+(1-k*p)/kappa)
                for profile in exp.PROFILES:
                    permutation = list(reversed(range(k)))
                    costs = exp.cost_vector(k, kappa, profile, permutation)
                    result = exp.biased_rgs(parameters, costs, permutation[0])
                    self.assertAlmostEqual(result["weighted_cost"]/expected, 1, places=14)
                    self.assertAlmostEqual(sum(result["sampling_probabilities"]), 1)
                    self.assertGreaterEqual(min(result["sampling_probabilities"]), p)
                    self.assertGreaterEqual(result["success_lower_bound"], .99-1e-15)
                    # A linear objective on p_i>=p reaches its minimum at one
                    # of these k vertices, assigning all free mass to one ID.
                    vertices = [T*(p*sum(costs)+(1-k*p)*costs[j]) for j in range(k)]
                    self.assertAlmostEqual(result["weighted_cost"]/min(vertices), 1, places=14)

    def test_k2_matches_previous_biased_rgs(self):
        from experiment_unequal_cost import biased_rgs as old_rgs
        for kappa in (1, 4, 1000):
            costs = exp.cost_vector(2, kappa, "one_cheap", [0, 1])
            parameters = exp.rgs_parameters(2**48, 1, 149, 2, .99)
            result = exp.biased_rgs(parameters, costs, 0)
            old = old_rgs(2**48, 100, 50, 1, 2*kappa-1, .99)
            self.assertAlmostEqual(result["weighted_cost"]/(old["weighted_cost"]/kappa), 1)
            self.assertEqual(result["query_counts"], old["query_counts"])

    def test_invalid_regime_is_rejected_without_resampling(self):
        with self.assertRaises(ValueError):
            replace(exp.Config(), N=2**20).validate()
        with self.assertRaises(ValueError):
            exp.cost_vector(3, 10, "one_cheap", [0, 0, 2])


class ReproducibilityTests(unittest.TestCase):
    def test_geometry_matches_equal_cost_and_is_grid_independent(self):
        config = exp.Config(k_max=4, samples=2)
        first = exp.generate_cohort(config)
        second = exp.generate_cohort(replace(config, kappa_max=10, kappa_points=3))
        self.assertEqual(first, second)
        for row in first:
            equal = exp.common.generate_instance(config, row["k"], row["instance"])
            self.assertEqual(row["block_sizes"], equal["block_sizes"])
            self.assertEqual(sum(row["block_sizes"]), config.N)
            problem = nested.SearchProblem.from_blocks(row["block_sizes"])
            self.assertEqual(problem.r, config.r)
            self.assertEqual(sorted(row["cost_permutation"]), list(range(row["k"])))

    def test_naive_order_cost_recomputed_for_each_vector(self):
        blocks = [1, 12, 8, 79, 10, 177, 31, 2**40-318]
        for profile in exp.PROFILES:
            costs = exp.cost_vector(3, 30, profile, [2, 0, 1])
            problem = nested.SearchProblem.from_blocks(blocks, costs)
            result = nested.optimize_naive(problem, .99)
            exhaustive = min(nested.evaluate_naive(problem, order, .99).weighted_cost
                             for order in itertools.permutations(range(3)))
            self.assertAlmostEqual(result.weighted_cost/exhaustive, 1)

    def test_paired_statistics_use_ratio_of_means(self):
        result = exp.paired_comparison([1, 100], [2, 10], np.random.default_rng(2), 200)
        self.assertAlmostEqual(result["ratio_of_means"], 101/12)
        self.assertEqual(result["wins_a"], 1)
        self.assertEqual(result["wins_b"], 1)
        self.assertLessEqual(result["ratio_ci_low"], result["ratio_of_means"])
        self.assertGreaterEqual(result["ratio_ci_high"], result["ratio_of_means"])


class RunAndResumeTests(unittest.TestCase):
    def setUp(self):
        self.config = exp.Config(k_min=2, k_max=2, samples=2, kappa_points=2,
                                 kappa_max=10, bootstrap_resamples=30)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)/"results"

    def run_quiet(self, resume=False, config=None):
        with redirect_stdout(io.StringIO()):
            return exp.run_experiment(config or self.config, self.output, resume, False)

    def test_real_three_method_run_full_records_and_resume(self):
        with patch.object(nested, "optimize_naive", wraps=nested.optimize_naive) as order_spy:
            records = self.run_quiet()
        # One order search per case; SS does not call optimize_naive again.
        self.assertEqual(order_spy.call_count, len(records)+1)  # +warmup
        self.assertEqual(len(records), 12)
        for row in records.values():
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["methods"]["ss"]["order"], row["methods"]["naive"]["order"])
            self.assertEqual(row["methods"]["ss"]["completion"], "amplify")
            for result in row["methods"].values():
                exp._verify_result(result, row["costs"], .99)
            timing = row["planning_seconds"]
            self.assertEqual(timing["ss"], timing["naive"]+timing["ss_extra"])
        checkpoint = (self.output/"results.jsonl").read_bytes()
        with patch.object(nested, "optimize_naive", side_effect=AssertionError("must reuse checkpoints")):
            resumed = self.run_quiet(resume=True)
        self.assertEqual(len(resumed), len(records))
        self.assertEqual((self.output/"results.jsonl").read_bytes(), checkpoint)
        with self.assertRaises(ValueError):
            self.run_quiet(resume=True, config=replace(self.config, seed=123))
        with self.assertRaises(FileExistsError):
            self.run_quiet()
        instances = self.output/"instances.json"
        instances.write_bytes(instances.read_bytes()+b" ")
        with self.assertRaisesRegex(ValueError, "instance/permutation"):
            self.run_quiet(resume=True)

    def test_failure_checkpoint_no_resampling_and_retry(self):
        original = nested.optimize_ss_for_order
        def fail(problem, *args, **kwargs):
            if problem.N != 64:  # permit warmup
                raise RuntimeError("deliberate test failure")
            return original(problem, *args, **kwargs)
        with patch.object(nested, "optimize_ss_for_order", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "no resampling"):
                self.run_quiet()
        cohort = (self.output/"instances.json").read_bytes()
        records = exp.load_records(self.output/"results.jsonl")
        self.assertEqual(len(records), 1)
        self.assertEqual(next(iter(records.values()))["status"], "error")
        resumed = self.run_quiet(resume=True)
        self.assertEqual(len(resumed), 12)
        self.assertTrue(all(r["status"] == "ok" for r in resumed.values()))
        self.assertEqual((self.output/"instances.json").read_bytes(), cohort)

    def test_incomplete_final_line_repair_only_on_resume(self):
        path = Path(self.temp.name)/"partial.jsonl"
        good = {"k": 2, "kappa_index": 0, "profile": "one_cheap", "instance": 0, "status": "ok"}
        line = (json.dumps(good)+"\n").encode()
        path.write_bytes(line+b'{"k":')
        with self.assertRaises(ValueError):
            exp.load_records(path)
        with self.assertWarns(UserWarning):
            records = exp.load_records(path, repair_tail=True)
        self.assertEqual(len(records), 1)
        self.assertEqual(path.read_bytes(), line)
        path.write_bytes(b"invalid\n"+line)
        with self.assertRaises(ValueError):
            exp.load_records(path, repair_tail=True)


if __name__ == "__main__":
    unittest.main()
