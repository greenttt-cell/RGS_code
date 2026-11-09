"""Checks for fixed-instance, cost-weighted comparisons and reproducible data."""
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import nested_amplification as n
import experiment_unequal_cost as e


class UnequalCost(unittest.TestCase):
    def test_valid_reproducible_fixed_cohort_and_log_grid(self):
        config=e.Config();config.validate()
        first=e.generate_cohort(config)
        self.assertEqual(first,e.generate_cohort(config))
        self.assertEqual(len(first),50)
        for instance in first:
            problem=n.SearchProblem.from_blocks(instance["block_sizes"],[1,1])
            self.assertEqual(problem.intersection_sizes,tuple(instance["intersection_sizes"]))
            self.assertEqual(problem.r,1)
            self.assertTrue(all(10<=instance[key]<=1000 for key in ("a0","a1")))
            self.assertEqual(sum(instance["block_sizes"]),config.N)
        grid=config.lambdas()
        self.assertEqual((grid[0],grid[-1]),(1,1e4))
        self.assertTrue(np.allclose(np.diff(np.log10(grid)),.1))
        uniform=e.generate_cohort(e.Config(size_distribution="uniform"))
        self.assertNotEqual(first,uniform)

    def test_biased_probability_full_formula_and_affine_cost(self):
        N,a0,a1,r,target=2**48,1000,100,1,.99
        m=a0+a1-r
        p=(4*(m-r)/math.sqrt(N*r)+2*math.sqrt((m-r)/N))/(1-target-4*r/N-2*(m-r)/math.sqrt(N*r))
        previous=None
        for lam in (1,10,1e4):
            result=e.biased_rgs(N,a0,a1,r,lam,target)
            self.assertAlmostEqual(result["sampling_probabilities"][1],p)
            self.assertAlmostEqual(result["weighted_cost"],result["steps"]*(1+(lam-1)*p),delta=1e-5)
            self.assertGreaterEqual(result["success_lower_bound"],target)
            self.assertIsNone(result["success_probability"])
            self.assertAlmostEqual(result["weighted_cost"],sum(q*c for q,c in zip(result["query_counts"],(1,lam))),delta=1e-5)
            if previous:self.assertEqual(previous["sampling_probabilities"],result["sampling_probabilities"])
            previous=result

    def test_both_naive_orders_and_actual_ss_completion(self):
        for lam in (1,100,1e4):
            problem=n.SearchProblem([2**48,1000,100,1],[1,lam])
            naive=n.optimize_naive(problem,.99)
            expected=min(n.evaluate_naive(problem,order,.99).weighted_cost for order in ((0,1),(1,0)))
            self.assertEqual(naive.weighted_cost,expected)
            ss=n.optimize_ss_for_order(problem,naive.order,.99)
            checked=n.evaluate_ss(problem,ss.iterations,naive.order,.99,"amplify")
            self.assertEqual(ss.weighted_cost,checked.weighted_cost)
            self.assertEqual(ss.query_counts,checked.query_counts)
            self.assertGreaterEqual(ss.success_probability,.99)

    def test_paired_ratio_and_crossover_brackets(self):
        comparison=e.paired_comparison([2,4,6],[1,2,3],np.random.default_rng(2),100)
        self.assertEqual(comparison["ratio_of_means"],2)
        self.assertEqual(comparison["ratio_ci_low"],2)
        self.assertEqual(comparison["ratio_ci_high"],2)
        self.assertEqual(comparison["win_fraction_b"],1)
        # Ratio of means differs from the mean of individual ratios.
        comparison=e.paired_comparison([1,9],[1,3],np.random.default_rng(2),100)
        self.assertEqual(comparison["ratio_of_means"],2.5)
        rows=[{"method_a":"naive","method_b":"rgs","n":50,"lambda_index":0,"lambda":1,"ratio_of_means":1.5},
              {"method_a":"naive","method_b":"rgs","n":50,"lambda_index":1,"lambda":10,"ratio_of_means":.8}]
        result=e.crossover_brackets(rows,50)["brackets"]
        self.assertEqual(len(result),1)
        self.assertEqual((result[0]["lambda_left"],result[0]["lambda_right"]),(1,10))
        self.assertEqual(result[0]["cheaper_at_left"],"rgs")
        self.assertEqual(result[0]["cheaper_at_right"],"naive")

    def test_invalid_domain_rejected_without_selective_sampling(self):
        for config in (e.Config(samples=0),e.Config(size_min=1),e.Config(lambda_min=.5),
                       e.Config(lambda_points=1),e.Config(N=2**20),e.Config(target_success=1)):
            with self.assertRaises(ValueError):config.validate()

    def test_checkpoint_same_instances_all_lambdas_and_plot_only(self):
        config=e.Config(samples=2,lambda_points=3,lambda_max=1000,timing_repeats=1,
                        timing_min_seconds=.00001,bootstrap_resamples=50)
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)
            records=e.run_experiment(config,output,make_plots=False)
            self.assertEqual(len(records),6)
            self.assertEqual(len(e.load_records(output/"results.jsonl")),6)
            for instance in range(2):
                selected=[row for (_,j),row in records.items() if j==instance]
                self.assertEqual(len({tuple(row["intersection_sizes"]) for row in selected}),1)
                probabilities=[row["methods"]["rgs"]["result"]["sampling_probabilities"] for row in selected]
                self.assertTrue(all(p==probabilities[0] for p in probabilities))
            for row in records.values():
                self.assertEqual(row["costs"],[1,row["lambda"]])
                naive,ss=(row["methods"][m] for m in ("naive","ss"))
                self.assertEqual(naive["result"]["order"],ss["result"]["order"])
                self.assertEqual(ss["timing"]["seconds"],naive["timing"]["seconds"]+ss["timing"]["ss_extra_seconds"])
            for name,count in (("per_instance.csv",18),("summary.csv",9),("comparisons.csv",9)):
                with (output/name).open(encoding="utf-8-sig",newline="") as f:self.assertEqual(len(list(csv.DictReader(f))),count)
            checkpoint=(output/"results.jsonl").read_bytes()
            with patch.object(e.common,"measure_strategy",side_effect=AssertionError("resume reran experiment")):
                e.run_experiment(config,output,resume=True,make_plots=False)
            self.assertEqual(checkpoint,(output/"results.jsonl").read_bytes())
            with self.assertRaises(FileExistsError):e.run_experiment(config,output,make_plots=False)
            with patch.object(n,"optimize_ss_for_order",side_effect=AssertionError("plot reran optimizer")), \
                    patch.object(e,"generate_cohort",side_effect=AssertionError("plot resampled")):
                plots=e.plot_saved(output)
            self.assertEqual(len(plots),4)
            self.assertTrue(all(p.stat().st_size>1000 for p in plots))
            self.assertEqual(checkpoint,(output/"results.jsonl").read_bytes())
            metadata=json.loads((output/"metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["completed"])

    def test_failed_instance_saved_and_retried_without_resampling(self):
        config=e.Config(samples=1,lambda_points=2,lambda_max=10,timing_repeats=1,
                        timing_min_seconds=.00001,bootstrap_resamples=50)
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)
            with patch.object(e.common,"measure_strategy",side_effect=RuntimeError("test solver error")):
                with self.assertRaisesRegex(RuntimeError,"without resampling"):
                    e.run_experiment(config,output,make_plots=False)
            failed=e.load_records(output/"results.jsonl")[(0,0)]
            self.assertEqual(failed["status"],"error")
            fixed_sizes=failed["intersection_sizes"]
            complete=e.run_experiment(config,output,resume=True,make_plots=False)
            self.assertEqual(complete[(0,0)]["intersection_sizes"],fixed_sizes)
            self.assertTrue(all(row["status"]=="ok" for row in complete.values()))


if __name__=="__main__":unittest.main(verbosity=2)
