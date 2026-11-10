"""Validation of finite crossings, optimal-order envelopes and saved plots."""
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import experiment_unequal_cost_fixed as e
import nested_amplification as n


class FixedCost(unittest.TestCase):
    def test_default_predictions_and_actual_winner_changes(self):
        config=e.Config();config.validate()
        errors=[]
        for i,(a0,a1) in enumerate(config.cases):
            case=e.analyze_case(config,i+1,a0,a1)
            p=case["rgs"]["sampling_probabilities"][1]
            leading=(math.pi/2-1+p)/(p-math.sqrt(a0/config.N))
            self.assertEqual(case["lambda_leading"],leading)
            root=case["lambda_finite"]
            self.assertAlmostEqual(e.evaluate_curve(case,root)["cost_ratio"],1,places=12)
            self.assertGreater(e.evaluate_curve(case,root*.99)["cost_ratio"],1)
            self.assertLess(e.evaluate_curve(case,root*1.01)["cost_ratio"],1)
            self.assertGreaterEqual(case["rgs"]["success_lower_bound"],.99)
            self.assertTrue(all(o["success_probability"]==1 for o in case["naive_orders"]))
            self.assertLess(case["asymptotic_regime"]["m_over_sqrt_Nr"],.0001)
            errors.append(case["relative_error_percent"])
        # This deliberately chosen family progressively reduces final-stage
        # rounding effects; this is not asserted for arbitrary cardinalities.
        self.assertGreater(errors[0],errors[1]);self.assertGreater(errors[1],errors[2])
        self.assertLess(errors[2],.2)

    def test_two_crossings_of_an_order_envelope(self):
        orders=[{"order":[0,1],"query_counts":[1,12]},
                {"order":[1,0],"query_counts":[100,1]}]
        result=e.finite_crossings([10,10],orders)
        self.assertEqual([r["lambda"] for r in result],[4.5,10])
        self.assertEqual((result[0]["cheaper_below"],result[0]["cheaper_above"]),("naive","rgs"))
        self.assertEqual((result[1]["cheaper_below"],result[1]["cheaper_above"]),("rgs","naive"))
        hidden=[{"order":[0,1],"query_counts":[1,0]},orders[1]]
        self.assertEqual(e.finite_crossings([10,10],hidden),[])

    def test_parallel_and_nonpositive_leading_denominator(self):
        orders=[{"order":[0,1],"query_counts":[12,10]},
                {"order":[1,0],"query_counts":[20,11]}]
        self.assertEqual(e.finite_crossings([10,10],orders),[])
        orders[0]["query_counts"]=[10,10]
        self.assertEqual(e.finite_crossings([10,10],orders),[])
        self.assertIsNone(e.leading_critical_point(.01,100,10000))

    def test_affine_envelope_matches_actual_naive_optimizer(self):
        config=e.Config(cases=((1000,10000),))
        case=e.analyze_case(config,1,1000,10000)
        for lam in (1,2,100,1e5):
            row=e.evaluate_curve(case,lam)
            actual=n.optimize_naive(n.SearchProblem([config.N,1000,10000,config.r],[1,lam]),1)
            self.assertEqual(row["naive_cost"],actual.weighted_cost)
            self.assertEqual(json.loads(row["naive_order"]),list(actual.order))
            self.assertEqual((row["naive_q0"],row["naive_q1"]),actual.query_counts)
            self.assertAlmostEqual(row["rgs_cost"],row["rgs_q0"]+lam*row["rgs_q1"],places=4)

    def test_saved_data_and_plot_only_with_no_ss_calls(self):
        config=e.Config(cases=((1000,1000),(10000,10000)),lambda_points=5)
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)
            with patch.object(n,"optimize_ss",side_effect=AssertionError("SS called")), \
                    patch.object(n,"optimize_ss_for_order",side_effect=AssertionError("SS called")), \
                    patch.object(n,"optimize_naive",side_effect=AssertionError("unnecessary optimizer called")):
                cases=e.run_experiment(config,output,make_plots=False)
            with (output/"curves.csv").open(encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
            with (output/"critical_points.csv").open(encoding="utf-8-sig",newline="") as f: critical=list(csv.DictReader(f))
            self.assertEqual(len(critical),2)
            for case in cases:
                xs=[float(r["lambda"]) for r in rows if int(r["case"])==case["case"]]
                self.assertIn(case["lambda_finite"],xs)
                self.assertIn(case["lambda_leading"],xs)
            with self.assertRaises(FileExistsError):e.run_experiment(config,output,make_plots=False)
            saved=(output/"curves.csv").read_bytes()
            with patch.object(n,"evaluate_naive",side_effect=AssertionError("plot reevaluated")), \
                    patch.object(e,"analyze_case",side_effect=AssertionError("plot analyzed")):
                paths=e.plot_saved(output)
            self.assertEqual(len(paths),4)
            self.assertTrue(all(p.stat().st_size>1000 for p in paths))
            self.assertEqual((output/"curves.csv").read_bytes(),saved)
            metadata=json.loads((output/"metadata.json").read_text(encoding="utf-8"))
            self.assertIn("source_sha256",metadata)

    def test_custom_cases_and_invalid_parameters(self):
        self.assertEqual(e.parse_case("1000, 100"),(1000,100))
        for config in (e.Config(cases=()),e.Config(cases=((1,100),)),
                       e.Config(N=2**20),e.Config(rgs_target=1),e.Config(lambda_points=1)):
            with self.assertRaises(ValueError):config.validate()


if __name__=="__main__":unittest.main(verbosity=2)
