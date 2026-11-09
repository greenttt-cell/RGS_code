"""Fixed-instance RGS vs naive nesting, validating manuscript Section 3.4.2.

    python experiment_unequal_cost_fixed.py run --output results/unequal_cost_fixed
    python experiment_unequal_cost_fixed.py plot --input results/unequal_cost_fixed

Defaults: N=2**64, r=1, (|A0|,|A1|)=(10**3,10**3), (10**4,10**4),
(10**5,10**5), costs (1,lambda), lambda=1..10**5. Repeat --case A0,A1 to
supply different fixed instances. No random sampling, averaging, or SS search.

To match Section 3.4.2, naive nesting has success ONE, including exact final
phase adjustment, while RGS uses the manuscript's biased distribution with a
0.99 success guarantee. The common required threshold is therefore 0.99.
RGS uses the paper's near-peak stopping time; it is not optimized over all
successful RGS probabilities or stopping times. No quantum state is simulated.

For either naive order, query counts do not depend on lambda. Its exact cost
is Q0+lambda*Q1. Compare both orders and analytically solve their intersections
with RGS's cost T*(1-p)+lambda*T*p. Retain only intersections on the lower
envelope. Hence finite critical points are not estimated from a plotting grid.

The leading-term prediction is
  lambda_leading = (pi/2 - 1 + p)/(p - sqrt(|A0|/N)),
when its denominator is positive. It is an asymptotic approximation, not a
finite-instance equality. Integer rounds and phase-query costs are retained in
the finite model. The same finite-N probability p is used in both comparisons.

Save curves.csv, critical_points.csv and full parameters/results.json. The plot
command reads only those files and performs no query evaluation or optimization.
The previous random-instance experiment and its results remain available.
"""
import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import experiment_equal_cost as common
from experiment_unequal_cost import biased_rgs


DEFAULT_CASES = ((1000, 1000), (10000, 10000), (100000, 100000))
DEFAULT_OUTPUT = Path("results/unequal_cost_fixed")


@dataclass(frozen=True)
class Config:
    N: int = 2**64
    r: int = 1
    cases: tuple = DEFAULT_CASES
    rgs_target: float = .99
    lambda_min: float = 1.0
    lambda_max: float = 1e5
    lambda_points: int = 241

    def validate(self):
        if not (isinstance(self.N, int) and isinstance(self.r, int) and 0 < self.r < self.N):
            raise ValueError("need integer N,r with 0<r<N")
        if not self.cases or not 0 < self.rgs_target < 1:
            raise ValueError("provide fixed cases and 0<rgs_target<1")
        if (self.lambda_points < 2 or not math.isfinite(self.lambda_min)
                or not math.isfinite(self.lambda_max)
                or not 1 <= self.lambda_min < self.lambda_max):
            raise ValueError("need lambda_points>=2 and 1<=lambda_min<lambda_max<infinity")
        for case in self.cases:
            if len(case) != 2 or any(not isinstance(x, int) or x <= self.r for x in case):
                raise ValueError("each case requires two integer cardinalities greater than r")
            biased_rgs(self.N, *case, self.r, 1, self.rgs_target)


def leading_critical_point(p, a0, N):
    denominator = p-math.sqrt(a0/N)
    return ((math.pi/2-1+p)/denominator) if denominator > 0 else None


def finite_crossings(rgs_queries, orders, lower=1.0):
    """Isolated crossings with the minimum of two affine naive-order costs.

    Parallel/coincident lines have no isolated intersection and are skipped.
    Both true crossings and touches are recorded, with their actual winners.
    """
    r0, r1 = rgs_queries
    if any(q["query_counts"][0] == r0 and q["query_counts"][1] == r1 for q in orders):
        # Equality with one envelope branch is degenerate rather than an
        # isolated intersection; do not manufacture a unique critical price.
        return []
    roots = []
    for entry in orders:
        q0, q1 = entry["query_counts"]
        denominator = r1-q1
        if denominator == 0:
            continue
        root = (q0-r0)/denominator
        if not math.isfinite(root) or root < lower:
            continue
        value = r0+root*r1
        best = min(q["query_counts"][0]+root*q["query_counts"][1] for q in orders)
        if not math.isclose(value, best, rel_tol=1e-12, abs_tol=1e-9):
            continue
        if any(math.isclose(root, old["lambda"], rel_tol=1e-11) for old in roots):
            continue

        def winner(lam):
            nr = min(q["query_counts"][0]+lam*q["query_counts"][1] for q in orders)
            return "naive" if nr < r0+lam*r1 else "rgs"

        step = max(1e-9, abs(root)*1e-6)
        left = winner(root-step) if root-step >= lower else "outside_domain"
        right = winner(root+step)
        active = [q["order"] for q in orders if math.isclose(
            q["query_counts"][0]+root*q["query_counts"][1], best, rel_tol=1e-12)]
        roots.append({"lambda": root, "cheaper_below": left, "cheaper_above": right,
                      "changes_winner": left != "outside_domain" and left != right,
                      "optimal_orders": active})
    return sorted(roots, key=lambda item:item["lambda"])


def analyze_case(config, case_index, a0, a1):
    import nested_amplification as nested
    problem = nested.SearchProblem([config.N, a0, a1, config.r], [1, 1])
    rgs = biased_rgs(config.N, a0, a1, config.r, 1, config.rgs_target)
    orders = [nested.evaluate_naive(problem, order, target_success=1).as_dict()
              for order in ((0, 1), (1, 0))]
    crossings = finite_crossings(rgs["query_counts"], orders)
    main = next((x for x in crossings if x["cheaper_below"] == "rgs"
                 and x["cheaper_above"] == "naive"), None)
    leading = leading_critical_point(rgs["sampling_probabilities"][1], a0, config.N)
    finite = main["lambda"] if main else None
    return {"case": case_index, "a0": a0, "a1": a1, "N": config.N, "r": config.r,
            "m": a0+a1-config.r, "rgs": rgs, "naive_orders": orders,
            "lambda_leading": leading, "lambda_finite": finite,
            "relative_error_percent": 100*(finite/leading-1) if finite and leading else None,
            "finite_crossings": crossings,
            "asymptotic_regime": {"m_over_sqrt_Nr": (a0+a1-config.r)/math.sqrt(config.N*config.r),
                                  "r_over_a0": config.r/a0, "r_over_a1": config.r/a1}}


def evaluate_curve(case, lam):
    r0, r1 = case["rgs"]["query_counts"]
    chosen = min(case["naive_orders"], key=lambda x:x["query_counts"][0]+lam*x["query_counts"][1])
    q0, q1 = chosen["query_counts"]
    baseline = math.pi/4*math.sqrt(case["N"]/case["r"])
    p = case["rgs"]["sampling_probabilities"][1]
    cr = math.fsum((r0, lam*r1))
    cn = math.fsum((q0, lam*q1))
    lr = baseline*(1+(lam-1)*p)
    ln = baseline*(math.pi/2+lam*math.sqrt(case["a0"]/case["N"]))
    return {"case": case["case"], "a0": case["a0"], "a1": case["a1"], "lambda": lam,
            "rgs_cost": cr, "naive_cost": cn, "cost_ratio": cn/cr,
            "leading_rgs_cost": lr, "leading_naive_cost": ln, "leading_cost_ratio": ln/lr,
            "naive_order": json.dumps(chosen["order"]), "naive_q0": q0, "naive_q1": q1,
            "rgs_q0": r0, "rgs_q1": r1, "rgs_p1": p,
            "rgs_success_lower_bound": case["rgs"]["success_lower_bound"],
            "naive_success_probability": chosen["success_probability"]}


def run_experiment(config, output, make_plots=True):
    config.validate()
    output = Path(output)
    if (output/"results.json").exists():
        raise FileExistsError("Results exist; use plot or select a new --output directory")
    import nested_amplification as nested
    import experiment_unequal_cost as random_experiment
    cases = [analyze_case(config, i+1, *case) for i, case in enumerate(config.cases)]
    rows, critical = [], []
    grid = np.geomspace(config.lambda_min, config.lambda_max, config.lambda_points).tolist()
    grid[0], grid[-1] = config.lambda_min, config.lambda_max
    for case in cases:
        extras = [case["lambda_leading"]]+[x["lambda"] for x in case["finite_crossings"]]
        values = sorted(set(grid+[x for x in extras if x is not None
                                  and config.lambda_min <= x <= config.lambda_max]))
        rows.extend(evaluate_curve(case, lam) for lam in values)
        o01, o10 = case["naive_orders"]
        critical.append({"case":case["case"], "a0":case["a0"], "a1":case["a1"],
                         "N":config.N,"r":config.r,"m":case["m"],
                         "rgs_p1":case["rgs"]["sampling_probabilities"][1],
                         "lambda_leading":case["lambda_leading"],"lambda_finite":case["lambda_finite"],
                         "relative_error_percent":case["relative_error_percent"],
                         "rgs_q0":case["rgs"]["query_counts"][0],"rgs_q1":case["rgs"]["query_counts"][1],
                         "naive_01_q0":o01["query_counts"][0],"naive_01_q1":o01["query_counts"][1],
                         "naive_10_q0":o10["query_counts"][0],"naive_10_q1":o10["query_counts"][1],
                         "crossings_json":json.dumps(case["finite_crossings"]),
                         **case["asymptotic_regime"]})
    output.mkdir(parents=True, exist_ok=True)
    common._write_csv(output/"curves.csv", rows)
    common._write_csv(output/"critical_points.csv", critical)
    common._write_json(output/"results.json", {"config":asdict(config),"cases":cases})
    common._write_json(output/"metadata.json", {
        "schema_version":1,"protocol":"fixed_instances_rgs_vs_exact_naive_v1",
        "created_utc":datetime.now(timezone.utc).isoformat(),"config":asdict(config),
        "source_sha256":{Path(path).name:common._source_hash(path) for path in (
            __file__, nested.__file__, common.__file__, random_experiment.__file__)},
        "environment":{"python":sys.version,"platform":platform.platform(),
                       "packages":{x:importlib.metadata.version(x) for x in ("numpy","matplotlib")}},
        "protocol_notes":["Only RGS and naive nesting; fixed cardinalities, no random averaging or confidence intervals.",
                          "Naive success=1, RGS certified success>=rgs_target, matching Section 3.4.2.",
                          "RGS uses the paper's sufficient-bound biased sampling and near-peak stopping time.",
                          "Both naive orders are compared exactly at every price; order-specific counts are computed once.",
                          "Leading critical points use the full finite-N p to isolate leading-cost approximation errors.",
                          "Finite crossings analytically intersect the affine query costs and verify the optimal-order envelope.",
                          "Coincident intervals have no isolated critical point. All isolated candidates are stored.",
                          "Curve grids include in-range analytical critical points; grid resolution does not determine their accuracy."]})
    for row in critical:
        print("case {case}: (a0,a1)=({a0},{a1}), leading={lambda_leading}, finite={lambda_finite}, error(%)={relative_error_percent}".format(**row))
    if make_plots:
        for path in plot_saved(output):
            print("Saved "+str(path))
    return cases


def _math_size(value):
    exponent = round(math.log10(value))
    return r"10^{%d}" % exponent if 10**exponent == value else str(value)


def plot_saved(input_dir, figure_dir=None, dpi=240):
    """Redraw saved finite-query curves; do not evaluate any algorithm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    input_dir = Path(input_dir)
    figure_dir = Path(figure_dir) if figure_dir else input_dir/"figures"
    with (input_dir/"curves.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (input_dir/"critical_points.csv").open(encoding="utf-8-sig", newline="") as handle:
        critical = list(csv.DictReader(handle))
    if not rows or not critical:
        raise ValueError("saved tables are empty")
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    def selected(case):
        return sorted((r for r in rows if r["case"] == case["case"]), key=lambda r:float(r["lambda"]))
    def number(row, key):
        return float(row[key]) if row[key] else None
    with plt.rc_context({"font.family":"DejaVu Sans","font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42}):
        cols = min(3, len(critical)); nr = math.ceil(len(critical)/cols)
        fig, axes = plt.subplots(nr, cols, figsize=(4.1*cols,3.65*nr+.3), squeeze=False, sharey=True)
        fig.subplots_adjust(left=max(.065,.16/cols),right=.99,bottom=.19,top=.74,wspace=.29,hspace=.55)
        for ax, case in zip(axes.flat, critical):
            data = selected(case); x = [float(r["lambda"]) for r in data]
            for field,color,style in (("rgs_cost","#0072B2","-"),("naive_cost","#D55E00","-")):
                ax.plot(x,[float(r[field]) for r in data],color=color,linestyle=style,
                        linewidth=1.6 if style=="-" else 1.0,alpha=1 if style=="-" else .75)
            for field,color,style in (("lambda_finite",".15","-."),):
                root=number(case,field)
                if root and x[0]<=root<=x[-1]: ax.axvline(root,color=color,linestyle=style,linewidth=1)
            title=r"$(|A_0|,|A_1|)=(%s,%s)$" % (_math_size(int(case["a0"])),_math_size(int(case["a1"])))
            ax.set(xscale="log",yscale="log",xlabel=r"Cost ratio $\lambda$",ylabel="Total oracle cost",title=title)
            ax.grid(True,which="major",alpha=.22,linewidth=.6)
        for ax in list(axes.flat)[len(critical):]: ax.set_visible(False)
        handles=[Line2D([],[],color=c,linestyle=s,label=l) for c,s,l in (
            ("#0072B2","-","Random Grover search"),("#D55E00","-","Naïve nested AA"),
            (".15","-.","Crossing"))]
        fig.legend(handles=handles,loc="upper center",ncol=3,frameon=False,fontsize=10)
        for ext in ("pdf","png"):
            path=figure_dir/("oracle_cost_fixed."+ext);fig.savefig(path,dpi=dpi);outputs.append(path)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.8,4.5),layout="constrained")
        colors=plt.get_cmap("tab10")
        case_handles=[]
        ax.axhline(1,color=".25",linewidth=1)
        for i,case in enumerate(critical):
            data=selected(case);x=[float(r["lambda"]) for r in data];color=colors(i%10)
            label=r"$(%s,%s)$" % (_math_size(int(case["a0"])),_math_size(int(case["a1"])))
            handle,=ax.plot(x,[float(r["cost_ratio"]) for r in data],color=color,linewidth=1.7,label=label)
            case_handles.append(handle)
            for field,marker in (("lambda_finite","o"),):
                root=number(case,field)
                if root and x[0]<=root<=x[-1]: ax.plot(root,1,marker=marker,color=color,markersize=6,linestyle="none")
        fig.legend(handles=case_handles,title=r"$(|A_0|,|A_1|)$",loc="outside upper center",
                   ncol=min(3,len(case_handles)),frameon=False,fontsize=9)
        styles=[Line2D([],[],color=".25",linestyle=s,marker=m,label=l) for s,m,l in (
            ("-",None,"Finite query counts"),("none","o","Finite crossing"))]
        ax.legend(handles=styles,loc="lower left",frameon=False,fontsize=9)
        ax.set(xscale="log",yscale="log",xlabel=r"Cost ratio $\lambda=c_1/c_0$",
               ylabel=r"Naïve nested AA cost / RGS cost")
        ax.grid(True,which="major",alpha=.22,linewidth=.6)
        for ext in ("pdf","png"):
            path=figure_dir/("relative_cost_fixed."+ext);fig.savefig(path,dpi=dpi);outputs.append(path)
        plt.close(fig)
    return outputs


def parse_case(value):
    try:
        pair=tuple(int(x.strip()) for x in value.split(","))
        if len(pair)!=2:raise ValueError
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use --case A0,A1 with two integers") from exc
    return pair


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    commands=parser.add_subparsers(dest="command",required=True)
    run=commands.add_parser("run",help="Evaluate fixed cases and exact crossover points")
    run.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);run.add_argument("--no-plot",action="store_true")
    run.add_argument("--case",dest="cases",type=parse_case,action="append")
    defaults=Config()
    for field,value in asdict(defaults).items():
        if field!="cases":run.add_argument("--"+field.replace("_","-"),dest=field,type=type(value),default=value)
    plot=commands.add_parser("plot",help="Read saved CSVs only")
    plot.add_argument("--input",type=Path,default=DEFAULT_OUTPUT);plot.add_argument("--figure-dir",type=Path)
    plot.add_argument("--dpi",type=int,default=240)
    args=parser.parse_args()
    if args.command=="plot":
        for path in plot_saved(args.input,args.figure_dir,args.dpi):print("Saved "+str(path))
    else:
        config={k:getattr(args,k) for k in asdict(defaults) if k!="cases"}
        config["cases"]=tuple(args.cases) if args.cases else DEFAULT_CASES
        run_experiment(Config(**config),args.output,not args.no_plot)


if __name__=="__main__":main()
