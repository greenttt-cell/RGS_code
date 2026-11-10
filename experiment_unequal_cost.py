"""Two-oracle comparison with costs (1, lambda) on a fixed random cohort.

    python experiment_unequal_cost.py run --output results/unequal_cost
    python experiment_unequal_cost.py plot --input results/unequal_cost

Defaults: N=2**48, r=1, 50 independent pairs of set cardinalities drawn
log-uniformly from 10..1000, and 41 log-spaced cost ratios from 1 to 10**4.
Exactly the SAME instances are reused at every lambda. No quantum trajectories
are simulated. Every method meets success >=0.99 under the existing query model.

RGS uses the paper's theorem-constrained cost-minimizing biased distribution.
This is not a claim of optimality over all successful RGS distributions.
Naive nesting optimizes both possible orders for each lambda; SS uses that
order and reoptimizes its counts INCLUDING completion='amplify'. SS's numerical
solution has no general global-optimality certificate. Optional analysis of
other SS orders or phase-oracle circuit simplifications is outside this script.

Data, component timings, paired comparisons and sampled crossover brackets
are saved. The plot command reads ONLY saved CSV files and runs no optimizer.
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


METHODS = ("rgs", "naive", "ss")
PAIRS = (("naive", "rgs"), ("ss", "rgs"), ("ss", "naive"))
PROTOCOL = "two_oracles_fixed_cohort_biased_rgs_v1"
DEFAULT_OUTPUT = Path("results/unequal_cost")


@dataclass(frozen=True)
class Config:
    N: int = 2**48
    r: int = 1
    samples: int = 50
    size_min: int = 10
    size_max: int = 1000
    size_distribution: str = "loguniform"
    lambda_min: float = 1.0
    lambda_max: float = 1e4
    lambda_points: int = 41
    target_success: float = .99
    seed: int = 20261028
    timing_repeats: int = 3
    timing_min_seconds: float = .005
    ss_maxiter: int = 50000
    bootstrap_resamples: int = 2000

    def validate(self):
        for name in ("N", "r", "samples", "size_min", "size_max", "timing_repeats",
                     "ss_maxiter", "bootstrap_resamples"):
            if getattr(self, name) <= 0:
                raise ValueError(name+" must be positive")
        if not self.r < self.size_min <= self.size_max < self.N:
            raise ValueError("need 0 < r < size_min <= size_max < N")
        if self.size_distribution not in ("loguniform", "uniform"):
            raise ValueError("size_distribution must be loguniform or uniform")
        if self.seed < 0 or not 0 < self.target_success < 1:
            raise ValueError("need seed>=0 and 0<target_success<1")
        if (self.lambda_points < 2 or not math.isfinite(self.lambda_min)
                or not math.isfinite(self.lambda_max)
                or not 1 <= self.lambda_min < self.lambda_max):
            raise ValueError("need lambda_points>=2 and 1<=lambda_min<lambda_max<infinity")
        if not math.isfinite(self.timing_min_seconds) or self.timing_min_seconds <= 0:
            raise ValueError("timing_min_seconds must be finite and positive")
        # Validate the full sampling domain; do not reject selected instances
        # afterwards and thereby change the stated sampling distribution.
        biased_rgs(self.N, self.size_max, self.size_max, self.r,
                   self.lambda_min, self.target_success)

    def lambdas(self):
        grid = np.geomspace(self.lambda_min, self.lambda_max, self.lambda_points)
        grid[0], grid[-1] = self.lambda_min, self.lambda_max
        return [float(x) for x in grid]


def generate_cohort(config):
    """Independently sample |A0| and |A1|; r is fixed and not resampled.

    The four disjoint regions are [r,a0-r,a1-r,N-a0-a1+r]. The lower size
    limit exceeds r, and the largest possible union is checked in advance.
    Loguniform means floor(exp(U(log(size_min),log(size_max+1)))).
    """
    cohort = []
    for instance in range(config.samples):
        seed = [config.seed, instance, 202]
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        if config.size_distribution == "uniform":
            sizes = [int(x) for x in rng.integers(config.size_min, config.size_max+1, size=2)]
        else:
            sizes = [min(config.size_max, max(config.size_min, int(math.exp(x))))
                     for x in rng.uniform(math.log(config.size_min), math.log(config.size_max+1), size=2)]
        a0, a1 = sizes
        m = a0+a1-config.r
        if m >= config.N:
            raise ValueError("sampled set sizes have a union larger than the search space")
        cohort.append({"instance": instance, "seed_components": seed,
                       "N": config.N, "r": config.r, "a0": a0, "a1": a1, "m": m,
                       "intersection_sizes": [config.N, a0, a1, config.r],
                       "block_sizes": [config.r, a0-config.r, a1-config.r, config.N-m]})
    return cohort


def biased_rgs(N, a0, a1, r, cost_ratio, target_success=.99):
    """Paper Eq. (cond_p1), with its FULL denominator, and costs (1,lambda).

    The distribution is unchanged across lambda>=1: p1=p_min, p0=1-p_min.
    At lambda=1 this is one of the tied theorem-feasible optimal distributions.
    Query counts are expectations; total cost is T*(1-p+lambda*p).
    """
    if not (math.isfinite(cost_ratio) and cost_ratio >= 1 and 0 < r <= min(a0, a1)):
        raise ValueError("need lambda>=1 and 0<r<=min(a0,a1)")
    m = a0+a1-r
    result = common.rgs_strategy(N, r, m, 2, target_success)
    base = 4*r/N + 2*(m-r)/math.sqrt(N*r)
    numerator = 4*(m-r)/math.sqrt(N*r) + 2*math.sqrt((m-r)/N)
    denominator = 1-target_success-base
    if numerator <= 0 or denominator <= 0:
        raise ValueError("the biased-sampling sufficient conditions are not feasible")
    p = numerator/denominator
    if p > .5:
        raise ValueError("the biased-sampling bound requires p_min<=1/2; increase N or reduce set sizes")
    # Round upward by one ulp so evaluating base+numerator/p cannot spoil a
    # boundary success guarantee solely through floating-point rounding.
    p = min(.5, math.nextafter(p, math.inf))
    guarantee = math.fsum((1, -base, -numerator/p))
    if guarantee < target_success-8*math.ulp(target_success):
        raise ValueError("biased RGS fails its requested success guarantee")
    T = result["steps"]
    queries = [T*(1-p), T*p]
    result.update(sampling_probabilities=[1-p, p], query_counts=queries,
                  weighted_cost=T*(1+(cost_ratio-1)*p),
                  success_lower_bound=guarantee,
                  success_certificate="theorem_biased_sampling_lower_bound",
                  probability_lower_constraint=p,
                  probability_optimality="optimal_under_theorem_sufficient_constraints")
    return result


def load_records(path):
    records = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError("invalid checkpoint line %d" % number) from exc
                records[(row["lambda_index"], row["instance"])] = row
    return records


def paired_comparison(cost_a, cost_b, rng, resamples):
    """Ratio of mean costs and a paired-bootstrap interval, plus win counts."""
    a, b = np.asarray(cost_a, dtype=float), np.asarray(cost_b, dtype=float)
    if len(a) != len(b) or not len(a) or np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("need paired nonempty positive cost arrays")
    ratios = []
    for start in range(0, resamples, 200):
        indices = rng.integers(0, len(a), size=(min(200,resamples-start),len(a)))
        ratios.extend(a[indices].mean(axis=1)/b[indices].mean(axis=1))
    low, high = np.quantile(ratios, [.025,.975])
    tied = np.isclose(a, b, rtol=1e-12, atol=0)
    wins_a, wins_b = int(np.sum((a < b)&~tied)), int(np.sum((b < a)&~tied))
    return {"ratio_of_means": float(a.mean()/b.mean()),
            "ratio_ci_low": float(low), "ratio_ci_high": float(high),
            "wins_a": wins_a, "wins_b": wins_b, "ties": int(tied.sum()),
            "win_fraction_a": wins_a/len(a), "win_fraction_b": wins_b/len(a)}


def crossover_brackets(comparisons, required_samples):
    """Locate changes in ordering of sampled MEAN costs; do not infer roots."""
    brackets = []
    for a,b in PAIRS:
        rows = sorted((r for r in comparisons if r["method_a"]==a and r["method_b"]==b
                       and r["n"]==required_samples), key=lambda r:r["lambda_index"])
        for left,right in zip(rows,rows[1:]):
            if (left["ratio_of_means"]-1)*(right["ratio_of_means"]-1) < 0:
                brackets.append({"method_a":a,"method_b":b,
                                 "lambda_left":left["lambda"],"lambda_right":right["lambda"],
                                 "cheaper_at_left":a if left["ratio_of_means"]<1 else b,
                                 "cheaper_at_right":a if right["ratio_of_means"]<1 else b})
    return {"criterion":"sign changes of mean-cost differences at adjacent sampled grid points; not exact crossing values",
            "brackets":brackets}


def export_tables(output, records, config):
    rows, groups = [], {}
    baseline = math.pi/4*math.sqrt(config.N/config.r)
    for _, record in sorted(records.items()):
        if record["status"] != "ok":
            continue
        groups.setdefault(record["lambda_index"],[]).append(record)
        for method in METHODS:
            result, timing = (record["methods"][method][key] for key in ("result","timing"))
            probabilities = result.get("sampling_probabilities",["",""])
            rows.append({"lambda_index":record["lambda_index"],"lambda":record["lambda"],
                         "instance":record["instance"],"method":method,
                         "a0":record["a0"],"a1":record["a1"],"r":config.r,"N":config.N,
                         "oracle_cost":result["weighted_cost"],"normalized_cost":result["weighted_cost"]/baseline,
                         "queries_0":result["query_counts"][0],"queries_1":result["query_counts"][1],
                         "p0":probabilities[0],"p1":probabilities[1],
                         "order":json.dumps(result.get("order",[])),
                         "success_lower_bound":result.get("success_lower_bound",result["success_probability"]),
                         "planning_seconds":timing["seconds"],
                         "order_seconds":timing["seconds"] if method=="naive" else timing.get("order_seconds",0),
                         "ss_extra_seconds":timing.get("ss_extra_seconds",0)})
    common._write_csv(output/"per_instance.csv",rows)
    summary, comparisons = [], []
    for li, group in sorted(groups.items()):
        group = sorted(group,key=lambda row:row["instance"])
        cost_arrays = {m:[r["methods"][m]["result"]["weighted_cost"] for r in group] for m in METHODS}
        for mi,method in enumerate(METHODS):
            rng = np.random.default_rng(np.random.SeedSequence([config.seed,li,mi,901]))
            estimate = common._mean_interval(cost_arrays[method],rng,config.bootstrap_resamples)
            row = {"lambda_index":li,"lambda":group[0]["lambda"],"method":method,"n":len(group)}
            for suffix,value in zip(("mean","std","ci_low","ci_high"),estimate):
                row["oracle_cost_"+suffix]=value
                row["normalized_cost_"+suffix]=value/baseline
            row["queries_0_mean"]=float(np.mean([r["methods"][method]["result"]["query_counts"][0] for r in group]))
            row["queries_1_mean"]=float(np.mean([r["methods"][method]["result"]["query_counts"][1] for r in group]))
            row["planning_seconds_mean"]=float(np.mean([r["methods"][method]["timing"]["seconds"] for r in group]))
            row["success_lower_bound_min"]=min(r["methods"][method]["result"].get(
                "success_lower_bound",r["methods"][method]["result"]["success_probability"]) for r in group)
            summary.append(row)
        for pi,(a,b) in enumerate(PAIRS):
            rng=np.random.default_rng(np.random.SeedSequence([config.seed,li,pi,902]))
            comparisons.append({"lambda_index":li,"lambda":group[0]["lambda"],
                                "method_a":a,"method_b":b,"n":len(group),
                                **paired_comparison(cost_arrays[a],cost_arrays[b],rng,config.bootstrap_resamples)})
    common._write_csv(output/"summary.csv",summary)
    common._write_csv(output/"comparisons.csv",comparisons)
    common._write_json(output/"crossings.json",crossover_brackets(comparisons,config.samples))
    return summary,comparisons


def plot_saved(input_dir, figure_dir=None, dpi=240):
    """Only read saved CSVs: no sampling, optimization, or statistics rerun."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    input_dir=Path(input_dir)
    figure_dir=Path(figure_dir) if figure_dir else input_dir/"figures"
    with (input_dir/"summary.csv").open(encoding="utf-8-sig",newline="") as f: summary=list(csv.DictReader(f))
    with (input_dir/"comparisons.csv").open(encoding="utf-8-sig",newline="") as f: comparisons=list(csv.DictReader(f))
    if not summary or not comparisons:raise ValueError("saved data tables are empty")
    figure_dir.mkdir(parents=True,exist_ok=True)
    colors={"rgs":"#0072B2","naive":"#D55E00","ss":"#009E73"}
    labels={"rgs":"Random Grover search","naive":"Naïve nested AA","ss":"Schrottenloher-Stevens"}
    markers={"rgs":"o","naive":"s","ss":"^"}
    outputs=[]
    with plt.rc_context({"font.family":"DejaVu Sans","font.size":11,
                         "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42}):
        for relative,stem,ylabel in ((False,"oracle_cost_vs_lambda","Mean total oracle cost"),
                                     (True,"relative_cost_vs_lambda","Ratio of mean costs to RGS")):
            fig,ax=plt.subplots(figsize=(6.5,4.3),layout="constrained")
            if relative:ax.axhline(1,color=colors["rgs"],linewidth=1.6,label="RGS baseline")
            for method in (("naive","ss") if relative else METHODS):
                rows=sorted((r for r in comparisons if r["method_a"]==method and r["method_b"]=="rgs"),
                            key=lambda r:int(r["lambda_index"])) if relative else sorted(
                    (r for r in summary if r["method"]==method),key=lambda r:int(r["lambda_index"]))
                columns=("ratio_of_means","ratio_ci_low","ratio_ci_high") if relative else (
                    "oracle_cost_mean","oracle_cost_ci_low","oracle_cost_ci_high")
                x=np.array([float(r["lambda"]) for r in rows])
                y,low,high=([float(r[c]) for r in rows] for c in columns)
                ax.plot(x,y,color=colors[method],marker=markers[method],markevery=max(1,len(x)//10),
                        markersize=4,linewidth=1.7,label=labels[method])
                ax.fill_between(x,low,high,color=colors[method],alpha=.16,linewidth=0)
            ax.set(xscale="log",yscale="log",xlabel=r"Oracle cost ratio $\lambda=c_1/c_0$",ylabel=ylabel)
            ax.grid(True,which="major",alpha=.25,linewidth=.6)
            ax.legend(frameon=False,fontsize=9)
            for ext in ("pdf","png"):
                path=figure_dir/(stem+"."+ext);fig.savefig(path,dpi=dpi);outputs.append(path)
            plt.close(fig)
    return outputs


def run_experiment(config,output,resume=False,make_plots=True):
    config.validate()
    import scipy.optimize
    import nested_amplification as nested
    output=Path(output);metadata_path=output/"metadata.json";record_path=output/"results.jsonl"
    hashes={"experiment_unequal_cost.py":common._source_hash(__file__),
            "experiment_equal_cost.py":common._source_hash(common.__file__),
            "nested_amplification.py":common._source_hash(nested.__file__)}
    if metadata_path.exists():
        if not resume:raise FileExistsError("Output exists; use --resume or a new output directory")
        metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("protocol_id")!=PROTOCOL or metadata["config"]!=asdict(config) or metadata["source_sha256"]!=hashes:
            raise ValueError("resume requires the same protocol, configuration, and source code")
        cohort=json.loads((output/"instances.json").read_text(encoding="utf-8"))
    else:
        if record_path.exists():raise ValueError("checkpoint exists without metadata")
        output.mkdir(parents=True,exist_ok=True);cohort=generate_cohort(config)
        metadata={"schema_version":1,"protocol_id":PROTOCOL,"completed":False,
                  "created_utc":datetime.now(timezone.utc).isoformat(),"config":asdict(config),
                  "source_sha256":hashes,"lambda_grid":config.lambdas(),
                  "environment":{"python":sys.version,"platform":platform.platform(),"processor":platform.processor(),
                                 "packages":{x:importlib.metadata.version(x) for x in ("numpy","scipy","matplotlib")}},
                  "protocol":{"instances":"One fixed cohort reused at all cost ratios; no sorting or cost-dependent relabeling of sets.",
                              "rgs":"Paper's biased distribution, optimal only under its sufficient success constraints; full denominator retained; peak stopping time; success lower bound, not simulation.",
                              "nested":"Naive reoptimizes order at every lambda; SS follows it, one oracle per level, and reoptimizes counts with final amplify; best found, not globally certified.",
                              "timing":common.TIMING_PROTOCOL,
                              "excluded":"Cardinality acquisition, instance generation/validation, quantum simulation, imports, I/O, statistics and plotting.",
                              "relative_cost":"Ratio of paired sample mean costs; confidence intervals use paired bootstrap resampling.",
                              "crossings":"Brackets from adjacent scanned mean-cost orderings, not precise theoretical thresholds."}}
        common._write_json(output/"instances.json",cohort);common._write_json(metadata_path,metadata)
    warm=nested.SearchProblem([64,2,2,1],[1,1]);ref=nested.optimize_naive(warm,config.target_success)
    nested.optimize_ss_for_order(warm,ref.order,config.target_success,maxiter=config.ss_maxiter)
    records=load_records(record_path)
    for li,lam in enumerate(config.lambdas()):
        for instance in cohort:
            key=(li,instance["instance"])
            if key in records and records[key]["status"]=="ok":continue
            problem=nested.SearchProblem(instance["intersection_sizes"],[1,lam])
            row={**instance,"lambda_index":li,"lambda":lam,"costs":[1,lam],"status":"running","methods":{}}
            order=(("rgs","naive","ss_extra"),("naive","rgs","ss_extra"),
                   ("naive","ss_extra","rgs"))[instance["instance"]%3]
            row["measurement_order"]=list(order);reference=None
            try:
                for stage in order:
                    if stage=="rgs":
                        function=lambda:biased_rgs(config.N,instance["a0"],instance["a1"],config.r,lam,config.target_success)
                    elif stage=="naive":function=lambda:nested.optimize_naive(problem,config.target_success)
                    else:function=lambda:nested.optimize_ss_for_order(problem,reference.order,config.target_success,maxiter=config.ss_maxiter)
                    result,timing=common.measure_strategy(function,config.timing_repeats,config.timing_min_seconds)
                    method="ss" if stage=="ss_extra" else stage
                    if stage=="naive":reference=result
                    result=result if isinstance(result,dict) else result.as_dict()
                    if stage=="ss_extra":
                        timing=common.combine_ss_timing(row["methods"]["naive"]["timing"],timing)
                        result["naive_target_success"]=config.target_success
                    guarantee=result.get("success_lower_bound",result["success_probability"])
                    if guarantee<config.target_success-8*math.ulp(config.target_success):raise ValueError(method+" fails success target")
                    row["methods"][method]={"result":result,"timing":timing}
                row["status"]="ok"
            except Exception as exc:
                row.update(status="error",error=type(exc).__name__+": "+str(exc))
                common._append_record(record_path,row);records[key]=row;export_tables(output,records,config)
                raise RuntimeError("lambda_index=%d instance=%d failed and was saved without resampling: %s"%(li,instance["instance"],exc)) from exc
            common._append_record(record_path,row);records[key]=row
        summary,_=export_tables(output,records,config)
        point={r["method"]:r["oracle_cost_mean"] for r in summary if r["lambda_index"]==li}
        print("lambda=%10.5g [%d/%d] | mean costs RGS=%.7g naive=%.7g SS=%.7g | lowest=%s"%
              (lam,li+1,config.lambda_points,point["rgs"],point["naive"],point["ss"],min(point,key=point.get)),flush=True)
    metadata["completed"]=True;common._write_json(metadata_path,metadata)
    if make_plots:
        for path in plot_saved(output):print("Saved "+str(path),flush=True)
    print("Data saved in "+str(output.resolve()),flush=True)
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    commands=parser.add_subparsers(dest="command",required=True)
    run=commands.add_parser("run",help="Run, checkpoint, summarize, and plot")
    run.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    run.add_argument("--resume",action="store_true");run.add_argument("--no-plot",action="store_true")
    defaults=Config()
    for field,value in asdict(defaults).items():
        if field=="size_distribution":run.add_argument("--size-distribution",choices=("loguniform","uniform"),default=value)
        else:run.add_argument("--"+field.replace("_","-"),dest=field,type=type(value),default=value)
    plot=commands.add_parser("plot",help="Plot saved CSV data without rerunning experiments")
    plot.add_argument("--input",type=Path,default=DEFAULT_OUTPUT);plot.add_argument("--figure-dir",type=Path)
    plot.add_argument("--dpi",type=int,default=240)
    args=parser.parse_args()
    if args.command=="plot":
        for path in plot_saved(args.input,args.figure_dir,args.dpi):print("Saved "+str(path))
    else:run_experiment(Config(**{k:getattr(args,k) for k in asdict(defaults)}),args.output,args.resume,not args.no_plot)


if __name__=="__main__":main()
