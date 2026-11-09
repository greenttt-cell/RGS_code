"""Three-method comparison versus oracle number k and cost heterogeneity kappa.

    python experiment_cost_heterogeneity.py run
    python experiment_cost_heterogeneity.py plot --input results/cost_heterogeneity

Costs have mean 1 and kappa=mean(cost)/min(cost). Three profiles share the same
kappa: one cheap oracle, one expensive oracle, and a linear cost gradient.
Reuse one fixed random set cohort and cost-to-oracle permutation at every
kappa and for every profile. Cardinalities are known; quantum trajectories
are not simulated. All three methods use the same success threshold.

RGS minimizes cost under the manuscript's sufficient probability constraints.
Naive nesting optimizes its order by exact subset DP. SS follows that order,
one predicate per level, and optimizes counts including final amplification.
SS is a numerical strategy, not generally a globally optimal SS solution.

Checkpoint every case, save full evaluations and paired-bootstrap summaries.
Plot reads CSVs only. Times are single-call component diagnostics, NOT the
repeated timing benchmarks in experiment_equal_cost.py.
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
import time
import warnings

import numpy as np
import experiment_equal_cost as common


PROFILES = ("one_cheap", "one_expensive", "gradual")
PROFILE_LABELS = {"one_cheap": "One cheap oracle",
                  "one_expensive": "One expensive oracle",
                  "gradual": "Linear cost gradient"}
METHODS = ("rgs", "naive", "ss")
PAIRS = (("naive", "rgs"), ("ss", "rgs"), ("ss", "naive"))
PROTOCOL = "multi_oracle_mean_normalized_cost_heterogeneity_v1"
DEFAULT_OUTPUT = Path("results/cost_heterogeneity")


@dataclass(frozen=True)
class Config:
    N: int = 2**48
    r: int = 1
    k_min: int = 2
    k_max: int = 15
    samples: int = 50
    union_min: int = 256
    union_max: int = 1024
    dirichlet_alpha: float = 1.0
    kappa_min: float = 1.0
    kappa_max: float = 1000.0
    kappa_points: int = 13
    target_success: float = .99
    seed: int = 20261028
    ss_maxiter: int = 50000
    bootstrap_resamples: int = 2000

    def validate(self):
        for name in ("N", "r", "k_min", "k_max", "samples", "kappa_points",
                     "ss_maxiter", "bootstrap_resamples"):
            if getattr(self, name) <= 0:
                raise ValueError(name + " must be positive")
        if not 2 <= self.k_min <= self.k_max:
            raise ValueError("need 2 <= k_min <= k_max")
        if not 0 < self.r < self.union_min <= self.union_max < self.N:
            raise ValueError("need 0 < r < union_min <= union_max < N")
        if self.seed < 0 or not 0 < self.target_success < 1:
            raise ValueError("need seed >= 0 and 0 < target_success < 1")
        if not math.isfinite(self.dirichlet_alpha) or self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be finite and positive")
        if (not math.isfinite(self.kappa_min) or not math.isfinite(self.kappa_max)
                or not 1 <= self.kappa_min <= self.kappa_max):
            raise ValueError("need 1 <= kappa_min <= kappa_max < infinity")
        if (self.kappa_points == 1) != (self.kappa_min == self.kappa_max):
            raise ValueError("use one kappa point exactly when kappa_min == kappa_max")
        # Check the whole sampling domain, rather than resampling failures.
        rgs_parameters(self.N, self.r, self.union_max, self.k_max, self.target_success)
        for profile in PROFILES:
            cost_vector(self.k_max, self.kappa_max, profile, list(range(self.k_max)))

    def kappas(self):
        values = np.geomspace(self.kappa_min, self.kappa_max, self.kappa_points)
        values[0], values[-1] = self.kappa_min, self.kappa_max
        return [float(x) for x in values]


def cost_vector(k, kappa, profile, permutation):
    """Assign ascending normalized costs to ORIGINAL oracle IDs in permutation.

    Before division by kappa the profiles are (1,a,...,a), (1,...,1,b),
    and linspace(1,2*kappa-1,k), respectively. Here a=(k*kappa-1)/(k-1)
    and b=1+k*(kappa-1). Stable normalized expressions avoid large products.
    """
    if (k < 2 or not math.isfinite(kappa) or kappa < 1
            or sorted(permutation) != list(range(k))):
        raise ValueError("need k>=2, finite kappa>=1, and a permutation of oracle IDs")
    low, spread = 1/kappa, 1-1/kappa
    if profile == "one_cheap":
        ordered = [low] + [1 + spread/(k-1)]*(k-1)
    elif profile == "one_expensive":
        ordered = [low]*(k-1) + [1 + (k-1)*spread]
    elif profile == "gradual":
        ordered = [low + 2*spread*i/(k-1) for i in range(k)]
    else:
        raise ValueError("unknown cost profile: " + profile)
    costs = [0.0]*k
    for oracle, cost in zip(permutation, ordered):
        costs[oracle] = cost
    if min(costs) <= 0 or not all(math.isfinite(x) for x in costs):
        raise ValueError("cost vector is outside floating-point range")
    return costs


def cost_statistics(costs):
    mean = math.fsum(costs)/len(costs)
    return {"mean_cost": mean, "min_cost": min(costs), "max_cost": max(costs),
            "kappa_actual": mean/min(costs), "max_min_ratio": max(costs)/min(costs),
            "cv": math.sqrt(math.fsum((x/mean-1)**2 for x in costs)/len(costs))}


def rgs_parameters(N, r, m, k, target_success):
    """Stopping time T and minimum p, using the full finite-parameter bound."""
    result = common.rgs_strategy(N, r, m, k, target_success)
    base = 4*r/N + 2*(m-r)/math.sqrt(N*r)
    numerator = 4*(m-r)/math.sqrt(N*r) + 2*math.sqrt((m-r)/N)
    denominator = 1-target_success-base
    if numerator <= 0 or denominator <= 0:
        raise ValueError("the biased RGS sufficient constraints are not feasible")
    p = numerator/denominator
    if p > 1/k:
        raise ValueError("RGS needs p_min<=1/k: increase N or reduce union_max or k_max")
    p = min(1/k, math.nextafter(p, math.inf))
    bound = math.fsum((1, -base, -numerator/p))
    if bound < target_success - 8*math.ulp(target_success):
        raise ValueError("RGS success bound falls below target")
    return {"steps": result["steps"], "probability_lower_constraint": p,
            "success_lower_bound": bound, "target_success": target_success}


def biased_rgs(parameters, costs, cheapest):
    """All oracles get p; remaining mass goes to a cheapest oracle (LP optimum)."""
    k, p = len(costs), parameters["probability_lower_constraint"]
    if not 0 <= cheapest < k or costs[cheapest] != min(costs) or not 0 < p <= 1/k:
        raise ValueError("invalid cheapest oracle or probability constraint")
    probabilities = [p]*k
    probabilities[cheapest] = 1-(k-1)*p
    T = parameters["steps"]
    queries = [T*x for x in probabilities]
    weighted = math.fsum(q*c for q, c in zip(queries, costs))
    # C/T = p*sum(c)+(1-k*p)*min(c). Stable at the uniform boundary.
    formula = T*(p*math.fsum(costs) + max(0.0, 1-k*p)*min(costs))
    if not math.isclose(weighted, formula, rel_tol=3e-15, abs_tol=1e-12):
        raise ArithmeticError("RGS weighted-query identity failed")
    return {**parameters, "method": "rgs", "completion": "none",
            "sampling_probabilities": probabilities, "query_counts": queries,
            "query_counts_are_expectations": True, "weighted_cost": weighted,
            "cost_from_identity": formula, "cheapest_oracle": cheapest,
            "success_probability": None,
            "success_certificate": "theorem_biased_sampling_lower_bound",
            "probability_optimality": "optimal_under_theorem_sufficient_constraints"}


def generate_cohort(config):
    cohort = []
    for k in range(config.k_min, config.k_max+1):
        for instance in range(config.samples):
            row = common.generate_instance(config, k, instance)
            seed = [config.seed, k, instance, 403]
            rng = np.random.default_rng(np.random.SeedSequence(seed))
            row["cost_permutation"] = [int(x) for x in rng.permutation(k)]
            row["cost_permutation_seed"] = seed
            row["rgs_parameters"] = rgs_parameters(config.N, config.r, row["m"], k,
                                                   config.target_success)
            cohort.append(row)
    return cohort


def record_key(row):
    return row["k"], row["kappa_index"], row["profile"], row["instance"]


def load_records(path, repair_tail=False):
    """Last retry wins. --resume may discard only an incomplete final JSON line."""
    records = {}
    if not path.exists():
        return records
    with path.open("rb+" if repair_tail else "rb") as handle:
        number = 0
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            number += 1
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                if repair_tail and not line.endswith(b"\n"):
                    handle.seek(offset)
                    handle.truncate()
                    warnings.warn("Discarded incomplete final checkpoint line %d" % number)
                    break
                raise ValueError("invalid checkpoint line %d" % number) from exc
            if row["status"] not in ("ok", "error"):
                raise ValueError("invalid checkpoint status on line %d" % number)
            records[record_key(row)] = row
            if repair_tail and not line.endswith(b"\n"):
                handle.write(b"\n")
    return records


def paired_comparison(a, b, rng, resamples):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if (a.ndim != 1 or a.shape != b.shape or not len(a)
            or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))
            or np.any(a <= 0) or np.any(b <= 0)):
        raise ValueError("need paired, finite, positive cost arrays")
    ratios = []
    for start in range(0, resamples, 200):
        indices = rng.integers(0, len(a), (min(200, resamples-start), len(a)))
        ratios.extend(a[indices].mean(axis=1)/b[indices].mean(axis=1))
    low, high = np.quantile(ratios, [.025, .975])
    tied = np.isclose(a, b, rtol=1e-12, atol=0)
    wins_a = int(np.sum((a < b) & ~tied))
    wins_b = int(np.sum((b < a) & ~tied))
    return {"ratio_of_means": float(a.mean()/b.mean()), "ratio_ci_low": float(low),
            "ratio_ci_high": float(high), "wins_a": wins_a, "wins_b": wins_b,
            "ties": int(tied.sum()), "win_fraction_a": wins_a/len(a),
            "win_fraction_b": wins_b/len(a)}


def export_tables(output, records, config):
    """Partial groups are labeled incomplete and excluded from heatmaps."""
    rows, groups = [], {}
    for _, record in sorted(records.items()):
        if record["status"] != "ok":
            continue
        group_key = record["k"], record["kappa_index"], record["profile"]
        groups.setdefault(group_key, []).append(record)
        for method in METHODS:
            result = record["methods"][method]
            timing = record["planning_seconds"]
            rows.append({"k": record["k"], "kappa_index": record["kappa_index"],
                         "kappa": record["kappa"], "profile": record["profile"],
                         "instance": record["instance"], "method": method,
                         "N": config.N, "r": config.r, "m": record["m"],
                         **record["cost_statistics"],
                         "costs": json.dumps(record["costs"]),
                         "cost_permutation": json.dumps(record["cost_permutation"]),
                         "oracle_cost": result["weighted_cost"],
                         "query_counts": json.dumps(result["query_counts"]),
                         "sampling_probabilities": json.dumps(result.get("sampling_probabilities", [])),
                         "p_min": result.get("probability_lower_constraint", ""),
                         "order": json.dumps(result.get("order", [])),
                         "iterations": json.dumps(result.get("iterations", [])),
                         "outer_iterations": result.get("outer_iterations", ""),
                         "completion": result["completion"],
                         "success_lower_bound": result.get("success_lower_bound", result["success_probability"]),
                         "optimality": result.get("optimality", result.get("probability_optimality")),
                         "planning_seconds": timing[method],
                         "order_seconds": timing["naive"] if method != "rgs" else 0,
                         "ss_extra_seconds": timing["ss_extra"] if method == "ss" else 0})
    common._write_csv(output/"per_instance.csv", rows)
    summary, comparisons = [], []
    for (k, ki, profile), group in sorted(groups.items()):
        group.sort(key=lambda x: x["instance"])
        complete = len(group) == config.samples
        head = {"k": k, "kappa_index": ki, "kappa": group[0]["kappa"],
                "profile": profile, **group[0]["cost_statistics"],
                "n": len(group), "expected_n": config.samples, "complete": int(complete)}
        arrays = {m: [r["methods"][m]["weighted_cost"] for r in group] for m in METHODS}
        profile_index = PROFILES.index(profile)
        for mi, method in enumerate(METHODS):
            rng = np.random.default_rng(np.random.SeedSequence([config.seed, k, ki, profile_index, mi, 801]))
            estimate = common._mean_interval(arrays[method], rng, config.bootstrap_resamples)
            summary.append({**head, "method": method,
                            **dict(zip(("oracle_cost_mean", "oracle_cost_std", "oracle_cost_ci_low",
                                        "oracle_cost_ci_high"), estimate)),
                            "success_lower_bound_min": min(
                                r["methods"][method].get("success_lower_bound", r["methods"][method]["success_probability"])
                                for r in group),
                            "planning_seconds_mean": float(np.mean([r["planning_seconds"][method] for r in group]))})
        for pi, (a, b) in enumerate(PAIRS):
            rng = np.random.default_rng(np.random.SeedSequence([config.seed, k, ki, profile_index, pi, 802]))
            comparisons.append({**head, "method_a": a, "method_b": b,
                                **paired_comparison(arrays[a], arrays[b], rng, config.bootstrap_resamples)})
    common._write_csv(output/"summary.csv", summary)
    common._write_csv(output/"comparisons.csv", comparisons)
    return summary, comparisons


def _edges(values, logarithmic=False):
    values = np.asarray(values, float)
    if logarithmic:
        return 10**_edges(np.log10(values))
    if len(values) == 1:
        return np.array([values[0]-.5, values[0]+.5])
    mid = (values[1:]+values[:-1])/2
    return np.r_[values[0]-(mid[0]-values[0]), mid, values[-1]+(values[-1]-mid[-1])]


def plot_saved(input_dir, figure_dir=None, dpi=240):
    """CSV-only heatmaps; blank cells are incomplete, never interpolated."""
    input_dir = Path(input_dir)
    with (input_dir/"comparisons.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("no saved comparisons to plot")
    ks = sorted({int(row["k"]) for row in rows})
    kappas = sorted({float(row["kappa"]) for row in rows})
    matrices = {}
    for method in ("naive", "ss"):
        for profile in PROFILES:
            grid = np.full((len(kappas), len(ks)), np.nan)
            for row in rows:
                if (row["method_a"] == method and row["method_b"] == "rgs"
                        and row["profile"] == profile and int(row["complete"])):
                    ratio = float(row["ratio_of_means"])
                    if not math.isfinite(ratio) or ratio <= 0:
                        raise ValueError("invalid cost ratio in saved comparisons")
                    grid[kappas.index(float(row["kappa"])) , ks.index(int(row["k"]))] = math.log10(ratio)
            matrices[method, profile] = grid
    finite = np.concatenate([a[np.isfinite(a)] for a in matrices.values()])
    if not len(finite):
        raise ValueError("no complete cohort groups to plot; resume the experiment first")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.ticker import FuncFormatter
    figure_dir = Path(figure_dir) if figure_dir else input_dir/"figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    limit = max(.05, float(np.max(np.abs(finite))))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e5e7eb")
    outputs = []
    for method in ("naive", "ss"):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True,
                                 layout="constrained")
        log_kappas = np.log10(kappas)
        for ax, profile in zip(axes, PROFILES):
            grid = matrices[method, profile]
            # Interpolate the equality contour in log(kappa), matching the
            # logarithmic scan rather than interpolating in raw kappa.
            mesh = ax.pcolormesh(_edges(ks), _edges(log_kappas), np.ma.masked_invalid(grid),
                                 cmap=cmap, norm=norm, shading="flat", rasterized=True)
            if (len(ks) > 1 and len(kappas) > 1 and np.any(grid < 0) and np.any(grid > 0)):
                ax.contour(ks, log_kappas, np.ma.masked_invalid(grid), levels=[0],
                           colors="#20252b", linewidths=1)
            powers = np.arange(math.ceil(log_kappas[0]), math.floor(log_kappas[-1])+1)
            if not len(powers):
                powers = log_kappas[::max(1, math.ceil(len(kappas)/5))]
            ax.set_yticks(powers[::max(1, math.ceil(len(powers)/7))])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: r"$10^{%g}$" % x))
            if len(kappas) > 1:
                ax.set_ylim(log_kappas[0], log_kappas[-1])
            ax.set_xlabel("Number of oracles, k")
            ax.set_title(PROFILE_LABELS[profile], fontsize=11)
            ax.set_xticks(ks[::max(1, math.ceil(len(ks)/8))])
            ax.tick_params(labelsize=9)
        axes[0].set_ylabel(r"Cost heterogeneity, $\kappa=\bar c/c_{\min}$")
        bar = fig.colorbar(mesh, ax=axes, pad=.02, fraction=.035)
        bar.set_ticks([-limit, -limit/2, 0, limit/2, limit])
        bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "%.3g" % (10**x)))
        bar.set_label("Ratio of mean oracle costs (log color scale)", fontsize=10)
        label = "Naïve nested AA" if method == "naive" else "Schrottenloher-Stevens"
        fig.suptitle(label + " / Random Grover search", fontsize=13)
        fig.supxlabel("Blue: nested cheaper (<1)    |    Red: RGS cheaper (>1)\n"
                      "Black contour: ratio 1; gray cells: incomplete data", fontsize=9)
        stem = figure_dir/("relative_cost_" + method + "_heatmap")
        for extension in ("pdf", "png"):
            path = stem.with_suffix("." + extension)
            fig.savefig(path, dpi=dpi)
            outputs.append(path)
        plt.close(fig)
    return outputs


def _verify_result(result, costs, target):
    guarantee = result.get("success_lower_bound", result["success_probability"])
    if not math.isfinite(guarantee) or guarantee < target-8*math.ulp(target):
        raise ValueError(result["method"] + " does not meet the requested success threshold")
    cost = result["weighted_cost"]
    if not math.isfinite(cost) or cost <= 0 or len(result["query_counts"]) != len(costs):
        raise ValueError("invalid weighted cost or query count vector")
    if not math.isclose(cost, math.fsum(q*c for q, c in zip(result["query_counts"], costs)), rel_tol=1e-12):
        raise ValueError("cost and query counts disagree")


def run_experiment(config, output, resume=False, make_plots=True):
    config.validate()
    import nested_amplification as nested
    import scipy.optimize  # Imports excluded from single-call planning diagnostics.
    output = Path(output)
    metadata_path, record_path = output/"metadata.json", output/"results.jsonl"
    hashes = {Path(__file__).name: common._source_hash(__file__),
              "experiment_equal_cost.py": common._source_hash(common.__file__),
              "nested_amplification.py": common._source_hash(nested.__file__)}
    packages = {p: importlib.metadata.version(p) for p in ("numpy", "scipy")}
    if metadata_path.exists():
        if not resume:
            raise FileExistsError("output exists: use --resume or another output directory")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata.get("protocol_id") != PROTOCOL or metadata["config"] != asdict(config)
                or metadata["source_sha256"] != hashes or metadata["packages"] != packages):
            raise ValueError("resume requires identical protocol, configuration, source, NumPy and SciPy")
        if common._source_hash(output/"instances.json") != metadata["cohort_sha256"]:
            raise ValueError("saved instance/permutation data changed")
        cohort = json.loads((output/"instances.json").read_text(encoding="utf-8"))
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("new run requires an empty output directory")
        output.mkdir(parents=True, exist_ok=True)
        cohort = generate_cohort(config)
        common._write_json(output/"instances.json", cohort)
        metadata = {"schema_version": 1, "protocol_id": PROTOCOL, "completed": False,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "config": asdict(config), "source_sha256": hashes, "packages": packages,
                    "cohort_sha256": common._source_hash(output/"instances.json"),
                    "kappa_grid": config.kappas(), "profiles": list(PROFILES),
                    "environment": {"python": sys.version, "platform": platform.platform()},
                    "protocol": {
                        "sets": "Same Dirichlet-multinomial union-region cohort as equal-cost experiment, fixed per k; m uniform on union_min..union_max; no rejected/resampled cases.",
                        "costs": "Mean 1; kappa=mean/min; one_cheap, one_expensive, linear gradual. Ascending costs mapped to a seeded random permutation of original oracle IDs, fixed across all kappa and profiles.",
                        "rgs": "Minimize cost under theorem sufficient constraints; full p denominator and peak stopping time; certified lower bound, no quantum simulation.",
                        "nested": "Exact naive order DP per cost vector; SS follows that order, one test per level, integer-count optimization including final amplify. SS generally best found, not globally certified.",
                        "timing": "Single-call diagnostics only: naive=order optimization, SS=that same order time plus SS extra time. Excludes instance generation/validation, cardinality acquisition, imports, I/O and plots. Not repeated timing benchmarks.",
                        "statistics": "Ratio of sample mean costs, paired percentile bootstrap 95% CI, and per-instance win fractions. Incomplete cohort groups are explicitly marked and excluded from plots.",
                        "heatmaps": "Two figures, each with three cost-profile panels. Shared log color scale centered at ratio 1; contours visualize mean-cost equality between sampled cells, not certified thresholds.",
                        "scope": "This fixed union-size ensemble may contain redundant filters at large k; conclusions are specific to the ensemble, not a universal scaling claim."}}
        common._write_json(metadata_path, metadata)
    records = load_records(record_path, repair_tail=resume)
    valid_instances = {(r["k"], r["instance"]): r for r in cohort}
    grid = config.kappas()
    for key, row in records.items():
        k, ki, profile, instance = key
        if ((k, instance) not in valid_instances or not 0 <= ki < len(grid)
                or profile not in PROFILES or row["kappa"] != grid[ki]):
            raise ValueError("checkpoint contains a case outside the saved experiment")
        source = valid_instances[k, instance]
        if (row["cost_permutation"] != source["cost_permutation"]
                or row["costs"] != cost_vector(k, grid[ki], profile, source["cost_permutation"])):
            raise ValueError("checkpoint cost assignment disagrees with saved cohort")
    expected = len(cohort)*len(grid)*len(PROFILES)
    print("Cases: %d total, %d already complete. Checkpoint after every case." %
          (expected, sum(r["status"] == "ok" for r in records.values())), flush=True)
    if not metadata["completed"]:
        # Warm up SciPy outside the recorded per-call diagnostic times.
        warm = nested.SearchProblem([64, 2, 2, 1], [1, 1])
        ref = nested.optimize_naive(warm, config.target_success)
        nested.optimize_ss_for_order(warm, ref.order, config.target_success, maxiter=config.ss_maxiter)
    try:
        for k in range(config.k_min, config.k_max+1):
            # Convert known region sizes once per instance; excluded from diagnostics.
            selected = [r for r in cohort if r["k"] == k]
            tables = {r["instance"]: nested.SearchProblem.from_blocks(r["block_sizes"]).intersection_sizes
                      for r in selected}
            for ki, kappa in enumerate(grid):
                for profile in PROFILES:
                    costs_group = []
                    for source in selected:
                        instance = source["instance"]
                        key = k, ki, profile, instance
                        if key not in records or records[key]["status"] != "ok":
                            permutation = source["cost_permutation"]
                            costs = cost_vector(k, kappa, profile, permutation)
                            problem = nested.SearchProblem(tables[instance], costs)
                            row = {"k": k, "kappa_index": ki, "kappa": kappa, "profile": profile,
                                   "instance": instance, "m": source["m"], "costs": costs,
                                   "cost_permutation": permutation, "cost_statistics": cost_statistics(costs),
                                   "status": "error", "methods": {}, "planning_seconds": {}}
                            try:
                                started = time.perf_counter()
                                rgs = biased_rgs(source["rgs_parameters"], costs, permutation[0])
                                row["planning_seconds"]["rgs"] = time.perf_counter()-started
                                row["methods"]["rgs"] = rgs
                                started = time.perf_counter()
                                naive = nested.optimize_naive(problem, config.target_success)
                                row["planning_seconds"]["naive"] = time.perf_counter()-started
                                row["methods"]["naive"] = naive.as_dict()
                                started = time.perf_counter()
                                ss = nested.optimize_ss_for_order(problem, naive.order, config.target_success,
                                                                   maxiter=config.ss_maxiter).as_dict()
                                row["planning_seconds"]["ss_extra"] = time.perf_counter()-started
                                row["planning_seconds"]["ss"] = (row["planning_seconds"]["naive"]
                                                                              + row["planning_seconds"]["ss_extra"])
                                ss["naive_target_success"] = config.target_success
                                row["methods"]["ss"] = ss
                                for result in row["methods"].values():
                                    _verify_result(result, costs, config.target_success)
                                row["status"] = "ok"
                            except Exception as exc:
                                row["error"] = type(exc).__name__ + ": " + str(exc)
                                common._append_record(record_path, row)
                                records[key] = row
                                raise RuntimeError("case %s failed and was saved; no resampling: %s" % (key, exc)) from exc
                            common._append_record(record_path, row)
                            records[key] = row
                        costs_group.append([records[key]["methods"][m]["weighted_cost"] for m in METHODS])
                    averages = np.mean(costs_group, axis=0)
                    print("k=%2d kappa=%9.4g %-13s | means: RGS=%.6g naive=%.6g SS=%.6g" %
                          (k, kappa, profile, *averages), flush=True)
    except (Exception, KeyboardInterrupt):
        export_tables(output, records, config)
        metadata["completed"] = False
        common._write_json(metadata_path, metadata)
        raise
    export_tables(output, records, config)
    metadata["completed"] = len(records) == expected and all(r["status"] == "ok" for r in records.values())
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    common._write_json(metadata_path, metadata)
    if make_plots:
        for path in plot_saved(output):
            print("Saved " + str(path), flush=True)
    print("Data saved in " + str(output.resolve()), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run, checkpoint, summarize, and draw heatmaps")
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--no-plot", action="store_true")
    defaults = Config()
    for field, value in asdict(defaults).items():
        run.add_argument("--" + field.replace("_", "-"), dest=field, type=type(value), default=value)
    plot = commands.add_parser("plot", help="Redraw saved CSVs, without importing the nested optimizer")
    plot.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    plot.add_argument("--figure-dir", type=Path)
    plot.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()
    if args.command == "plot":
        for path in plot_saved(args.input, args.figure_dir, args.dpi):
            print("Saved " + str(path))
    else:
        config = Config(**{k: getattr(args, k) for k in asdict(defaults)})
        run_experiment(config, args.output, args.resume, not args.no_plot)


if __name__ == "__main__":
    main()
