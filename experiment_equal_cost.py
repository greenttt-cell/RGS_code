"""Equal-cost comparison of RGS, naive nesting, and Schrottenloher--Stevens.

Run the experiment once (defaults: k=1..18, 50 instances per k):
    python experiment_equal_cost.py run --output results/equal_cost_components
Redraw its two figures without generating instances or optimizing again:
    python experiment_equal_cost.py plot --input results/equal_cost_components

Every oracle costs one Boolean query. All methods meet target_success.
RGS uses uniform sampling and the manuscript's stopping time / success bound;
its quantum evolution is NOT simulated. Nested methods use exact sizes.
SS uses optimize_naive's order, one oracle per level, and completion='amplify'.
Classical timings are component-based: measure the common naive-order search
once, then measure SS's additional optimization for that same order. Report
T_naive=T_order and T_SS=T_order+T_extra. The total is an additive estimate,
not a separately measured end-to-end SS call. Raw component timings are saved.
Exclude instance generation, conversion/validation of known cardinalities,
package imports, quantum simulation, serialization, statistics, and plotting.

Each completed instance is flushed to instances.jsonl immediately. This file
contains all region/intersection sizes, full method results, and timing samples.
per_instance.csv and summary.csv are derived tables. The plot command reads
ONLY summary.csv; it neither imports the nested optimizer nor runs experiments.
--resume continues an interrupted run using the same configuration and code.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np


METHODS = ("rgs", "naive", "ss")
LABELS = {"rgs": "Random Grover search", "naive": "Naïve nested AA", "ss": "Schrottenloher-Stevens"}
TIMING_PROTOCOL = "shared_order_plus_ss_extra_v1"
DEFAULT_OUTPUT = Path("results/equal_cost_components")


@dataclass(frozen=True)
class Config:
    N: int = 2**48
    r: int = 1
    k_min: int = 1
    k_max: int = 18
    samples: int = 50
    target_success: float = 0.99
    union_min: int = 256
    union_max: int = 1024
    dirichlet_alpha: float = 1.0
    seed: int = 20261028
    timing_repeats: int = 3
    timing_min_seconds: float = 0.005
    ss_maxiter: int = 50000
    bootstrap_resamples: int = 2000

    def validate(self):
        for name in ("N", "r", "k_min", "k_max", "samples", "timing_repeats",
                     "ss_maxiter", "bootstrap_resamples"):
            if getattr(self, name) <= 0:
                raise ValueError(name + " must be positive")
        if not 1 <= self.k_min <= self.k_max or not self.r < self.N:
            raise ValueError("need 1 <= k_min <= k_max and 0 < r < N")
        if self.seed < 0 or not 0 < self.target_success < 1:
            raise ValueError("need seed >= 0 and 0 < target_success < 1")
        if not math.isfinite(self.dirichlet_alpha) or self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be finite and positive")
        if not math.isfinite(self.timing_min_seconds) or self.timing_min_seconds <= 0:
            raise ValueError("timing_min_seconds must be finite and positive")
        if self.k_max >= 2:
            if not self.r < self.union_min <= self.union_max < self.N:
                raise ValueError("need r < union_min <= union_max < N")
            # Reject an unsuitable experimental regime, rather than silently
            # accepting only the easy instances from a broader distribution.
            rgs_strategy(self.N, self.r, self.union_max, self.k_max, self.target_success)
        if self.k_min == 1:
            rgs_strategy(self.N, self.r, self.r, 1, self.target_success)


def rgs_strategy(N, r, m, k, target_success):
    """Uniform RGS: the paper's integer stopping time and certified success.

    For k>=2 the returned success_lower_bound is Theorem 3.5 with p=1/k,
    not an empirical or exact success probability. k=1 is ordinary Grover.
    Integer products are formed before division to retain small r/N accuracy.
    """
    if not (1 <= k and 0 < r <= m < N and N*r > (m+r)**2):
        raise ValueError("RGS stopping-time assumptions require N*r > (m+r)^2 and 0<r<=m<N")
    if k == 1:
        theta0 = math.asin(math.sqrt(r/N))
        steps = max(0, math.floor(math.pi/(4*theta0) - 0.5))
        success = math.sin((2*steps+1)*theta0)**2
        certificate = "exact_single_oracle_probability"
    else:
        gamma2 = (4*r*(N-m) - (m-r)**2) / N**2
        rho2 = 1 - 2*(m-r)/N
        theta = math.asin(math.sqrt(gamma2/rho2))
        beta = math.asin(math.sqrt(gamma2 / (4*((N-m)/N))))
        steps = max(0, math.floor((math.pi/2-beta)/theta))
        error = (4*r/N + (2+4*k)*(m-r)/math.sqrt(N*r)
                 + 2*k*math.sqrt((m-r)/N))
        success = max(0.0, 1-error)
        certificate = "theorem_uniform_sampling_lower_bound"
    if success < target_success:
        raise ValueError("RGS success guarantee %.8g is below target %.8g; increase N or reduce union_max"
                         % (success, target_success))
    return {
        "method": "rgs", "completion": "none", "steps": steps,
        "sampling_probabilities": [1/k]*k, "query_counts": [steps/k]*k,
        "weighted_cost": steps, "success_lower_bound": success,
        "success_probability": success if k == 1 else None,
        "success_certificate": certificate, "target_success": target_success,
        "query_counts_are_expectations": True,
    }


def generate_instance(config, k, instance):
    """Sample a union size, then a Dirichlet--multinomial region partition.

    Region 0 (all constraints) is fixed to r; the last region is N-m.
    Distribute m-r points over the other 2**k-2 regions, allowing zero sizes.
    The per-instance seed is independent of loop order and resumed runs.
    With k=1 there are no intermediate regions: all 50 cases share [r,N-r].
    """
    seed_components = [config.seed, k, instance]
    rng = np.random.default_rng(np.random.SeedSequence(seed_components))
    if k == 1:
        blocks = [config.r, config.N-config.r]
        m = config.r
    else:
        m = int(rng.integers(config.union_min, config.union_max+1))
        weights = rng.dirichlet(np.full(2**k-2, config.dirichlet_alpha))
        middle = rng.multinomial(m-config.r, weights)
        blocks = [config.r] + [int(x) for x in middle] + [config.N-m]
    return {"k": k, "instance": instance, "seed_components": seed_components,
            "N": config.N, "r": config.r, "m": m, "block_sizes": blocks}


def measure_strategy(function, repeats, min_seconds):
    """Median seconds/call from repeated calibrated batches after a pilot."""
    started = time.perf_counter_ns()
    result = function()
    pilot = (time.perf_counter_ns()-started)/1e9
    batch = min(10000, max(1, math.ceil(min_seconds/max(pilot, 1e-9))))
    observations = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        for _ in range(batch):
            result = function()
        observations.append((time.perf_counter_ns()-started)/1e9/batch)
    return result, {
        "seconds": statistics.median(observations),
        "seconds_per_call_samples": observations,
        "batch_size": batch, "repeats": repeats, "pilot_seconds": pilot,
    }


def combine_ss_timing(order_timing, extra_timing):
    """Add measured stage medians; retain both stages' original observations.

    The two stages can use different batch sizes. Consequently no artificial
    list of end-to-end samples or common batch size is manufactured here.
    """
    return {
        "seconds": order_timing["seconds"] + extra_timing["seconds"],
        "mode": "sum_of_component_medians",
        "order_seconds": order_timing["seconds"],
        "ss_extra_seconds": extra_timing["seconds"],
        "batch_size": None, "repeats": extra_timing["repeats"],
        "components": {"order": order_timing, "ss_extra": extra_timing},
    }


def _source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, content):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    pending.replace(path)


def _write_csv(path, rows):
    if not rows:
        return
    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pending.replace(path)


def load_records(path):
    """Last entry wins if a failed instance was retried with --resume."""
    records = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError("Invalid checkpoint line %d in %s" % (line_number, path)) from exc
                    records[(record["k"], record["instance"])] = record
    return records


def _append_record(path, record):
    import os
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _mean_interval(values, rng, resamples):
    """Percentile bootstrap 95% interval for the mean across problem instances."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    if std == 0 or len(values) == 1:
        return mean, std, mean, mean
    # Chunk to keep memory bounded if a later experiment has more samples.
    estimates = []
    for start in range(0, resamples, 200):
        draws = rng.choice(values, size=(min(200, resamples-start), len(values)), replace=True)
        estimates.extend(draws.mean(axis=1))
    low, high = np.quantile(estimates, [.025, .975])
    return mean, std, float(low), float(high)


def export_tables(output, records, config):
    """Export both individual observations and aggregate plot data."""
    rows = []
    for _, record in sorted(records.items()):
        if record["status"] != "ok":
            continue
        baseline = math.pi/4*math.sqrt(record["N"]/record["r"])
        for method in METHODS:
            result = record["methods"][method]["result"]
            timing = record["methods"][method]["timing"]
            rows.append({
                "k": record["k"], "instance": record["instance"], "method": method,
                "N": record["N"], "r": record["r"], "m": record["m"],
                "oracle_cost": result["weighted_cost"],
                "normalized_cost": result["weighted_cost"]/baseline,
                "success_lower_bound": result.get("success_lower_bound", result["success_probability"]),
                "success_probability": result["success_probability"],
                "planning_seconds": timing["seconds"], "timing_batch_size": timing["batch_size"],
                "timing_repeats": timing["repeats"],
                "order_seconds": (timing["seconds"] if method == "naive"
                                  else timing.get("order_seconds", 0.0)),
                "ss_extra_seconds": timing.get("ss_extra_seconds", 0.0),
                "timing_mode": timing.get("mode", "measured_stage_median"),
            })
    _write_csv(output / "per_instance.csv", rows)
    summary = []
    for k in range(config.k_min, config.k_max+1):
        for mi, method in enumerate(METHODS):
            selected = [row for row in rows if row["k"] == k and row["method"] == method]
            if not selected:
                continue
            row = {"k": k, "method": method, "n": len(selected)}
            for metric_index, metric in enumerate(("oracle_cost", "normalized_cost", "planning_seconds",
                                                  "order_seconds", "ss_extra_seconds")):
                # Shared resampling indices preserve the paired relationship
                # between naive's common stage and SS's additive total.
                timing_seed = 0 if metric.endswith("seconds") else mi
                rng = np.random.default_rng(np.random.SeedSequence([config.seed, k, timing_seed, metric_index, 999]))
                estimate = _mean_interval([item[metric] for item in selected], rng, config.bootstrap_resamples)
                for suffix, value in zip(("mean", "std", "ci_low", "ci_high"), estimate):
                    row[metric+"_"+suffix] = value
            row["success_lower_bound_min"] = min(item["success_lower_bound"] for item in selected)
            row["timing_protocol"] = TIMING_PROTOCOL
            summary.append(row)
    _write_csv(output / "summary.csv", summary)
    return summary


def plot_saved(input_dir, figure_dir=None, normalized=False, dpi=240, labels=None, max_xticks=None):
    """Plot existing summary.csv; do not invoke data generation or optimizers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    input_dir = Path(input_dir)
    labels = LABELS if labels is None else labels
    figure_dir = Path(figure_dir) if figure_dir else input_dir / "figures"
    with (input_dir / "summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("summary.csv has no data")
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {"rgs": "#0072B2", "naive": "#D55E00", "ss": "#009E73"}
    markers = {"rgs": "o", "naive": "s", "ss": "^"}
    cost_metric = "normalized_cost" if normalized else "oracle_cost"
    cost_label = (r"Oracle cost / $[(\pi/4)\sqrt{N/r}]$" if normalized else "Total oracle cost")
    outputs = []
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        for metric, ylabel, stem, scale in (
                (cost_metric, cost_label, "oracle_cost_vs_k", 1),
                ("planning_seconds", "Classical strategy-selection time (ms)", "classical_time_vs_k", 1000)):
            fig, ax = plt.subplots(figsize=(6.5, 4.3), layout="constrained")
            for method in METHODS:
                selected = sorted((row for row in rows if row["method"] == method), key=lambda row: int(row["k"]))
                if not selected:
                    continue
                x = np.array([int(row["k"]) for row in selected])
                y = np.array([float(row[metric+"_mean"])*scale for row in selected])
                low = np.array([float(row[metric+"_ci_low"])*scale for row in selected])
                high = np.array([float(row[metric+"_ci_high"])*scale for row in selected])
                if np.any(low <= 0):
                    raise ValueError("Logarithmic plots require positive confidence bounds")
                ax.plot(x, y, color=colors[method], marker=markers[method], linewidth=1.8,
                        markersize=5, label=labels[method])
                # Shading avoids obscuring overlapping curves with large caps.
                ax.fill_between(x, low, high, color=colors[method], alpha=.16, linewidth=0)
            ax.set(xlabel=r"Number of oracles $k$", ylabel=ylabel, yscale="log")
            ticks = sorted({int(row["k"]) for row in rows})
            if max_xticks is not None and len(ticks) > max_xticks:
                indices = np.linspace(0, len(ticks)-1, max_xticks, dtype=int)
                ticks = [ticks[i] for i in indices]
            ax.set_xticks(ticks)
            ax.grid(True, which="major", alpha=.25, linewidth=.6)
            ax.legend(frameon=False, fontsize=9, loc="best")
            for extension in ("pdf", "png"):
                path = figure_dir / (stem+"."+extension)
                fig.savefig(path, dpi=dpi)
                outputs.append(path)
            plt.close(fig)
    return outputs


def run_experiment(config, output, resume=False, make_plots=True, normalized=False):
    """Run/checkpoint paired instances; fail explicitly without dropping cases."""
    config.validate()
    # Imports and a small warm-up precede all measured strategy calls.
    import scipy.optimize  # noqa: F401
    import nested_amplification as nested
    warmup = nested.SearchProblem([64, 2, 2, 1], [1, 1])
    warmup_naive = nested.optimize_naive(warmup, config.target_success)
    nested.optimize_ss_for_order(warmup, warmup_naive.order, config.target_success, maxiter=config.ss_maxiter)
    rgs_strategy(config.N, config.r, config.r, 1, config.target_success)
    output = Path(output)
    metadata_path = output / "metadata.json"
    records_path = output / "instances.jsonl"
    hashes = {"experiment_equal_cost.py": _source_hash(__file__),
              "nested_amplification.py": _source_hash(nested.__file__)}
    if metadata_path.exists():
        if not resume:
            raise FileExistsError("Output exists; use --resume or select another --output directory: " + str(output))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("timing_protocol") != TIMING_PROTOCOL:
            raise ValueError("Existing results use a different timing protocol; choose a new output directory")
        if metadata["config"] != asdict(config) or metadata["source_sha256"] != hashes:
            raise ValueError("Resume requires the original configuration and source code; choose a new output directory")
    else:
        if records_path.exists():
            raise ValueError("Checkpoint exists without metadata: " + str(records_path))
        output.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": 2, "timing_protocol": TIMING_PROTOCOL,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config), "source_sha256": hashes,
            "environment": {"python": sys.version, "platform": platform.platform(),
                            "processor": platform.processor(),
                            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib")}},
            "protocol": {
                "oracle_costs": "All c_i=1; same Boolean-query convention as nested_amplification.py.",
                "sampling": "Uniform integer union size; Dirichlet(alpha) / multinomial allocation over proper nonempty regions; r fixed.",
                "single_oracle": "k=1 has no random intermediate sets; repeated timing observations of the same [r,N-r] instance.",
                "rgs": "Uniform oracle sampling; paper stopping time; certified success bound; no trajectory simulation.",
                "ss": "One oracle per level; naive-optimal order; final amplify; numerical optimum not generally certified.",
                "timing": "Component-based: T_naive=T_order, T_SS=T_order+T_extra. Each component is a median per-call time across calibrated batches. SS receives the measured naive-optimal order and does not recompute it. The total is an additive estimate, not an independent end-to-end timing.",
                "excluded": "Imports, generation, known-size table conversion/validation, acquiring cardinalities, simulation, I/O, statistics and plotting.",
                "confidence_intervals": "Paired problem instances; marginal 95% percentile-bootstrap confidence intervals for each mean.",
            },
        }
        _write_json(metadata_path, metadata)
    records = load_records(records_path)
    for k in range(config.k_min, config.k_max+1):
        for instance in range(config.samples):
            key = (k, instance)
            if key in records and records[key]["status"] == "ok":
                continue
            record = generate_instance(config, k, instance)
            problem = nested.SearchProblem.from_blocks(record["block_sizes"], costs=[1]*k)
            record.update(intersection_sizes=list(problem.intersection_sizes), methods={}, status="running",
                          timing_protocol=TIMING_PROTOCOL)
            # Vary RGS's placement while respecting the order -> SS dependency.
            measurement_order = (("rgs", "naive", "ss_extra"),
                                 ("naive", "rgs", "ss_extra"),
                                 ("naive", "ss_extra", "rgs"))[instance % 3]
            record["measurement_order"] = list(measurement_order)
            naive_reference = None
            try:
                for stage in measurement_order:
                    if stage == "rgs":
                        function = lambda: rgs_strategy(config.N, config.r, record["m"], k, config.target_success)
                    elif stage == "naive":
                        function = lambda: nested.optimize_naive(problem, config.target_success)
                    else:
                        function = lambda: nested.optimize_ss_for_order(
                            problem, naive_reference.order, config.target_success, maxiter=config.ss_maxiter)
                    result, timing = measure_strategy(function, config.timing_repeats, config.timing_min_seconds)
                    method = "ss" if stage == "ss_extra" else stage
                    if stage == "naive":
                        naive_reference = result
                    result = result if isinstance(result, dict) else result.as_dict()
                    if stage == "ss_extra":
                        timing = combine_ss_timing(record["methods"]["naive"]["timing"], timing)
                        result["naive_target_success"] = config.target_success
                    guarantee = result.get("success_lower_bound", result["success_probability"])
                    if guarantee < config.target_success:
                        raise ValueError(method + " does not meet target_success")
                    record["methods"][method] = {"result": result, "timing": timing}
                record["status"] = "ok"
            except Exception as exc:
                record.update(status="error", error=type(exc).__name__+": "+str(exc))
                _append_record(records_path, record)
                records[key] = record
                export_tables(output, records, config)
                raise RuntimeError("Instance k=%d, sample=%d failed and was saved; no instance was silently discarded. %s"
                                   % (k, instance, exc)) from exc
            _append_record(records_path, record)
            records[key] = record
            print("k=%2d sample=%2d/%d | costs RGS=%.0f naive=%.0f SS=%.0f | time(ms) RGS=%.4f naive/order=%.4f SS_total=%.4f (SS_extra=%.4f)"
                  % (k, instance+1, config.samples,
                     *(record["methods"][m]["result"]["weighted_cost"] for m in METHODS),
                     *(1000*record["methods"][m]["timing"]["seconds"] for m in METHODS),
                     1000*record["methods"]["ss"]["timing"]["ss_extra_seconds"]), flush=True)
        export_tables(output, records, config)
    export_tables(output, records, config)
    if make_plots:
        for path in plot_saved(output, normalized=normalized):
            print("Saved " + str(path), flush=True)
    print("Data saved in " + str(output.resolve()), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Generate, benchmark, save, and plot")
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--no-plot", action="store_true")
    run.add_argument("--normalized", action="store_true", help="Plot query cost divided by (pi/4)*sqrt(N/r)")
    defaults = Config()
    for field, kind in (("N", int), ("r", int), ("k_min", int), ("k_max", int), ("samples", int),
                        ("target_success", float), ("union_min", int), ("union_max", int),
                        ("dirichlet_alpha", float), ("seed", int), ("timing_repeats", int),
                        ("timing_min_seconds", float), ("ss_maxiter", int), ("bootstrap_resamples", int)):
        run.add_argument("--"+field.replace("_", "-"), dest=field, type=kind, default=getattr(defaults, field))
    plot = commands.add_parser("plot", help="Redraw from summary.csv ONLY")
    plot.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    plot.add_argument("--figure-dir", type=Path)
    plot.add_argument("--normalized", action="store_true")
    plot.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()
    if args.command == "plot":
        for path in plot_saved(args.input, args.figure_dir, args.normalized, args.dpi):
            print("Saved " + str(path))
    else:
        config = Config(**{name: getattr(args, name) for name in asdict(defaults)})
        run_experiment(config, args.output, args.resume, not args.no_plot, args.normalized)


if __name__ == "__main__":
    main()
