"""One paired equal-cost experiment comparing five search strategies.

    python experiment_equal_cost_orders.py run --output results/equal_cost_orders
    python experiment_equal_cost_orders.py plot --input results/equal_cost_orders

Defaults: k=1..15, 50 instances per k, N=2**48, r=1, all costs one,
target success 0.99. A random order is drawn ONCE per instance and shared
by naive AA and SS. The optimized pair shares the exact naive-optimal
order from subset DP; SS does not independently optimize its order.
Both SS variants optimize counts including final coherent 'amplify'.

Classical time is recorded by components: random-order selection, naive
planning on that order, SS optimization on that order, naive order DP,
and SS optimization on the DP order, plus RGS planning. Shared order
selection/search is charged once to EACH method needing it, not rerun.
The figure has two panels (oracle queries and classical planning time).
The plot command reads summary.csv only. Existing experiments are retained.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import itertools
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import experiment_equal_cost as common


METHODS = ("rgs", "naive_random", "ss_random", "naive_optimized", "ss_optimized")
LABELS = {
    "rgs": "RGS",
    "naive_random": "Naïve nested AA (random order)",
    "ss_random": "SS (random order)",
    "naive_optimized": "Naïve nested AA (optimized order)",
    "ss_optimized": "SS (naïve-optimal order)",
}
PARTS = {
    "rgs": ("rgs",),
    "naive_random": ("random_order", "naive_random"),
    "ss_random": ("random_order", "ss_random"),
    "naive_optimized": ("naive_optimized",),
    "ss_optimized": ("naive_optimized", "ss_optimized"),
}
METRICS = ("oracle_cost", "normalized_cost", "planning_seconds", "random_order_seconds",
           "order_search_seconds", "naive_planning_seconds", "ss_optimization_seconds")
PROTOCOL = "five_methods_shared_random_and_naive_optimal_orders_v1"
DEFAULT_OUTPUT = Path("results/equal_cost_orders")


@dataclass(frozen=True)
class Config(common.Config):
    k_max: int = 15
    order_seed: int = 20261101
    instances_from: str = ""

    def validate(self):
        super().validate()
        if self.order_seed < 0:
            raise ValueError("order_seed must be nonnegative")


def random_order(config, k, instance):
    """Stable order, independent of set generation, timing batches and resume."""
    rng = np.random.default_rng(np.random.SeedSequence([config.order_seed, k, instance, 731]))
    return tuple(int(i) for i in rng.permutation(k))


def _input(config, k, instance, saved, nested):
    raw = common.generate_instance(config, k, instance) if saved is None else saved[k, instance]
    if int(raw["k"]) != k or int(raw["N"]) != config.N or int(raw["r"]) != config.r:
        raise ValueError("input set sizes do not match N,r,k")
    record = {key: raw[key] for key in ("k", "instance", "N", "r", "m")}
    if "block_sizes" in raw:
        problem = nested.SearchProblem.from_blocks(raw["block_sizes"], [1]*k)
        m = problem.N-raw["block_sizes"][-1]
        record["block_sizes"] = list(raw["block_sizes"])
    elif "intersection_sizes" in raw:
        problem = nested.SearchProblem(raw["intersection_sizes"], [1]*k)
        m = sum((1 if bin(mask).count("1") % 2 else -1)*size
                for mask, size in enumerate(problem.intersection_sizes) if mask)
    else:
        raise ValueError("order optimization needs full region or subset sizes; prefix-only fixed-order data are insufficient")
    if problem.k != k or problem.r != config.r or int(record["m"]) != m:
        raise ValueError("input cardinalities are inconsistent")
    record["intersection_sizes"] = list(problem.intersection_sizes)
    record["set_seed_components"] = raw.get("seed_components")
    record["input_source"] = "generated_regions" if saved is None else "saved_set_instance"
    record["random_order"] = list(random_order(config, k, instance))
    record["random_order_seed_components"] = [config.order_seed, k, instance, 731]
    return record, problem


def combine_timing(components, method):
    """Preserve real component samples; never invent end-to-end observations."""
    parts = {name: components[name] for name in PARTS[method]}
    return {"seconds": math.fsum(value["seconds"] for value in parts.values()),
            "mode": "measured_stage_median" if len(parts) == 1 else "sum_of_component_medians",
            "components": parts}


def export_tables(output, records, config):
    rows = []
    for _, record in sorted(records.items()):
        if record["status"] != "ok":
            continue
        baseline = math.pi/4*math.sqrt(record["N"]/record["r"])
        for method in METHODS:
            entry = record["methods"][method]
            result, parts = entry["result"], entry["timing"]["components"]
            seconds = lambda name: parts[name]["seconds"] if name in parts else 0.0
            rows.append({
                "k": record["k"], "instance": record["instance"], "method": method,
                "N": record["N"], "r": record["r"], "m": record["m"],
                "oracle_cost": result["weighted_cost"], "normalized_cost": result["weighted_cost"]/baseline,
                "planning_seconds": entry["timing"]["seconds"],
                "random_order_seconds": seconds("random_order"),
                "order_search_seconds": seconds("naive_optimized"),
                "naive_planning_seconds": seconds("naive_random"),
                "ss_optimization_seconds": seconds("ss_random")+seconds("ss_optimized"),
                "success_lower_bound": result.get("success_lower_bound", result["success_probability"]),
                "success_probability": result["success_probability"],
                "success_flags": result.get("success_flags", 0),
                "optimality": result.get("optimality", "prescribed_uniform_rule"),
                "timing_protocol": PROTOCOL,
            })
    common._write_csv(output/"per_instance.csv", rows)
    summary = []
    for k in range(config.k_min, config.k_max+1):
        for method in METHODS:
            selected = [row for row in rows if row["k"] == k and row["method"] == method]
            if not selected:
                continue
            group = {"k": k, "method": method, "n": len(selected), "complete": int(len(selected) == config.samples)}
            for metric_index, metric in enumerate(METRICS):
                # Identical resampling indices across all five paired methods.
                rng = np.random.default_rng(np.random.SeedSequence([config.seed, k, metric_index, 993]))
                stats = common._mean_interval([r[metric] for r in selected], rng, config.bootstrap_resamples)
                for suffix, value in zip(("mean", "std", "ci_low", "ci_high"), stats):
                    group[metric+"_"+suffix] = value
            group["success_lower_bound_min"] = min(row["success_lower_bound"] for row in selected)
            group["timing_protocol"] = PROTOCOL
            summary.append(group)
    common._write_csv(output/"summary.csv", summary)
    return summary


def plot_saved(input_dir, figure_dir=None, normalized=False, dpi=240):
    """One two-panel figure from saved means and confidence intervals only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    input_dir = Path(input_dir)
    with (input_dir/"summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = {(int(r["k"]), r["method"]): r for r in rows if int(r["complete"])}
    ks = sorted(k for k in {key[0] for key in groups} if all((k, method) in groups for method in METHODS))
    if not ks:
        raise ValueError("no complete five-method groups to plot")
    figure_dir = Path(figure_dir) if figure_dir else input_dir/"figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    styles = {
        "rgs": ("#0072B2", "-", "o", True),
        "naive_random": ("#E69F00", "--", "s", False),
        "ss_random": ("#009E73", "--", "^", False),
        "naive_optimized": ("#D55E00", "-", "D", True),
        "ss_optimized": ("#CC79A7", "-", "v", True),
    }
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.65))
        fig.subplots_adjust(left=.085, right=.99, bottom=.17, top=.80, wspace=.36)
        metrics = (("normalized_cost" if normalized else "oracle_cost", 1,
                    "Normalized oracle queries" if normalized else "Total oracle queries", "(a) Oracle queries"),
                   ("planning_seconds", 1000, "Classical planning time (ms)", "(b) Classical computation"))
        for ax, (metric, scale, ylabel, title) in zip(axes, metrics):
            # Draw open dashed markers over coincident solid curves.
            for method in ("ss_optimized", "ss_random", "naive_optimized", "naive_random", "rgs"):
                selected = [groups[k, method] for k in ks]
                y = np.array([float(r[metric+"_mean"])*scale for r in selected])
                low = np.array([float(r[metric+"_ci_low"])*scale for r in selected])
                high = np.array([float(r[metric+"_ci_high"])*scale for r in selected])
                if np.any(low <= 0):
                    raise ValueError("logarithmic plots require positive cost and timing intervals")
                color, linestyle, marker, filled = styles[method]
                ax.plot(ks, y, color=color, linestyle=linestyle, marker=marker, linewidth=1.3,
                        markersize=3.6, markerfacecolor=color if filled else "none",
                        markeredgewidth=.85, label=LABELS[method])
                ax.fill_between(ks, low, high, color=color, alpha=.12, linewidth=0)
            ax.set(xlabel=r"Number of oracles $k$", ylabel=ylabel, yscale="log",
                   xlim=(ks[0]-.4, ks[-1]+.4))
            ax.set_xticks(list(range(ks[0], ks[-1]+1)))
            ax.set_title(title, fontsize=9.5, pad=7)
            ax.tick_params(axis="both", which="major", labelsize=8)
            ax.grid(True, which="major", alpha=.22, linewidth=.5)
        handles, labels = axes[0].get_legend_handles_labels()
        lookup = dict(zip(labels, handles))
        legend_order = ("rgs", None, "naive_random", "ss_random", "naive_optimized", "ss_optimized")
        fig.legend([lookup[LABELS[m]] if m else Line2D([], [], linestyle="none") for m in legend_order],
                   [LABELS[m] if m else "" for m in legend_order],
                   loc="upper center", bbox_to_anchor=(.52, 1.005), ncol=3, frameon=False,
                   fontsize=8.6, columnspacing=1.4, handlelength=2.1)
        outputs = []
        for suffix in (".pdf", ".png"):
            path = figure_dir/("equal_cost_orders"+suffix)
            fig.savefig(path, dpi=dpi)
            outputs.append(path)
        plt.close(fig)
    return outputs


def _environment():
    return {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib")}}


def _measurement_orders():
    stages = ("random_order", "rgs", "naive_random", "ss_random", "naive_optimized", "ss_optimized")
    return tuple(p for p in itertools.permutations(stages)
                 if p.index("random_order") < min(p.index("naive_random"), p.index("ss_random"))
                 and p.index("naive_optimized") < p.index("ss_optimized"))


def _zero_order_timing():
    return {"seconds": 0.0, "seconds_per_call_samples": [], "batch_size": 0, "repeats": 0,
            "pilot_seconds": 0.0, "mode": "one_oracle_no_order_choice"}


def run_experiment(config, output, resume=False, make_plots=True, normalized=False):
    config.validate()
    import nested_amplification as nested
    output = Path(output)
    metadata_path, records_path = output/"metadata.json", output/"instances.jsonl"
    hashes = {Path(p).name: common._source_hash(p) for p in (__file__, common.__file__, nested.__file__)}
    saved, input_info = None, None
    if config.instances_from:
        source = Path(config.instances_from).resolve()/"instances.jsonl"
        if not source.is_file():
            raise FileNotFoundError(source)
        saved = common.load_records(source)
        for k in range(config.k_min, config.k_max+1):
            for instance in range(config.samples):
                if (k, instance) not in saved or saved[k, instance]["status"] != "ok":
                    raise ValueError("saved inputs lack a successful instance k=%d, sample=%d" % (k, instance))
        input_info = {"path": str(source), "sha256": common._source_hash(source)}
    environment = _environment()
    if metadata_path.exists():
        if not resume:
            raise FileExistsError("Output exists; use --resume with identical options, or choose a new --output")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata.get("timing_protocol") != PROTOCOL or metadata["config"] != asdict(config)
                or metadata["source_sha256"] != hashes or metadata["input_data"] != input_info
                or metadata["environment"] != environment):
            raise ValueError("Resume requires the same protocol, configuration, code, inputs and environment")
    else:
        if records_path.exists():
            raise ValueError("checkpoint exists without metadata")
        output.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": 1, "timing_protocol": PROTOCOL, "status": "partial",
            "created_utc": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
            "source_sha256": hashes, "input_data": input_info, "environment": environment,
            "method_labels": LABELS, "timing_components": PARTS,
            "protocol": {
                "instances": "All five methods share each full region instance; same ensemble and set seed as experiment_equal_cost.py.",
                "random_order": "Uniform permutation drawn once per instance from an independent recorded seed; shared by random-order naive and SS. Timing uses a separate RNG stream so calibration cannot change the selected order.",
                "optimized_order": "Subset DP minimizes naive AA's finite cost; both optimized-order variants share that order. SS does not independently optimize its order or merge tests.",
                "success": "Unit Boolean-query costs; common target success. RGS uses its theorem certificate, not an empirical probability; both SS variants use final coherent amplify.",
                "timing": "Random variants include shared random-draw time plus their own planning/optimization. Optimized naive is the DP stage, optimized SS adds its own SS optimization to the same measured DP stage. Totals are sums of component medians, not independently measured end-to-end samples.",
                "excluded": "Imports, RNG initialization, warmup, set generation, full-table conversion/validation, cardinality acquisition, quantum simulation, I/O, statistics and plotting. Only the optimized variants are charged subset DP.",
                "statistics": "Means with paired-index 95% bootstrap intervals; incomplete five-method groups are omitted from figures; failed cases stop the run without replacement.",
            },
        }
        common._write_json(metadata_path, metadata)
    records = common.load_records(records_path)
    wanted = [(k, i) for k in range(config.k_min, config.k_max+1) for i in range(config.samples)]
    pending = [key for key in wanted if key not in records or records[key]["status"] != "ok"]
    if pending:
        warm = nested.SearchProblem([64, 3, 3, 1])
        warm_naive = nested.optimize_naive(warm, config.target_success)
        nested.evaluate_naive(warm, (0, 1), config.target_success)
        nested.optimize_ss_for_order(warm, (0, 1), config.target_success, maxiter=config.ss_maxiter)
        nested.optimize_ss_for_order(warm, warm_naive.order, config.target_success, maxiter=config.ss_maxiter)
        common.rgs_strategy(config.N, config.r, config.r, 1, config.target_success)
    measurement_orders = _measurement_orders()
    for k, instance in pending:
        record, problem = _input(config, k, instance, saved, nested)
        order = tuple(record["random_order"])
        timing_rng = np.random.default_rng(np.random.SeedSequence([config.order_seed, k, instance, 732]))
        components, results, optimized_order = {}, {}, None
        measurement_order = measurement_orders[(k+instance) % len(measurement_orders)]
        record.update(methods={}, components=components, measurement_order=list(measurement_order), status="running")
        functions = {
            "random_order": lambda: tuple(int(i) for i in timing_rng.permutation(k)),
            "rgs": lambda: common.rgs_strategy(config.N, config.r, record["m"], k, config.target_success),
            "naive_random": lambda: nested.evaluate_naive(problem, order, config.target_success),
            "ss_random": lambda: nested.optimize_ss_for_order(problem, order, config.target_success, maxiter=config.ss_maxiter),
            "naive_optimized": lambda: nested.optimize_naive(problem, config.target_success),
            "ss_optimized": lambda: nested.optimize_ss_for_order(problem, optimized_order, config.target_success, maxiter=config.ss_maxiter),
        }
        try:
            for stage in measurement_order:
                if stage == "random_order" and k == 1:
                    components[stage] = _zero_order_timing()
                    continue
                result, timing = common.measure_strategy(functions[stage], config.timing_repeats, config.timing_min_seconds)
                components[stage] = timing
                if stage == "random_order":
                    continue  # Calibration draws never replace the saved order.
                if stage == "naive_optimized":
                    optimized_order = tuple(result.order)
                result = result if stage == "rgs" else result.as_dict()
                guarantee = result.get("success_lower_bound", result["success_probability"])
                if guarantee < config.target_success-1e-12:
                    raise ValueError(stage+" does not meet target_success")
                if stage != "rgs":
                    expected = order if stage.endswith("random") else optimized_order
                    if tuple(result["order"]) != expected:
                        raise ValueError("method used an unexpected order")
                results[stage] = result
                record["methods"][stage] = {"result": result}
            record["optimized_order"] = list(optimized_order)
            record["methods"] = {method: {"result": results[method], "timing": combine_timing(components, method)}
                                 for method in METHODS}
            record["status"] = "ok"
        except Exception as exc:
            record.update(status="error", error=type(exc).__name__+": "+str(exc))
            common._append_record(records_path, record)
            records[k, instance] = record
            export_tables(output, records, config)
            metadata.update(status="failed", failed_instance=[k, instance])
            common._write_json(metadata_path, metadata)
            raise RuntimeError("Five-method case k=%d, sample=%d failed and was saved; no resampling. %s"
                               % (k, instance, exc)) from exc
        common._append_record(records_path, record)
        records[k, instance] = record
        print("k=%2d sample=%2d/%d | queries RGS=%.0f naive-random=%.0f SS-random=%.0f naive-opt=%.0f SS-naive-opt=%.0f"
              % (k, instance+1, config.samples, *(results[m]["weighted_cost"] for m in METHODS)), flush=True)
        if instance == config.samples-1:
            export_tables(output, records, config)
    export_tables(output, records, config)
    metadata.update(status="complete", completed_instances=len(wanted), completed_utc=datetime.now(timezone.utc).isoformat())
    metadata.pop("failed_instance", None)
    common._write_json(metadata_path, metadata)
    if make_plots:
        for path in plot_saved(output, normalized=normalized):
            print("Saved "+str(path))
    print("Data saved in "+str(output.resolve()), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run all five methods on paired instances")
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--no-plot", action="store_true")
    run.add_argument("--normalized", action="store_true")
    defaults = Config()
    for name, value in asdict(defaults).items():
        run.add_argument("--"+name.replace("_", "-"), type=type(value), default=value)
    plot = commands.add_parser("plot", help="Draw the combined figure from summary.csv only")
    plot.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    plot.add_argument("--figure-dir", type=Path)
    plot.add_argument("--normalized", action="store_true")
    plot.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()
    if args.command == "plot":
        for path in plot_saved(args.input, args.figure_dir, args.normalized, args.dpi):
            print("Saved "+str(path))
    else:
        config = Config(**{name: getattr(args, name) for name in asdict(defaults)})
        run_experiment(config, args.output, args.resume, not args.no_plot, args.normalized)


if __name__ == "__main__":
    main()
