# Random Grover Search revision experiments

This repository contains code and selected data for the numerical experiments in the revised manuscript **Random Grover Search**. Commands below assume the repository root as the working directory. The code is distributed under the MIT license in `LICENSE`.

## Environment and checks

Install the listed Python packages in an isolated environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The 48 tests below were run with Python 3.9.6, NumPy 2.0.2, Matplotlib 3.9.4, and SciPy 1.13.1. They cover small calculations, checkpointing, statistics and plotting using temporary files.

```bash
python -m unittest -v test_biased_sampling test_equal_cost test_equal_cost_orders test_unequal_cost test_unequal_cost_fixed test_cost_heterogeneity
```

## Code map

| File | Purpose |
| --- | --- |
| `grover_3_random_oracle.py` | Figure 2: three-oracle biased sampling simulation |
| `experiment_equal_cost_orders.py` | Figure 3: five-method equal-cost comparison |
| `experiment_unequal_cost_fixed.py` | Figure 4: fixed two-oracle unequal-cost comparison |
| `experiment_cost_heterogeneity.py` | Figure 5: multi-oracle cost-heterogeneity experiment |
| `experiment_equal_cost.py` | Shared instance generation, timing and statistics routines |
| `experiment_unequal_cost.py` | Shared biased RGS calculation for the fixed-cost experiment |
| `nested_amplification.py` | Nested amplification cost and optimization models |
| `cost_heterogeneity/plot_heatmaps_paper.py` | Redraws Figure 5's manuscript-style panels from saved comparisons |

The corresponding tests are `test_biased_sampling.py`, `test_equal_cost.py`, `test_equal_cost_orders.py`, `test_unequal_cost.py`, `test_unequal_cost_fixed.py`, and `test_cost_heterogeneity.py`. The entry scripts import their shared dependencies automatically.

## Figure 2. Biased sampling

Generate a fresh set of simulations and plots with the following command.

```bash
python grover_3_random_oracle.py --output results/reproduce_fig2
```

The default experiment uses `N=2**40`, `r=1`, the three-oracle instance with set sizes 100, 200 and 50, and 50 independent sequences for each of `delta=0.005, 0.15, 0.4`. The script stores sampled curves, trial data, parameters and seeds. The plotted output is the curve panel; the manuscript's probability table and composite layout are assembled separately.

`results/recheck_fig2/` contains a later verification run with the current code. Redraw its saved curve with the following command.

```bash
python grover_3_random_oracle.py --plot-only --output results/recheck_fig2
```

## Figure 3. Equal oracle costs

Run a new experiment and draw normalized oracle costs with the following command.

```bash
python experiment_equal_cost_orders.py run --output results/reproduce_fig3 --normalized
```

The defaults use `N=2**48`, `r=1`, `k=1..15`, 50 samples per `k`, unit oracle costs, and target success 0.99. For `k>=2`, the union size is uniform on 256..1024; proper membership-region sizes are drawn using a Dirichlet-multinomial construction. The five entries are RGS, naive and SS using a shared random order, and naive and SS using the order minimizing the naive finite query cost by subset dynamic programming. SS uses this same naive-optimal order and chooses integer counts including final coherent amplification. The random-order seed is independent of set generation and timing calibration.

Query costs are normalized by `(pi/4)*sqrt(N/r)` for the figure; bands represent 95% bootstrap intervals. Classical planning uses medians of calibrated batches. Random-order methods include their shared random-order draw plus their own planning. Optimized naive includes subset DP, and optimized SS adds its own optimization to the same DP time. Imports, warmup, set generation, cardinality acquisition, quantum simulation, I/O, statistics and plotting are excluded. Timings depend on the machine and environment.

`results/equal_cost_orders/summary.csv` and `metadata.json` are historical saved results. Full inputs and per-instance records are supplied as gzip files; see the data section below. Redraw the saved summary with the following command.

```bash
python experiment_equal_cost_orders.py plot --input results/equal_cost_orders --normalized --figure-dir figs/figure3_original
```

## Figure 4. Fixed two-oracle unequal costs

```bash
python experiment_unequal_cost_fixed.py run --output results/reproduce_fig4
```

Defaults are `N=2**64`, `r=1`, set-size pairs `(1000,1000)`, `(10000,10000)`, and `(100000,100000)`, costs `(1,lambda)`, and a logarithmic scan from 1 to `10**5`. The experiment compares RGS with naive nesting. RGS uses the theorem's sufficient constraints and target success 0.99; naive nesting amplifies to success 1. The finite-model crossings are computed from the affine costs for both naive orders and checked against the optimal-order envelope. Leading predictions are approximations. The finite crossings retain integer-rounding and phase-query effects. Each curve represents a fixed instance.

`results/unequal_cost_fixed/` preserves the historical curves, critical points, complete results and metadata. Redraw those saved CSVs with the following command.

```bash
python experiment_unequal_cost_fixed.py plot --input results/unequal_cost_fixed --figure-dir figs/figure4_original
```

The script produces separate figures and data tables; the manuscript's combined Figure 4 layout is assembled separately.

## Figure 5. Cost heterogeneity

Redraw the manuscript-style panels using the saved original comparisons with the following command.

```bash
python cost_heterogeneity/plot_heatmaps_paper.py
```

The script reads `cost_heterogeneity/comparisons.csv` and saves two PDF/PNG panels under `figs/`, creating the directory if needed. It overwrites files of the same names. The manuscript places the two separately generated panels (a) and (b) together.

Generate a fresh computation and the experiment script's heatmaps with the following command.

```bash
python experiment_cost_heterogeneity.py run --output results/reproduce_fig5
```

The default grid has `k=2..15`, 13 logarithmically spaced values of `kappa=1..1000`, three cost profiles and 50 paired instances per `k`, giving 27,300 three-method cases. Costs have mean one and `kappa=mean(cost)/min(cost)`. Before dividing by `kappa`, the profiles are one cheap oracle `(1,a,...,a)` with `a=(k*kappa-1)/(k-1)`, one expensive oracle `(1,...,1,b)` with `b=1+k*(kappa-1)`, and a linear gradient from 1 to `2*kappa-1`. A seeded permutation assigns costs to oracle indices and is reused across the grid for each instance. The fixed-union ensemble can contain redundant filters; conclusions concern that ensemble.

All methods target success 0.99. RGS minimizes cost under the theorem's sufficient constraints, with remaining sampling probability assigned to a cheapest oracle. Naive nesting finds a cost-optimal order by subset DP; SS follows that order and numerically optimizes its iteration counts, including final amplification. SS is generally a best-found numerical solution. The plotted values are ratios of **sample mean costs**. Red favors RGS, and blue favors the nested method. The black contour is a visual guide interpolating equality in `k` and `log10(kappa)`.

`cost_heterogeneity/` preserves the original summary, comparisons and metadata used by the paper-style plot. `results/cost_heterogeneity/` contains those summaries and compressed full original records. `results/recheck_fig5/` contains summaries and metadata from a later run with the current code: the later run provides an independent verification. Inputs match, but some SS results differ slightly; the original metadata records source hashes that do not all match the present source files. The historical source identity and the current code snapshot are documented separately.

## Data, compression and restarting

Historical Figure 3 and Figure 5 full records are stored as `.gz` files. Compression preserves the original file bytes on decompression. For tools that require the uncompressed files in place, extract them with these commands.

```bash
gzip -dk results/equal_cost_orders/instances.jsonl.gz
gzip -dk results/equal_cost_orders/per_instance.csv.gz
gzip -dk results/cost_heterogeneity/instances.json.gz
gzip -dk results/cost_heterogeneity/per_instance.csv.gz
gzip -dk results/cost_heterogeneity/results.jsonl.gz
```

The `-k` option keeps each compressed archive. Run these commands when the corresponding uncompressed files are absent. Saved summary-based plotting reads the CSV files supplied alongside the archives. Full records may be large. Figure 2's saved verification data are identified above; Figure 3 and 4 historical results and Figure 5 original data should remain distinct from any new runs.

Use new, empty output directories for fresh runs. The Figure 3 and Figure 5 experiments support checkpoint resume by repeating the original command with `--resume`, with the same configuration, input, source and required environment. Continue the fresh runs above with these commands.

```bash
python experiment_equal_cost_orders.py run --output results/reproduce_fig3 --normalized --resume
python experiment_cost_heterogeneity.py run --output results/reproduce_fig5 --resume
```

Figure 3 checks the saved environment, input, protocol and source hashes. Figure 5 checks configuration, source hashes, NumPy/SciPy versions and the saved cohort hash. For Figure 2 and Figure 4, use a new output directory for a new run. The original source hashes are preserved in the historical metadata so readers can distinguish the archived experiment records from the current source snapshot.

## Archived previous release

The earlier release is archived at [Zenodo DOI 10.5281/zenodo.20603404](https://doi.org/10.5281/zenodo.20603404). That DOI identifies the previous release. A separate release and archive are needed to cite an exact revision snapshot.
