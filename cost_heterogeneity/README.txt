Figure 5 manuscript-style heatmap data and plotting

The summary.csv, comparisons.csv and metadata.json here were copied from the
original results/cost_heterogeneity experiment on 2026-10-29. The metadata
records the original configuration, source hashes, cohort hash and method
conventions. The corresponding full original instances and per-case records
are in results/cost_heterogeneity/ as gzip files; see the repository README for
decompression. Some historical source hashes differ from the current source. The metadata
identifies the original experiment separately from the current code snapshot.

To redraw the two manuscript-style panels from this directory's saved
comparisons, run from the repository root.

    python cost_heterogeneity/plot_heatmaps_paper.py

The script creates the repository-root figs/ directory if needed, then writes
relative_cost_naive_heatmap.pdf/.png and relative_cost_ss_heatmap.pdf/.png.
Existing files of those names are replaced. The manuscript places these two
separate panels together. The script reads the saved comparisons directly.

All costs have mean one. kappa=mean(cost)/min(cost).
Each plotted value is mean(C_nested)/mean(C_RGS), based on 50 paired instances.
The black contour is a visual guide interpolating equality in k and
log10(kappa).

results/recheck_fig5/ holds summaries from a later current-code verification
run, separate from the historical data used here. Inputs agree, while some SS
results show small numerical differences. See the repository README for full
experiment commands and additional provenance details.
