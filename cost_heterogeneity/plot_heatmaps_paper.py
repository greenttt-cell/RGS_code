"""Redraw the two manuscript heatmaps from the saved full experiment.

Run with Python from any directory. No instances or strategies are recomputed.
The input comparisons.csv reports ratios of sample MEAN costs, not means of
per-instance ratios. Both figures use the same centered logarithmic color scale.
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROFILES = ("one_cheap", "one_expensive", "gradual")
TITLES = ("One cheap oracle", "One expensive oracle", "Linear cost gradient")


def edges(values):
    values = np.asarray(values, float)
    middle = (values[1:]+values[:-1])/2
    return np.r_[values[0]-(middle[0]-values[0]), middle,
                 values[-1]+(values[-1]-middle[-1])]


def main():
    with (HERE/"comparisons.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ks = sorted({int(row["k"]) for row in rows})
    kappas = sorted({float(row["kappa"]) for row in rows})
    matrices = {}
    for method in ("naive", "ss"):
        for profile in PROFILES:
            selected = [r for r in rows if r["method_a"] == method
                        and r["method_b"] == "rgs" and r["profile"] == profile]
            assert len(selected) == len(ks)*len(kappas)
            assert all(r["complete"] == "1" and int(r["n"]) == 50 for r in selected)
            grid = np.full((len(kappas), len(ks)), np.nan)
            for row in selected:
                grid[kappas.index(float(row["kappa"])), ks.index(int(row["k"]))] = float(row["ratio_of_means"])
            assert np.all(np.isfinite(grid)) and np.all(grid > 0)
            matrices[method, profile] = np.log10(grid)
    limit = max(float(np.max(np.abs(grid))) for grid in matrices.values())
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    log_kappas = np.log10(kappas)
    (ROOT/"figs").mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "axes.linewidth": .6,
                         "xtick.major.width": .6, "ytick.major.width": .6}):
        for method, heading in (("naive", "(a) Naïve nested AA / RGS"),
                                ("ss", "(b) Schrottenloher-Stevens / RGS")):
            fig, axes = plt.subplots(1, 3, figsize=(6, 2.5), sharex=True, sharey=True)
            fig.subplots_adjust(left=.09, right=.84, bottom=.21, top=.78, wspace=.13)
            for ax, profile, title in zip(axes, PROFILES, TITLES):
                grid = matrices[method, profile]
                mesh = ax.pcolormesh(edges(ks), edges(log_kappas), grid,
                                     cmap="RdBu_r", norm=norm, shading="flat", rasterized=True)
                if np.any(grid < 0) and np.any(grid > 0):
                    ax.contour(ks, log_kappas, grid, levels=[0], colors="#20252b", linewidths=.85)
                ax.set(xlim=(ks[0]-.5, ks[-1]+.5), ylim=(log_kappas[0], log_kappas[-1]),
                       xticks=[2, 6, 10, 15], yticks=[0, 1, 2, 3])
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: r"$10^{%g}$" % x))
                ax.set_title(title, fontsize=8, pad=5)
                ax.tick_params(labelsize=8, length=3, pad=2)
            axes[0].set_ylabel(r"Cost heterogeneity, $\kappa$", fontsize=8.5, labelpad=3)
            fig.text(.455, .045, "Number of oracles, k", ha="center", fontsize=8.5)
            color_axis = fig.add_axes([.865, .21, .022, .57])
            bar = fig.colorbar(mesh, cax=color_axis)
            bar.set_ticks(np.log10([.01, .1, 1, 10, 100]))
            bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "%g" % (10**x)))
            bar.ax.tick_params(labelsize=7.5, length=2.5, pad=2)
            bar.set_label("Mean cost ratio", fontsize=8, labelpad=5)
            fig.suptitle(heading, fontsize=10, y=.985)
            destination = ROOT/"figs"/("relative_cost_"+method+"_heatmap")
            fig.savefig(destination.with_suffix(".pdf"))
            fig.savefig(destination.with_suffix(".png"), dpi=300)
            plt.close(fig)
            print(destination.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
