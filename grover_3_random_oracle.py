#second subsection of numerical experiments
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ============================================================
# 1. Build 8D reduced model for 3 sets
# Basis order:
#   I1 = A0 ∩ A1 ∩ A2
#   I2 = A0 ∩ A1 ∩ A2^c
#   I3 = A0 ∩ A1^c ∩ A2
#   I4 = A0 ∩ A1^c ∩ A2^c
#   I5 = A0^c ∩ A1 ∩ A2
#   I6 = A0^c ∩ A1 ∩ A2^c
#   I7 = A0^c ∩ A1^c ∩ A2
#   I8 = (A0 ∪ A1 ∪ A2)^c
# ============================================================
def build_8d_system(block_sizes):
    """
    block_sizes = [n1, n2, ..., n8]
    """
    if len(block_sizes) != 8:
        raise ValueError("block_sizes must have length 8.")

    block_sizes = np.array(block_sizes, dtype=int)
    if np.any(block_sizes < 0):
        raise ValueError("All block sizes must be nonnegative.")

    N = int(np.sum(block_sizes))
    if N <= 0:
        raise ValueError("Total size N must be positive.")

    # initial uniform state reduced to the 8D block basis
    psi0 = np.sqrt(block_sizes / N).astype(float)

    # membership table of each block in A0,A1,A2
    membership = np.array([
        [1, 1, 1],  # I1
        [1, 1, 0],  # I2
        [1, 0, 1],  # I3
        [1, 0, 0],  # I4
        [0, 1, 1],  # I5
        [0, 1, 0],  # I6
        [0, 0, 1],  # I7
        [0, 0, 0],  # I8
    ], dtype=int)

    # diffusion operator
    D = 2 * np.outer(psi0, psi0) - np.eye(8)

    # oracle-based Grover operators G0,G1,G2
    Gs = []
    for j in range(3):
        signs = np.where(membership[:, j] == 1, -1.0, 1.0)
        Oj = np.diag(signs)
        Gj = D @ Oj
        Gs.append(Gj)

    return N, psi0, Gs


# ============================================================
# 2. Basic quantities
# r = triple intersection size = n1
# m = union size = N - n8
# ============================================================
def get_r_and_m(block_sizes):
    block_sizes = np.array(block_sizes, dtype=int)
    r = int(block_sizes[0])
    N = int(np.sum(block_sizes))
    m = N - int(block_sizes[7])
    return r, m, N


# ============================================================
# 3. Probabilities from your formula
#
# p1 = p2 = [4(m-r)/sqrt(rN) + 2 sqrt((m-r)/N)]
#           / [delta - 4r/N - 2(m-r)/sqrt(rN)]
#
# p0 = 1 - p1 - p2
# ============================================================
def compute_probabilities_three_sets(N, m, r, delta):
    if r <= 0:
        raise ValueError("Need r > 0.")

    denom = delta - 4 * r / N - 2 * (m - r) / np.sqrt(r * N)
    if denom <= 0:
        raise ValueError(
            "Denominator <= 0. Need "
            "delta > 4r/N + 2(m-r)/sqrt(rN)."
        )

    numer = 4 * (m - r) / np.sqrt(r * N) + 2 * np.sqrt((m - r) / N)

    p1 = numer / denom
    p2 = p1
    p0 = 1.0 - p1 - p2

    if p1 < 0 or p2 < 0 or p0 < 0:
        raise ValueError(
            f"Invalid probabilities: p0={p0:.12f}, p1={p1:.12f}, p2={p2:.12f}."
        )

    return p0, p1, p2


# ============================================================
# 4. Standard Grover steps (same as binary case)
# ============================================================
def standard_grover_steps(N, r):
    return int(np.round(np.pi / 4 * np.sqrt(N / r)) - 0.5)


# ============================================================
# 5. One random trial
# At each step:
#   choose G0 with prob p0
#   choose G1 with prob p1
#   choose G2 with prob p2
# Record:
#   - target probability curve = |a0(t)|^2
#   - operator usage counts
# ============================================================
def run_one_random_trial_three(
    psi0,
    Gs,
    probs,
    T_steps,
    rng
):
    state = psi0.copy()

    prob_curve = np.zeros(T_steps + 1, dtype=float)
    prob_curve[0] = abs(state[0]) ** 2

    op_used = np.full(T_steps + 1, -1, dtype=int)
    counts = np.zeros(3, dtype=int)

    p0, p1, p2 = probs
    cum_probs = np.array([p0, p0 + p1, p0 + p1 + p2], dtype=float)

    for t in range(1, T_steps + 1):
        u = rng.random()

        if u < cum_probs[0]:
            idx = 0
        elif u < cum_probs[1]:
            idx = 1
        else:
            idx = 2

        state = Gs[idx] @ state
        counts[idx] += 1
        op_used[t] = idx
        prob_curve[t] = abs(state[0]) ** 2

    ratios = counts / T_steps if T_steps > 0 else np.zeros(3)

    return {
        "prob_curve": prob_curve,
        "final_state": state,
        "final_prob": prob_curve[-1],
        "op_used": op_used,
        "counts": counts,
        "ratios": ratios,
    }


# ============================================================
# 6. Many random trials
# ============================================================
def run_many_random_trials_three(
    block_sizes,
    delta,
    n_trials=50,
    seed_base=1
):
    N, psi0, Gs = build_8d_system(block_sizes)
    r, m, _ = get_r_and_m(block_sizes)

    # three set sizes
    n1,n2,n3,n4,n5,n6,n7,n8 = block_sizes

    size_A0 = n1 + n2 + n3 + n4
    size_A1 = n1 + n2 + n5 + n6
    size_A2 = n1 + n3 + n5 + n7

    p0, p1, p2 = compute_probabilities_three_sets(N, m, r, delta)
    T_steps = standard_grover_steps(N, r)

    print("===== Experiment parameters =====")
    print(f"block_sizes   = {list(block_sizes)}")
    print(f"N             = {N}")

    print(f"|A0|          = {size_A0}")
    print(f"|A1|          = {size_A1}")
    print(f"|A2|          = {size_A2}")

    print(f"r             = {r}")
    print(f"m             = {m}")
    print(f"delta         = {delta}")
    print(f"p0            = {p0:.12f}")
    print(f"p1            = {p1:.12f}")
    print(f"p2            = {p2:.12f}")
    print(f"T_steps       = {T_steps}")
    print(f"n_trials      = {n_trials}")
    print(f"seed_base     = {seed_base}")
    print()

    all_prob_curves = np.zeros((n_trials, T_steps + 1), dtype=float)
    final_probs = np.zeros(n_trials, dtype=float)
    all_counts = np.zeros((n_trials, 3), dtype=int)
    all_ratios = np.zeros((n_trials, 3), dtype=float)
    all_ops = np.zeros((n_trials, T_steps + 1), dtype=int)
    used_seeds = []

    for k in range(n_trials):
        seed_k = seed_base + k
        rng = np.random.default_rng(seed_k)
        used_seeds.append(seed_k)

        result = run_one_random_trial_three(
            psi0=psi0,
            Gs=Gs,
            probs=(p0, p1, p2),
            T_steps=T_steps,
            rng=rng
        )

        all_prob_curves[k, :] = result["prob_curve"]
        final_probs[k] = result["final_prob"]
        all_counts[k, :] = result["counts"]
        all_ratios[k, :] = result["ratios"]
        all_ops[k, :] = result["op_used"]

    summary = {
        "mean_final_prob": np.mean(final_probs),
        "std_final_prob": np.std(final_probs, ddof=1) if n_trials > 1 else 0.0,
        "mean_prob_curve": np.mean(all_prob_curves, axis=0),
        "mean_counts": np.mean(all_counts, axis=0),
        "std_counts": np.std(all_counts, axis=0, ddof=1) if n_trials > 1 else np.zeros(3),
        "mean_ratios": np.mean(all_ratios, axis=0),
        "std_ratios": np.std(all_ratios, axis=0, ddof=1) if n_trials > 1 else np.zeros(3),
    }

    return {
        "block_sizes": np.array(block_sizes, dtype=int),
        "N": N,
        "r": r,
        "m": m,
        "delta": delta,
        "p0": p0,
        "p1": p1,
        "p2": p2,
        "T_steps": T_steps,
        "n_trials": n_trials,
        "used_seeds": used_seeds,
        "all_prob_curves": all_prob_curves,
        "final_probs": final_probs,
        "all_counts": all_counts,
        "all_ratios": all_ratios,
        "all_ops": all_ops,
        "summary": summary,
    }


# ============================================================
# 7. Pretty print summary
# ============================================================
def print_trial_summary_three(results, show_each_trial=True):
    print("===== Summary over all random trials =====")
    print(f"mean final prob   = {results['summary']['mean_final_prob']:.6f}")
    print(f"std  final prob   = {results['summary']['std_final_prob']:.6f}")
    print()

    mean_counts = results["summary"]["mean_counts"]
    std_counts = results["summary"]["std_counts"]
    mean_ratios = results["summary"]["mean_ratios"]
    std_ratios = results["summary"]["std_ratios"]

    print("Operator usage:")
    for j in range(3):
        print(
            f"G{j}: mean count = {mean_counts[j]:.6f}, "
            f"std count = {std_counts[j]:.6f}, "
            f"mean ratio = {mean_ratios[j]:.6f}, "
            f"std ratio = {std_ratios[j]:.6f}"
        )
    print()

    if show_each_trial:
        print("===== Per-trial data =====")
        print(
            f"{'trial':>5} {'seed':>8} "
            f"{'G0_count':>10} {'G1_count':>10} {'G2_count':>10} "
            f"{'G0_ratio':>10} {'G1_ratio':>10} {'G2_ratio':>10} "
            f"{'final_prob':>12}"
        )

        for i in range(results["n_trials"]):
            c0, c1, c2 = results["all_counts"][i]
            r0, r1, r2 = results["all_ratios"][i]
            fp = results["final_probs"][i]
            print(
                f"{i+1:>5} {results['used_seeds'][i]:>8} "
                f"{c0:>10} {c1:>10} {c2:>10} "
                f"{r0:>10.6f} {r1:>10.6f} {r2:>10.6f} "
                f"{fp:>12.6f}"
            )



# ============================================================
# 8.many delta together in the same graph
# ============================================================
def run_multi_delta_experiments(block_sizes, delta_list, n_trials=50, seed_base=1):
    results_dict = {}

    for i, delta in enumerate(delta_list):
        results = run_many_random_trials_three(
            block_sizes=block_sizes,
            delta=delta,
            n_trials=n_trials,
            seed_base=seed_base + 1000 * i
        )
        results_dict[delta] = results

    return results_dict



# ============================================================
#9. 3-delta picture and usage ratio picture 
# ============================================================
def plot_multi_delta_mean_curves(results_dict, save_pdf=False, filename_prefix="multi_delta"):

    # =====================================================
    # Match style/size with plot_vary_delta_for_fixed_A0
    # =====================================================
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.grid": True,
        "grid.alpha": 0.30,
        "grid.linewidth": 0.45,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # same size as previous function
    fig, ax = plt.subplots(figsize=(4.6, 4.6))

    # =====================================================
    # x-axis scaling
    # =====================================================
    first_delta = list(results_dict.keys())[0]
    T_steps = results_dict[first_delta]["T_steps"]

    magnitude = 10 ** max(0, int(np.floor(np.log10(T_steps))) - 1)
    T_steps_sc = T_steps / magnitude

    # =====================================================
    # curves
    # =====================================================
    for delta, results in results_dict.items():
        mean_curve = results["summary"]["mean_prob_curve"]
        steps_sc = np.arange(len(mean_curve)) / magnitude

        ax.plot(
            steps_sc,
            mean_curve,
            linewidth=2.0,
            label=rf"$\delta={delta}$"
        )

    # =====================================================
    # vertical T0 line
    # =====================================================
    ax.axvline(
        T_steps_sc,
        linestyle=":",
        color="0.35",
        linewidth=1.1
    )

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks([T_steps_sc])
    ax_top.set_xticklabels([r"$T_0$"])
    ax_top.tick_params(axis="x", direction="in", length=3, width=0.7, pad=2)
    ax_top.xaxis.set_minor_locator(mticker.NullLocator())

    # =====================================================
    # labels
    # =====================================================
    exp = int(np.log10(magnitude)) if magnitude >= 1e3 else None

    if exp is not None:
        ax.set_xlabel(rf"Iteration step $\times 10^{{{exp}}}$")
    else:
        ax.set_xlabel("Iteration step")

    ax.set_ylabel(r"$a_0^2(t)$")

    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))

    # =====================================================
    # title (optional but recommended for consistency)
    # =====================================================
    

    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()

    if save_pdf:
        fig.savefig(f"{filename_prefix}.pdf")
        print(f"Saved: {filename_prefix}.pdf")

    plt.show()
    plt.close(fig)

def plot_usage_ratio_bar(results_dict, save_pdf=False, filename_prefix="usage_bar"):
    # =====================================================
    # Match size/style with other 4.6 × 4.6 figures
    # =====================================================
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.grid": False,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # =====================================================
    # Data
    # =====================================================
    deltas = list(results_dict.keys())

    g0, g1, g2 = [], [], []

    for delta in deltas:
        ratios = results_dict[delta]["summary"]["mean_ratios"]
        g0.append(ratios[0])
        g1.append(ratios[1])
        g2.append(ratios[2])

    g0 = np.array(g0)
    g1 = np.array(g1)
    g2 = np.array(g2)

    x = np.arange(len(deltas))
    width = 0.55

    # =====================================================
    # Figure size aligned with previous plots
    # =====================================================
    fig, ax = plt.subplots(figsize=(4.6, 4.6))

    # =====================================================
    # Bars
    # =====================================================
    bars0 = ax.bar(x, g0, width, label=r"$G_0$", zorder=3)
    bars1 = ax.bar(x, g1, width, bottom=g0, label=r"$G_1$", zorder=3)
    bars2 = ax.bar(x, g2, width, bottom=g0 + g1, label=r"$G_2$", zorder=3)

    # =====================================================
    # Ratio labels inside bars
    # =====================================================
    MIN_HEIGHT = 0.03

    for i in range(len(deltas)):
        for val, bot in [
            (g0[i], 0.0),
            (g1[i], g0[i]),
            (g2[i], g0[i] + g1[i]),
        ]:
            if val >= MIN_HEIGHT:
                ax.text(
                    x[i],
                    bot + val / 2,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white",
                    fontweight="bold"
                )

    # =====================================================
    # Formatting
    # =====================================================
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$\delta={d}$" for d in deltas])

    ax.set_ylabel("Mean usage ratio")
    ax.set_xlabel("Experiment setting")

    ax.set_ylim(0, 1.12)

    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))

    ax.grid(axis="y", linewidth=0.45, alpha=0.30, zorder=0)

    # =====================================================
    # Legend
    # =====================================================
    ax.legend(
        loc="upper right",
        ncol=3,
        frameon=True,
        borderpad=0.4,
        labelspacing=0.2,
        handletextpad=0.3,
        columnspacing=0.6,
        handlelength=1.2,
    )

    plt.tight_layout()

    if save_pdf:
        fig.savefig(f"{filename_prefix}.pdf")
        print(f"Saved: {filename_prefix}.pdf")

    plt.show()
    plt.close(fig)

# ============================================================
# 9. Example usage
# ============================================================
if __name__ == "__main__":
    # Example block sizes:
    # [I1, I2, I3, I4, I5, I6, I7, I8]
    N_total = 2**40
    block_sizes = [1, 12, 8, 79, 10, 177, 31, 2**40-318]

    delta_list = [0.005,0.15,0.4]
    results_dict = run_multi_delta_experiments(
        block_sizes=block_sizes,
        delta_list=delta_list,
        n_trials=50,
        seed_base=1
    )
    
    print("\n===== Expected success probabilities =====")
    for delta, results in results_dict.items():
        mean_prob = results["summary"]["mean_final_prob"]
        std_prob = results["summary"]["std_final_prob"]
        print(
            f"delta = {delta:<5} | "
            f"E[a0^2(T0)] = {mean_prob:.6f} | "
            f"std = {std_prob:.6f} | "
            f"theoretical bound = {1-delta:.6f}"
        )

    plot_usage_ratio_bar(results_dict, True)
    plot_multi_delta_mean_curves(results_dict, True)
