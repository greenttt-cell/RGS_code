# Compare deterministic sequence (G0 G0 G1)^n
# with random sampling: 2/3 G0 + 1/3 G1
# Output four small PDF figures for TeX 2x2 subfigure layout

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ============================================================
# 1. Build 4D reduced model
# Basis:
#   I1 = A0 ∩ A1
#   I2 = A0 ∩ A1^c
#   I3 = A0^c ∩ A1
#   I4 = (A0 ∪ A1)^c
# ============================================================
def build_4d_system(N: int, size_A0: int, size_A1: int, r: int):
    if not (0 < r <= size_A0 and 0 < r <= size_A1):
        raise ValueError("Need 0 < r <= |A0| and 0 < r <= |A1|.")

    n1 = r
    n2 = size_A0 - r
    n3 = size_A1 - r
    n4 = N - size_A0 - size_A1 + r

    if min(n1, n2, n3, n4) < 0:
        raise ValueError("Invalid parameters: check N, |A0|, |A1|, r.")

    psi0 = np.array([
        np.sqrt(n1 / N),
        np.sqrt(n2 / N),
        np.sqrt(n3 / N),
        np.sqrt(n4 / N),
    ], dtype=float)

    O0 = np.diag([-1, -1,  1,  1])
    O1 = np.diag([-1,  1, -1,  1])
    D = 2 * np.outer(psi0, psi0) - np.eye(4)

    G0 = D @ O0
    G1 = D @ O1

    return psi0, G0, G1


# ============================================================
# 2. Standard Grover steps
# ============================================================
def standard_grover_steps(N: int, r: int) -> int:
    return int(np.round(np.pi / 4 * np.sqrt(N / r)) - 0.5)


# ============================================================
# 2b. First local peak of a probability curve
# ============================================================
def first_local_peak_step(prob_curve: np.ndarray) -> int:
    for t in range(1, len(prob_curve) - 1):
        if prob_curve[t] > prob_curve[t - 1] and prob_curve[t] >= prob_curve[t + 1]:
            return t

    return int(np.argmax(prob_curve))


# ============================================================
# 3. Choose T0 so that 3n = T0
# T0 is the largest multiple of 3 not exceeding standard Grover steps
# ============================================================
def choose_T0_multiple_of_3(N: int, r: int):
    Tg = standard_grover_steps(N, r)
    T0 = Tg - (Tg % 3)
    n = T0 // 3
    return Tg, T0, n


# ============================================================
# 4. Deterministic sequence: (G0 G0 G1)^n
# ============================================================
def run_deterministic_sequence(
    psi0: np.ndarray,
    G0: np.ndarray,
    G1: np.ndarray,
    n: int
):
    T0 = 3 * n
    state = psi0.copy()

    prob_curve = np.zeros(T0 + 1, dtype=float)
    prob_curve[0] = abs(state[0]) ** 2

    step = 0
    for _ in range(n):
        state = G0 @ state
        step += 1
        prob_curve[step] = abs(state[0]) ** 2

        state = G0 @ state
        step += 1
        prob_curve[step] = abs(state[0]) ** 2

        state = G1 @ state
        step += 1
        prob_curve[step] = abs(state[0]) ** 2

    return {
        "prob_curve": prob_curve,
        "final_state": state,
        "final_prob": prob_curve[-1],
        "T0": T0,
    }


# ============================================================
# 5. One random trial
# ============================================================
def run_one_random_trial(
    psi0: np.ndarray,
    G0: np.ndarray,
    G1: np.ndarray,
    p1: float,
    T0: int,
    rng: np.random.Generator
):
    p0 = 1.0 - p1
    if not (0.0 <= p1 <= 1.0):
        raise ValueError("Need p1 in [0,1].")
    if not (0.0 <= p0 <= 1.0):
        raise ValueError("Need p0 in [0,1].")

    state = psi0.copy()
    prob_curve = np.zeros(T0 + 1, dtype=float)
    prob_curve[0] = abs(state[0]) ** 2

    for t in range(1, T0 + 1):
        if rng.random() < p1:
            state = G1 @ state
        else:
            state = G0 @ state

        prob_curve[t] = abs(state[0]) ** 2

    return {
        "prob_curve": prob_curve,
        "final_state": state,
        "final_prob": prob_curve[-1],
    }


# ============================================================
# 6. Run many random trials
# ============================================================
def run_many_random_trials(
    psi0: np.ndarray,
    G0: np.ndarray,
    G1: np.ndarray,
    p1: float,
    T0: int,
    n_trials: int = 50,
    seed_base: int = 1
):
    all_curves = np.zeros((n_trials, T0 + 1), dtype=float)
    final_probs = np.zeros(n_trials, dtype=float)

    for k in range(n_trials):
        rng = np.random.default_rng(seed_base + k)
        result = run_one_random_trial(
            psi0=psi0,
            G0=G0,
            G1=G1,
            p1=p1,
            T0=T0,
            rng=rng
        )
        all_curves[k, :] = result["prob_curve"]
        final_probs[k] = result["final_prob"]

    return {
        "mean_curve": np.mean(all_curves, axis=0),
        "mean_final_prob": np.mean(final_probs),
        "std_final_prob": np.std(final_probs, ddof=1) if n_trials > 1 else 0.0,
        "all_curves": all_curves,
        "final_probs": final_probs,
    }


# ============================================================
# 7. Run one parameter setting
# ============================================================
def run_one_setting(
    N: int,
    size_A0: int,
    size_A1: int,
    r: int,
    p1: float = 1/3,
    n_trials: int = 50,
    seed_base: int = 1
):
    psi0, G0, G1 = build_4d_system(N, size_A0, size_A1, r)
    Tg, T0, n = choose_T0_multiple_of_3(N, r)

    det_result = run_deterministic_sequence(
        psi0=psi0,
        G0=G0,
        G1=G1,
        n=n
    )
    det_result["t_det"] = first_local_peak_step(det_result["prob_curve"])

    rnd_result = run_many_random_trials(
        psi0=psi0,
        G0=G0,
        G1=G1,
        p1=p1,
        T0=T0,
        n_trials=n_trials,
        seed_base=seed_base
    )

    return {
        "N": N,
        "size_A0": size_A0,
        "size_A1": size_A1,
        "r": r,
        "p1": p1,
        "p0": 1 - p1,
        "Tg": Tg,
        "T0": T0,
        "n": n,
        "deterministic": det_result,
        "random": rnd_result,
    }


# ============================================================
# 8. Print summary
# ============================================================
def print_summary(results):
    print("===== Comparison: deterministic (G0G0G1)^n vs random sampling =====")
    for i, res in enumerate(results, start=1):
        print(f"--- Setting {i} ---")
        print(f"N              = {res['N']}")
        print(f"|A0|           = {res['size_A0']}")
        print(f"|A1|           = {res['size_A1']}")
        print(f"r              = {res['r']}")
        print(f"p0             = {res['p0']:.6f}")
        print(f"p1             = {res['p1']:.6f}")
        print(f"Tg             = {res['Tg']}")
        print(f"T0             = {res['T0']}")
        print(f"n              = {res['n']}")
        print(f"t_det          = {res['deterministic']['t_det']}")
        print(f"det final prob = {res['deterministic']['final_prob']:.6f}")
        print(f"rnd mean final = {res['random']['mean_final_prob']:.6f}")
        print(f"rnd std final  = {res['random']['std_final_prob']:.6f}")
        print()


# ============================================================
# 9. Plot one setting: deterministic vs random
# ============================================================
def plot_one_comparison(
    res,
    panel_label=None,
    save_pdf=False,
    filename_prefix="comparison"
):
    # ---- style: match previous small figures ----
    FIG_W, FIG_H = 6.2, 4.4
    BASE = 16
    LABEL = 18
    LEGEND = 13
    TITLE = 16
    ANNOT = 14
    LW_DATA = 2.8
    LW_VLINE = 1.1

    plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size": BASE,
    "axes.labelsize": LABEL,
    "axes.titlesize": TITLE,
    "axes.linewidth": 0.8,
    "axes.labelpad": 4,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.fontsize": LEGEND,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "0.7",
    "legend.handlelength": 2.2,
    "axes.grid": True,
    "grid.linewidth": 0.45,
    "grid.alpha": 0.30,
    "lines.linewidth": LW_DATA,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

    det_curve = res["deterministic"]["prob_curve"]
    rnd_curve = res["random"]["mean_curve"]
    T0 = res["T0"]
    steps = np.arange(T0 + 1)

    # ---- x-axis scaling to avoid offset clash ----
    magnitude = 10 ** max(0, int(np.floor(np.log10(T0))) - 1)
    x_scaled = steps / magnitude
    T0_scaled = T0 / magnitude

    # ---- deterministic first local peak location ----
    det_peak_idx = res["deterministic"]["t_det"]
    det_peak_scaled = det_peak_idx / magnitude
    det_peak_prob = det_curve[det_peak_idx]

    if magnitude >= 1e3:
        exp_str = rf"$\times 10^{{{int(np.log10(magnitude))}}}$"
    else:
        exp_str = ""

    xlabel_str = rf"Iteration step {exp_str}" if exp_str else "Iteration step"

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    ax.plot(
        x_scaled, det_curve,
        linewidth=LW_DATA,
        linestyle="-",
        label="Deterministic",
        zorder=3,
    )

    ax.plot(
        x_scaled, rnd_curve,
        linewidth=LW_DATA,
        linestyle="--",
        label="Random",
        zorder=3,
    )

    # Vertical line at the deterministic peak
    ax.axvline(
        det_peak_scaled,
        linestyle="-.",
        linewidth=LW_VLINE,
        color="0.20",
        alpha=0.75,
        zorder=2,
    )

    # Vertical line at the prescribed stopping time T0
    ax.axvline(
        T0_scaled,
        linestyle=":",
        linewidth=LW_VLINE,
        color="0.35",
        alpha=0.75,
        zorder=2,
    )

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    # ax_top.set_xticks([T0_scaled])
    # ax_top.set_xticklabels([r"$T_0$"], fontsize=ANNOT)
    ax_top.set_xticks([det_peak_scaled, T0_scaled])
    ax_top.set_xticklabels([r"$t_{\mathrm{det}}^\ast$", r"$T_0$"], fontsize=ANNOT)
    ax_top.tick_params(
        axis="x",
        direction="in",
        length=4,
        width=0.7,
        pad=2,
    )
    ax_top.xaxis.set_minor_locator(mticker.NullLocator())

    ax.set_xlabel(xlabel_str)
    ax.set_ylabel(r"$a_0^2(t)$")
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    ax.xaxis.get_major_formatter().set_scientific(False)

    # ---- top title with condition ----
    title_str = (
        rf"$N=2^{{{int(np.log2(res['N']))}}}$, "
        rf"$|\mathcal{{A}}_0|={res['size_A0']}$, "
        rf"$|\mathcal{{A}}_1|={res['size_A1']}$, "
        rf"$r={res['r']}$"
    )
    ax.set_title(title_str, pad=8)

    ax.legend(loc="upper left", frameon=True, borderpad=0.4, labelspacing=0.3,handletextpad=0.5)

    plt.tight_layout(pad=0.4)

    if save_pdf:
        filename = (
            f"{filename_prefix}"
            f"_N{int(np.log2(res['N']))}"
            f"_A0{res['size_A0']}"
            f"_A1{res['size_A1']}"
            f"_r{res['r']}.pdf"
        )
        fig.savefig(filename)
        print(f"Saved: {filename}")

    plt.show()
    plt.close(fig)


# ============================================================
# 10. Run four settings and plot four separate small figures
# ============================================================
if __name__ == "__main__":
    settings = [
        {"panel": "A", "N": 2**45, "size_A0": 50,  "size_A1": 20, "r": 1},
       {"panel": "B", "N": 2**45, "size_A0": 50,  "size_A1": 50, "r": 1},
        {"panel": "C", "N": 2**45, "size_A0": 50, "size_A1": 100, "r": 1},
        {"panel": "D", "N": 2**45, "size_A0": 50, "size_A1": 400, "r": 1},
    ]

    results = []
    for i, s in enumerate(settings):
        res = run_one_setting(
            N=s["N"],
            size_A0=s["size_A0"],
            size_A1=s["size_A1"],
            r=s["r"],
            p1=1/3,
            n_trials=50,
            seed_base=1 + 1000 * i
        )
        results.append(res)

    print_summary(results)

    for s, res in zip(settings, results):
        plot_one_comparison(
            res,
            panel_label=s["panel"],
            save_pdf=True,
            filename_prefix="cmp"
        )
