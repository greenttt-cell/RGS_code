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
# 2. Basic quantities
# ============================================================
def union_size(size_A0: int, size_A1: int, r: int) -> int:
    return size_A0 + size_A1 - r


def compute_p1_from_lemma(N: int, m: int, r: int, delta: float) -> float:
    denom = delta - 4 * r / N
    if denom <= 0:
        raise ValueError(
            f"Need delta > 4r/N = {4*r/N:.6e}, but got delta = {delta}."
        )

    numer = 4 * (m - r) / np.sqrt(r * N) + 2 * np.sqrt((m - r) / N)
    return numer / denom


def compute_T0_from_theorem(N: int, m: int, r: int) -> int:
    alpha0_sq = r / N
    alpha3_sq = (N - m) / N

    if alpha0_sq <= 0:
        raise ValueError("Need r > 0.")
    if alpha3_sq <= 0:
        raise ValueError("Need N > m.")

    alpha0 = np.sqrt(alpha0_sq)
    alpha3 = np.sqrt(alpha3_sq)

    inside_gamma = 4 * alpha0_sq * alpha3_sq - (1 - alpha0_sq - alpha3_sq) ** 2
    if inside_gamma <= 0:
        raise ValueError("Gamma^2 <= 0. The formula is not applicable.")

    Gamma = np.sqrt(inside_gamma)

    inside_rho = 2 * (alpha0_sq + alpha3_sq) - 1
    if inside_rho <= 0:
        raise ValueError("rho^2 <= 0. The formula is not applicable.")

    rho = np.sqrt(inside_rho)

    theta = np.arcsin(np.clip(Gamma / rho, -1.0, 1.0))
    beta = np.arcsin(np.clip(Gamma / (2 * alpha3), -1.0, 1.0))

    T0 = int(np.floor(np.pi / (2 * theta) - beta / theta))
    return max(T0, 0)


# ============================================================
# 3. Random trial
# ============================================================
def run_one_random_trial(psi0, G0, G1, p1, T0, rng):
    state = psi0.copy()
    prob_curve = np.zeros(T0 + 1, dtype=float)
    prob_curve[0] = abs(state[0]) ** 2

    g1_count = 0

    for t in range(1, T0 + 1):
        if rng.random() < p1:
            state = G1 @ state
            g1_count += 1
        else:
            state = G0 @ state

        prob_curve[t] = abs(state[0]) ** 2

    return prob_curve, g1_count


# ============================================================
# 4. Only G0 baseline
# ============================================================
def run_only_G0(psi0, G0, T0):
    state = psi0.copy()
    prob_curve = np.zeros(T0 + 1, dtype=float)
    prob_curve[0] = abs(state[0]) ** 2

    for t in range(1, T0 + 1):
        state = G0 @ state
        prob_curve[t] = abs(state[0]) ** 2

    return prob_curve


# ============================================================
# 5. Run one setting
# Uses online mean/std to save memory
# ============================================================
def run_one_setting(
    N,
    size_A0,
    size_A1,
    r,
    delta,
    n_trials=50,
    seed_base=1
):
    psi0, G0, G1 = build_4d_system(N, size_A0, size_A1, r)

    m = union_size(size_A0, size_A1, r)
    p1 = compute_p1_from_lemma(N, m, r, delta)
    p0 = 1.0 - p1
    T0 = compute_T0_from_theorem(N, m, r)

    if not (0 <= p1 <= 1):
        raise ValueError(f"p1={p1:.6f} is not in [0,1].")

    sum_curve = np.zeros(T0 + 1)
    g1_ratios = []
    final_probs = []

    for k in range(n_trials):
        rng = np.random.default_rng(seed_base + k)
        curve, g1_count = run_one_random_trial(
            psi0=psi0,
            G0=G0,
            G1=G1,
            p1=p1,
            T0=T0,
            rng=rng
        )

        sum_curve += curve
        g1_ratios.append(g1_count / T0)
        final_probs.append(curve[-1])

    mean_curve = sum_curve / n_trials
    only_G0_curve = run_only_G0(psi0, G0, T0)

    return {
        "N": N,
        "size_A0": size_A0,
        "size_A1": size_A1,
        "r": r,
        "m": m,
        "delta": delta,
        "p0": p0,
        "p1": p1,
        "T0": T0,
        "n_trials": n_trials,
        "mean_curve": mean_curve,
        "only_G0_curve": only_G0_curve,
        "mean_final_prob": float(np.mean(final_probs)),
        "std_final_prob": float(np.std(final_probs, ddof=1)) if n_trials > 1 else 0.0,
        "mean_g1_ratio": float(np.mean(g1_ratios)),
    }


# ============================================================
# 6. Plot revised Figure 2
# Fixed large N, vary delta and |A0|
# Each panel shows:
#   Random mean probability
#   Only G0 baseline
#   p1 above the panel
# ============================================================
def plot_vary_delta_for_fixed_A0(
    results_list,
    save_pdf=True,
    filename="vary_delta.pdf"
):
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

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    

    res0 = results_list[0]
    T0 = res0["T0"]
    magnitude = 10 ** max(0, int(np.floor(np.log10(T0))) - 1)

    for res in results_list:
        x_scaled = np.arange(res["T0"] + 1) / magnitude

        ax.plot(
            x_scaled,
            res["mean_curve"],
            linewidth=2.0,
            label=rf"$\delta={res['delta']}$, $p_1={res['p1']:.3f}$"
        )

    # only G0 baseline: same for all delta under fixed A0
    x_scaled = np.arange(res0["T0"] + 1) / magnitude
    ax.plot(
        x_scaled,
        res0["only_G0_curve"],
        linestyle="--",
        color="black",
        linewidth=1.6,
        label=r"Only $G_0$"
    )

    T0_scaled = T0 / magnitude
    ax.axvline(T0_scaled, linestyle=":", color="0.35", linewidth=1.1)

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks([T0_scaled])
    ax_top.set_xticklabels([r"$T_0$"])
    ax_top.tick_params(axis="x", direction="in", length=3, width=0.7, pad=2)
    ax_top.xaxis.set_minor_locator(mticker.NullLocator())

    exp = int(np.log10(magnitude)) if magnitude >= 1e3 else None
    if exp is not None:
        ax.set_xlabel(rf"Iteration step $\times 10^{{{exp}}}$")
    else:
        ax.set_xlabel("Iteration step")

    ax.set_ylabel(r"$a_0^2(t)$")
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))

    ax.set_title(
        rf"$N=2^{{{int(np.log2(res0['N']))}}}$, "
        rf"$|\mathcal{{A}}_0|={res0['size_A0']}$, "
        rf"$|\mathcal{{A}}_1|={res0['size_A1']}$, "
        rf"$r={res0['r']}$"
    )

    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()

    if save_pdf:
        fig.savefig(filename)
        print(f"Saved: {filename}")

    plt.show()
    plt.close(fig)

if __name__ == "__main__":

    N = 2**40
    size_A1 = 100
    r = 1
    n_trials = 50
    seed_base = 1

    delta_list = [0.1, 0.3, 0.5]

    # Two different A0 settings
    A0_settings = [100, 1000]

    for idx, size_A0 in enumerate(A0_settings):

        results_delta = []

        for j, delta in enumerate(delta_list):
            res = run_one_setting(
                N=N,
                size_A0=size_A0,
                size_A1=size_A1,
                r=r,
                delta=delta,
                n_trials=n_trials,
                seed_base=seed_base + 1000 * idx + 100 * j
            )

            print("===================================")
            print(f"N          = 2^{int(np.log2(N))}")
            print(f"|A0|       = {size_A0}")
            print(f"|A1|       = {size_A1}")
            print(f"r          = {r}")
            print(f"delta      = {delta}")
            print(f"p1         = {res['p1']:.8f}")
            print(f"mean final = {res['mean_final_prob']:.6f}")
            print(f"G1/T0      = {res['mean_g1_ratio']:.6f}")

            results_delta.append(res)

        plot_vary_delta_for_fixed_A0(
            results_delta,
            save_pdf=True,
            filename=f"fig2_{size_A0}.pdf"
        )