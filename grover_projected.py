#code for projected trajectory
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Global style: close to LaTeX amsart 12pt
# =====================================================
plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})


# =====================================================
# Build reduced averaged Grover system
# =====================================================
def build_system(N: int, a0: int, a1: int, r: int = 1):
    if not (0 < r <= a0 and 0 < r <= a1 and a0 + a1 - r <= N):
        raise ValueError("Invalid parameters.")

    psi = np.array([
        np.sqrt(r / N),
        np.sqrt((a0 - r) / N),
        np.sqrt((a1 - r) / N),
        np.sqrt((N - a0 - a1 + r) / N),
    ], dtype=float)

    O1 = np.diag([-1, -1,  1,  1])
    O2 = np.diag([-1,  1, -1,  1])

    D = 2 * np.outer(psi, psi) - np.eye(4)

    G1 = D @ O1
    G2 = D @ O2
    G_avg = 0.5 * (G1 + G2)

    return psi, G_avg


# =====================================================
# Simulate trajectory
# x = \hat{a}_3(t), y = \hat{a}_0(t)
# =====================================================
def simulate_projected_trajectory(N: int, a0: int, a1: int, r: int = 1):
    psi, G_avg = build_system(N, a0, a1, r)

    steps = int(np.round(np.pi / 4 * np.sqrt(N / r)))

    state = psi.copy()
    x_vals = []
    y_vals = []

    for _ in range(steps + 1):
        y_vals.append(state[0])   # \hat{a}_0(t)
        x_vals.append(state[3])   # \hat{a}_3(t)
        state = G_avg @ state

    return np.array(x_vals), np.array(y_vals)


# =====================================================
# Quarter circle
# =====================================================
def quarter_circle(n: int = 500):
    x = np.linspace(0.0, 1.0, n)
    y = np.sqrt(np.maximum(0.0, 1.0 - x**2))
    return x, y


# =====================================================
# Draw two small arrows:
# initial state  -> near (1,0)
# final state    -> near (0,1)
# =====================================================
def draw_endpoint_arrows(ax, x_vals, y_vals):
    # 轨迹是从接近 (0,1) 往接近 (1,0) 走
    # 但你要标“初态”为靠近 (1,0)，标“末态”为靠近 (0,1)
    # 所以这里按几何位置来标，而不是按数组先后顺序来标

    # ---------- arrow near (1,0): initial state ----------
    idx_init = np.argmax(x_vals - y_vals)   # 更靠右下角
    idx_init_prev = max(idx_init - 3, 0)

    xi, yi = x_vals[idx_init], y_vals[idx_init]
    xi0, yi0 = x_vals[idx_init_prev], y_vals[idx_init_prev]

    vix = xi - xi0
    viy = yi - yi0
    norm_i = np.hypot(vix, viy)

    if norm_i > 0:
        vix /= norm_i
        viy /= norm_i
        L = 0.065
        ax.annotate(
            "",
            xy=(xi, yi),
            xytext=(xi - L * vix, yi - L * viy),
            arrowprops=dict(
                arrowstyle="->",
                lw=1.4,
                mutation_scale=10
            ),
            zorder=6,
            clip_on=False
        )

    # ---------- arrow near (0,1): final state ----------
    idx_final = np.argmax(y_vals - x_vals)  # 更靠左上角
    idx_final_next = min(idx_final + 3, len(x_vals) - 1)

    xf, yf = x_vals[idx_final], y_vals[idx_final]
    xf1, yf1 = x_vals[idx_final_next], y_vals[idx_final_next]

    vfx = xf - xf1
    vfy = yf - yf1
    norm_f = np.hypot(vfx, vfy)

    if norm_f > 0:
        vfx /= norm_f
        vfy /= norm_f
        L = 0.065
        ax.annotate(
            "",
            xy=(xf, yf),
            xytext=(xf - L * vfx, yf - L * vfy),
            arrowprops=dict(
                arrowstyle="->",
                lw=1.4,
                mutation_scale=10
            ),
            zorder=6,
            clip_on=False
        )


# =====================================================
# Main figure
# =====================================================
#     plt.show()
def make_publication_figure():
    cases = [
        {"label": "(A)", "N": 2**15, "a0": 200, "a1": 100, "r": 1},
        {"label": "(B)", "N": 2**20, "a0": 200, "a1": 100, "r": 1},
        {"label": "(C)", "N": 2**25, "a0": 200, "a1": 100, "r": 1},
        {"label": "(D)", "N": 2**35, "a0": 200, "a1": 100, "r": 1},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    axes = axes.flatten()

    x_ref, y_ref = quarter_circle()

    for idx, (ax, case) in enumerate(zip(axes, cases)):
        x_vals, y_vals = simulate_projected_trajectory(
            case["N"], case["a0"], case["a1"], case["r"]
        )

        ax.plot(
            x_ref,
            y_ref,
            linestyle="--",
            linewidth=1.2
        )

        ax.plot(
            x_vals,
            y_vals,
            linewidth=1.4
        )

        draw_endpoint_arrows(ax, x_vals, y_vals)

        ax.set_xlim(0.0, 1.02)
        ax.set_ylim(0.0, 1.02)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

        if idx in [2, 3]:
            ax.set_xlabel(r"$\widehat{a}_3(t)$",labelpad=2)
        else:
            ax.set_xlabel("")

        
        ax.set_ylabel(r"$\widehat{a}_0(t)$", labelpad=2)
    

        n_pow = int(np.log2(case["N"]))
        title = rf"{case['label']} $N=2^{{{n_pow}}}$"
        ax.set_title(title, pad=3)

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.08,
        top=0.94,
        wspace=0.18,
        hspace=0.22
    )

    fig.savefig("projected_trajectories_clean.pdf", bbox_inches="tight")
    fig.savefig("projected_trajectories_clean.png", dpi=600, bbox_inches="tight")

    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    make_publication_figure()