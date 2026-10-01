import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Parameters
# ============================================================

q = np.linspace(0, 0.14, 4000)

N_values = [2, 3, 4, 5, 6, 50]


# ============================================================
# Binary entropy
# ============================================================

def h(p):
    p = np.asarray(p, dtype=float)

    out = np.zeros_like(p)
    mask = (p > 0) & (p < 1)

    out[mask] = (
        -p[mask] * np.log2(p[mask])
        - (1 - p[mask]) * np.log2(1 - p[mask])
    )

    return out


# ============================================================
# pCHSH entropy bound
# ============================================================

def f_chsh(omega):
    inner = (4 * omega - 2) ** 2 - 1
    inner = np.maximum(inner, 0)

    arg = 0.5 + 0.5 * np.sqrt(inner)

    return 1 - h(arg)


# ============================================================
# pMSG lower entropy bound
# ============================================================

def f_msg(p):
    p = np.asarray(p)

    terms = np.array([
        265.55333 * p - 263.58333,
        66.7154   * p - 64.7455,
        20.296    * p - 19.2487,
        51.89538  * p - 49.99236,
        41.8976   * p - 40.1431,
    ])

    return np.max(terms, axis=0)


# ============================================================
# Colors
# ============================================================

base_colors = plt.cm.viridis(
    np.linspace(0, 0.85, 5)
)

colors = {
    2: base_colors[0],
    3: base_colors[1],
    4: base_colors[2],
    5: base_colors[3],
    6: base_colors[4],
    50: "red",
}


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))


for N in N_values:

    # ========================================================
    # pCHSH
    #
    # p_exp = 1/2
    #       + [(1-q) + (1-q)^(N-1)] / (4 sqrt(2))
    #
    # For N = 2:
    # p_exp = 1/2 + (1-q)/(2 sqrt(2))
    # ========================================================

    p_exp_chsh = (
        0.5
        + (
            (1 - q)
            + (1 - q) ** (N - 1)
        ) / (4 * np.sqrt(2))
    )

    r_chsh = (
        f_chsh(p_exp_chsh)
        - h(q / 2)
    )

    r_chsh = np.maximum(
        r_chsh,
        0
    )


    # ========================================================
    # pMSG -- corrected per-qubit Bob noise
    #
    # Let r = 1-q.
    #
    # p_exp = 1/2
    #       + r/9
    #       + r^2/18
    #       + r^(N-1)/9
    #       + r^N/9
    #       + r^(2N-2)/9
    #
    # For N = 2:
    #
    # p_exp = 1/2 + 2r/9 + 5r^2/18
    #
    # which accounts for the case where exactly one
    # of B_1's two qubits is depolarized.
    # ========================================================

    r = 1 - q

    p_exp_msg = (
        0.5
        + r / 9
        + r ** 2 / 18
        + r ** (N - 1) / 9
        + r ** N / 9
        + r ** (2 * N - 2) / 9
    )

    # pMSG produces two key bits, each with QBER q/2
    r_msg = (
        f_msg(p_exp_msg)
        - 2 * h(q / 2)
    )

    r_msg = np.maximum(
        r_msg,
        0
    )


    # ========================================================
    # Curves
    # ========================================================

    ax.plot(
        q,
        r_chsh,
        color=colors[N],
        linestyle="-",
        linewidth=1.8,
        label=rf"$N={N}$, pCHSH"
    )

    ax.plot(
        q,
        r_msg,
        color=colors[N],
        linestyle="--",
        linewidth=1.8,
        label=rf"$N={N}$, pMSG"
    )


# ============================================================
# Formatting
# ============================================================

ax.set_xlabel(
    r"Local depolarizing noise probability $q$",
    fontsize=12
)

ax.set_ylabel(
    "Key rate",
    fontsize=12
)

ax.set_title(
    r"Asymptotic key rate vs Local depolarizing noise probability "
    r"for $N=2$ to $N=6$, and $N=50$",
    fontsize=13
)

ax.set_xlim(0, 0.14)
ax.set_ylim(0, 2.05)

ax.set_xticks(
    np.arange(0, 0.141, 0.01)
)

ax.set_yticks(
    np.arange(0, 2.01, 0.25)
)

ax.grid(
    True,
    alpha=0.25
)

ax.legend(
    loc="upper right",
    ncol=2,
    fontsize=8
)

plt.tight_layout()
plt.show()