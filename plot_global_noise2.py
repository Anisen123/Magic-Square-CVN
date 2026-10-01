import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.transforms import ScaledTranslation


# ============================================================
# Parameters
# ============================================================

q = np.linspace(0, 0.16, 3000)


# ============================================================
# Binary entropy
# ============================================================

def h(p):
    p = np.asarray(p)

    out = np.zeros_like(p, dtype=float)
    mask = (p > 0) & (p < 1)

    out[mask] = (
        -p[mask] * np.log2(p[mask])
        - (1 - p[mask]) * np.log2(1 - p[mask])
    )

    return out


# ============================================================
# pCHSH entropy bound f'_gamma
# ============================================================

def f_chsh(omega):
    inner = (4 * omega - 2) ** 2 - 1
    inner = np.maximum(inner, 0)

    arg = 0.5 + 0.5 * np.sqrt(inner)

    return 1 - h(arg)


# ============================================================
# pMSG lower entropy bound f_gamma
# ============================================================

def f_msg(p):
    terms = np.array([
        265.55333 * p - 263.58333,
        66.7154   * p - 64.7455,
        20.296    * p - 19.2487,
        51.89538  * p - 49.99236,
        41.8976   * p - 40.1431,
    ])

    return np.max(terms, axis=0)


# ============================================================
# Noise model
# ============================================================

Q = q / 2

p_exp = 1 - Q


# ============================================================
# pCHSH key rate
# ============================================================

omega_chsh = (
    0.5
    + (1 - q) / (2 * np.sqrt(2))
)

r_chsh = (
    f_chsh(omega_chsh)
    - h(Q)
)

r_chsh = np.maximum(
    r_chsh,
    0
)


# ============================================================
# pMSG LOWER BOUND
# ============================================================

H_msg_lower = f_msg(
    p_exp
)

r_msg_lower = (
    H_msg_lower
    - h(Q)
)

r_msg_lower = np.maximum(
    r_msg_lower,
    0
)


# ============================================================
# pMSG UPPER BOUND
# ============================================================

omega_22 = (
    (2 + np.sqrt(2)) / 4
)

omega_bar = (
    1
    - (4 / 9) * (1 - omega_22)
)

lambda_msg = (
    (p_exp - omega_bar)
    / (1 - omega_bar)
)

lambda_msg = np.clip(
    lambda_msg,
    0,
    1
)

H_msg_upper = (
    2 * lambda_msg
)

r_msg_upper = (
    H_msg_upper
    - h(Q)
)

r_msg_upper = np.maximum(
    r_msg_upper,
    0
)


# ============================================================
# Determine comparison regimes
# ============================================================
#
# State convention:
#
#  2 = both zero
#  1 = pMSG strictly better
#  0 = open
# -1 = pCHSH strictly better
#
# pMSG strictly better:
#     pMSG lower bound > pCHSH tight bound
#
# pCHSH strictly better:
#     pCHSH tight bound > pMSG upper bound
#
# both zero:
#     pCHSH rate = 0 and the full pMSG interval is also 0
# ============================================================

msg_strictly_better = (
    r_msg_lower > r_chsh
)

chsh_strictly_better = (
    r_chsh > r_msg_upper
)

both_zero = (
    np.isclose(r_chsh, 0)
    & np.isclose(r_msg_lower, 0)
    & np.isclose(r_msg_upper, 0)
)


regime = np.zeros_like(
    q,
    dtype=int
)

regime[msg_strictly_better] = 1

regime[chsh_strictly_better] = -1

# "both zero" overrides other classifications
regime[both_zero] = 2


# ============================================================
# Find regime-transition points
# ============================================================

change_idx = np.where(
    regime[:-1] != regime[1:]
)[0]

boundaries = [
    q[0]
]

for i in change_idx:
    boundaries.append(
        0.5 * (q[i] + q[i + 1])
    )

boundaries.append(
    q[-1]
)

segment_start_indices = [
    0,
    *list(change_idx + 1)
]

segment_states = [
    regime[i]
    for i in segment_start_indices
]


# ============================================================
# Print transition points
# ============================================================

print("Regime transitions:")

for boundary in boundaries[1:-1]:
    print(
        f"q ≈ {boundary:.5f}"
    )


# ============================================================
# Custom legend handler for pCHSH tight bound
# ============================================================

class HandlerTightBound(HandlerBase):

    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans
    ):

        line_blue, line_red = orig_handle

        x = [
            xdescent,
            xdescent + width
        ]

        y_blue = (
            ydescent
            + 0.35 * height
        )

        y_red = (
            ydescent
            + 0.65 * height
        )

        blue_artist = Line2D(
            x,
            [y_blue, y_blue],
            color=line_blue.get_color(),
            linewidth=1.4,
            linestyle="-",
            transform=trans
        )

        red_artist = Line2D(
            x,
            [y_red, y_red],
            color=line_red.get_color(),
            linewidth=1.4,
            linestyle="-",
            transform=trans
        )

        return [
            blue_artist,
            red_artist
        ]


# ============================================================
# Figure
# ============================================================

fig, (
    ax,
    ax_regime
) = plt.subplots(
    2,
    1,
    figsize=(10, 6.8),
    sharex=True,
    gridspec_kw={
        "height_ratios": [
            14,
            1.25
        ],
        "hspace": 0.06
    }
)


# ============================================================
# Sparse diagonal shading between pMSG bounds
# ============================================================

shade_slope = 20.0

shade_spacing = 0.40

b_min = (
    -shade_slope * q.max()
)

b_max = 2.2


for b in np.arange(
    b_min,
    b_max,
    shade_spacing
):

    y_diag = (
        shade_slope * q
        + b
    )

    inside = (
        (r_msg_upper > r_msg_lower)
        & (y_diag >= r_msg_lower)
        & (y_diag <= r_msg_upper)
    )

    y_shade = np.where(
        inside,
        y_diag,
        np.nan
    )

    ax.plot(
        q,
        y_shade,
        color="gray",
        linewidth=0.7,
        alpha=0.45,
        zorder=1
    )


# ============================================================
# pMSG bounds
# ============================================================

msg_lower_line, = ax.plot(
    q,
    r_msg_lower,
    color="tab:orange",
    linestyle="--",
    linewidth=1.8,
    label="pMSG lower bound",
    zorder=3
)

msg_upper_line, = ax.plot(
    q,
    r_msg_upper,
    color="tab:green",
    linestyle="--",
    linewidth=1.8,
    label="pMSG upper bound",
    zorder=3
)


# ============================================================
# pCHSH tight bound
# ============================================================

offset_pts = 0.9

upper_offset = ScaledTranslation(
    0,
    offset_pts / 72,
    fig.dpi_scale_trans
)

lower_offset = ScaledTranslation(
    0,
    -offset_pts / 72,
    fig.dpi_scale_trans
)


chsh_blue, = ax.plot(
    q,
    r_chsh,
    color="tab:blue",
    linestyle="-",
    linewidth=1.4,
    transform=(
        ax.transData
        + upper_offset
    ),
    zorder=4
)

chsh_red, = ax.plot(
    q,
    r_chsh,
    color="tab:red",
    linestyle="-",
    linewidth=1.4,
    transform=(
        ax.transData
        + lower_offset
    ),
    zorder=4
)


# ============================================================
# Main plot labels / axes
# ============================================================

ax.set_ylabel(
    r"Key rate",
    fontsize=12
)

ax.set_title(
    r"Asymptotic key rate vs Global depolarizing noise probability $q$",
    fontsize=13
)

ax.set_xlim(
    0,
    0.16
)

ax.set_ylim(
    0,
    2.1
)

ax.set_xticks(
    np.arange(
        0,
        0.161,
        0.02
    )
)

ax.set_yticks(
    np.arange(
        0,
        2.01,
        0.25
    )
)

ax.grid(
    True,
    alpha=0.25
)

ax.set_axisbelow(
    True
)

ax.tick_params(
    axis="x",
    labelbottom=False
)


# ============================================================
# Main legend
# ============================================================

ax.legend(
    handles=[
        msg_lower_line,
        msg_upper_line,
        (
            chsh_red,
            chsh_blue
        )
    ],
    labels=[
        "pMSG lower bound",
        "pMSG upper bound",
        "pCHSH tight bound"
    ],
    handler_map={
        tuple: HandlerTightBound()
    },
    loc="upper right"
)


# ============================================================
# Regime bar
# ============================================================

regime_style = {
    1: {
        "color": "tab:orange",
        "label": "pMSG strictly better"
    },

    0: {
        "color": "lightgray",
        "label": "open"
    },

    -1: {
        "color": "tab:blue",
        "label": "pCHSH strictly better"
    },

    2: {
        "color": "white",
        "label": "both zero"
    }
}


for (
    left,
    right,
    state
) in zip(
    boundaries[:-1],
    boundaries[1:],
    segment_states
):

    style = regime_style[state]

    ax_regime.axvspan(
        left,
        right,
        ymin=0,
        ymax=1,
        color=style["color"],
        alpha=0.55,
        linewidth=0
    )

    interval_width = (
        right - left
    )

    if interval_width > 0.008:

        ax_regime.text(
            0.5 * (left + right),
            0.5,
            style["label"],
            ha="center",
            va="center",
            fontsize=8.5
        )


# ============================================================
# Regime transition separators
# ============================================================

for boundary in boundaries[1:-1]:

    ax_regime.axvline(
        boundary,
        color="black",
        linewidth=0.8,
        alpha=0.7
    )


# ============================================================
# Regime-bar formatting
# ============================================================

ax_regime.set_ylim(
    0,
    1
)

ax_regime.set_yticks(
    []
)

ax_regime.set_xlim(
    0,
    0.16
)

ax_regime.set_xticks(
    np.arange(
        0,
        0.161,
        0.02
    )
)

ax_regime.set_xlabel(
    r"Global depolarizing noise probability $q$",
    fontsize=12
)

ax_regime.tick_params(
    axis="x",
    labelsize=10
)

for spine in ax_regime.spines.values():
    spine.set_linewidth(
        0.8
    )


# ============================================================
# Final layout
# ============================================================

plt.tight_layout()

plt.show()