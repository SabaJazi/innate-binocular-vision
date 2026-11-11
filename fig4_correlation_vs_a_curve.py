"""
Figure 5: Inter-layer correlation as a function of the transfer parameter alpha.

Sweeps alpha from 0.0 to 1.0 across several p values, runs the LGN model
`N_TRIALS` times per point, and plots the mean inter-layer Pearson
correlation rho with standard-error bars.

NOTE on parameter choice: at fixed r and t, if p is too high the wave
saturates the whole grid (every cell fires), which gives zero variance and
an undefined correlation. The p values below are chosen to stay in the
non-saturating regime so rho is well defined across the full alpha range.

Run from the same folder as lgn_ibv.py:
    python figure5_correlation_vs_a.py
"""

import matplotlib
matplotlib.use("Agg")          # no display needed; saves straight to file
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from lgn_ibv import LGN        # reuse the model class

# ---------------------------------------------------------------------------
# Sweep settings  (edit here)
# ---------------------------------------------------------------------------

A_VALUES  = np.linspace(0.0, 1.0, 11)   # x-axis: 0.0, 0.1, ..., 0.5
P_VALUES  = [0.04, 0.06, 0.08, 0.10]    # non-saturating p values (one line each)
R         = 4                          # fixed propagation radius
T         = 3                          # fixed activation threshold
LGN_WIDTH = 128                        #  grid; large enough to avoid edge effects
N_TRIALS  = 10                        # repeats per (p, a) point, averaged

# ---------------------------------------------------------------------------
# Plot style (IEEE single-column friendly)
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.linewidth": 0.6,
})

# Distinct marker + linestyle per p so lines are readable in grayscale
STYLES = [
    {"marker": "*", "linestyle": ":",  "color":"#999999"},
    {"marker": "s", "linestyle": "--", "color": "#666666"},
    {"marker": "d", "linestyle": "-.",  "color":"#444444"},
    {"marker": "o", "linestyle": "-",  "color":"#222222"},

]


 
def mean_correlation(p: float, a: float, n_trials: int) -> tuple:
    """
    Run the LGN model `n_trials` times for given (p, a) and return
    (mean correlation, standard error). NaN results (saturated or empty
    layers) are skipped so they don't corrupt the average.
    """
    corrs = []
    for k in range(n_trials):
        np.random.seed(k)  # reproducible across runs
        lgn = LGN(width=LGN_WIDTH, p=p, r=R, t=T, trans=a,
                  num_layers=2, make_wave=True)
        c = lgn.correlation()
        if not np.isnan(c):
            corrs.append(c)

    if not corrs:
        return float("nan"), 0.0
    corrs = np.array(corrs)
    err = corrs.std(ddof=1) / np.sqrt(len(corrs)) if len(corrs) > 1 else 0.0
    return corrs.mean(), err


# ---------------------------------------------------------------------------
# Run sweep & plot
# ---------------------------------------------------------------------------
P_C = 0.11

fig, ax = plt.subplots(figsize=(3.5, 2.7))

for i, p in enumerate(P_VALUES):
    means, errs = [], []
    for a in A_VALUES:
        m, e = mean_correlation(p, a, N_TRIALS)
        means.append(m)
        errs.append(e)
        print(f"p={p:.2f}  a={a:.2f}  ->  rho={m:.3f} +/- {e:.3f}")

    ax.errorbar(
        A_VALUES, means, yerr=errs,
        label=f"p = {p:.2f}",
        markersize=4, linewidth=1.0, capsize=2,
        **STYLES[i % len(STYLES)],
    )

# Keep p_c as the final legend entry, after the p-series, but make it visually distinct.
p_c_handle = Line2D([], [], linestyle="None", marker="X", markerfacecolor="white",
                   markeredgecolor="black", markersize=5.5, markeredgewidth=1.2,
                   label=rf"$p_c \approx {P_C:.2f}$")
legend_handles, legend_labels = ax.get_legend_handles_labels()
legend_handles.append(p_c_handle)
legend_labels.append(rf"$p_c \approx {P_C:.2f}$")

ax.set_xlabel(r"Transfer parameter $\alpha$")
ax.set_ylabel(r"Mean inter-layer correlation $\rho$")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0, 1.0)
for label in ax.get_yticklabels():
    label.set_rotation(90)
    label.set_verticalalignment('center')
    label.set_horizontalalignment('center')
ax.grid(True, linewidth=0.3, alpha=0.4)
ax.legend(frameon=False, fontsize=6, loc="lower right", title="Recruitable fraction",
         handles=legend_handles, labels=legend_labels, borderpad=0.35,
         labelspacing=0.4, handletextpad=0.2, columnspacing=1.0)

# Remove extra whitespace so the data nearly reaches the figure edges.
fig.subplots_adjust(left=0.16, right=0.98, bottom=0.17, top=0.96)

# Save in IEEE-accepted formats
fig.savefig("figure4_correlation_vs_a.pdf")
fig.savefig("figure4_correlation_vs_a.png", dpi=600)
print("\nSaved figure4_correlation_vs_a.{pdf,png}")