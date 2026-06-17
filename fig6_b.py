"""
Figure 4b: Filter quality vs parameters (curve version).

Same analysis as figure4_filter_quality.py (heatmap), but plotted as line
curves instead:
    x-axis: recruitable fraction p
    y-axis: % of structured filters
    one curve per transfer parameter a

For each (p, a) point the script generates ICA filters from the LGN-IBV
forward path, computes an orientation-selectivity index (OSI) for each filter
from its 2D Fourier power spectrum, and reports the percentage of filters with
OSI above OSI_THRESHOLD.

A note on the threshold
-----------------------
ICA on binocular LGN activity tends to yield oriented filters across a wide
parameter range, so OSI values cluster fairly high. If every point comes out
near 100%, the OSI_THRESHOLD is below the whole OSI distribution: print the
per-filter OSI values, inspect their min/mean/max, and raise OSI_THRESHOLD
into that range (or narrow ANGLE_WINDOW) so the metric discriminates.

Run from the same folder as lgn_ibv.py:
    python figure4b_filter_quality_curves.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lgn_ibv import (
    generate_patches,
    perform_ica,
    generate_ident_hash,
)

# ---------------------------------------------------------------------------
# Sweep + analysis settings  (edit here)
# ---------------------------------------------------------------------------

P_VALUES   = [0.04, 0.06, 0.08, 0.10]    # x-axis
A_VALUES   = [0.1, 0.2, 0.3, 0.4, 0.5]   # one curve per a
R          = 4                           # fixed propagation radius
T          = 3                           # fixed activation threshold
LGN_WIDTH  = 256
PATCH_SIZE = 16
NUM_PATCHES    = 50000
NUM_COMPONENTS = 50
OSI_THRESHOLD  = 0.3
ANGLE_WINDOW   = 10.0

# ---------------------------------------------------------------------------
# Orientation selectivity
# ---------------------------------------------------------------------------

def orientation_selectivity(filter_2d: np.ndarray,
                            angle_window: float = ANGLE_WINDOW) -> float:
    """
    Orientation selectivity index from the 2D Fourier power spectrum.

    Returns a value in [0, 1]: the fraction of spectral energy lying within
    +/- `angle_window` degrees of the dominant orientation.
    """
    f = filter_2d - filter_2d.mean()
    if np.allclose(f, 0):
        return 0.0

    power = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2

    h, w = power.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    angles = np.degrees(np.arctan2(yy - cy, xx - cx)) % 180.0

    mask = ~((yy == cy) & (xx == cx))
    angles = angles[mask]
    energy = power[mask]

    total = energy.sum()
    if total <= 0:
        return 0.0

    bins = np.zeros(180)
    np.add.at(bins, np.clip(angles.astype(int), 0, 179), energy)
    dominant = int(np.argmax(bins))

    d = np.abs(angles - dominant)
    d = np.minimum(d, 180.0 - d)
    return energy[d <= angle_window].sum() / total


def fraction_structured(p: float, a: float) -> float:
    """
    Generate ICA filters for one (p, a) point and return the percentage whose
    orientation selectivity exceeds OSI_THRESHOLD.
    """
    ident = generate_ident_hash(p, a, R, T)
    scratch = "_fig4b_scratch"

    patches, _ = generate_patches(
        NUM_PATCHES, PATCH_SIZE, LGN_WIDTH,
        p, R, T, a, ident, scratch,
    )
    components = perform_ica(NUM_COMPONENTS, patches)

    half = components.shape[1] // 2
    dim = int(np.sqrt(half))

    osis = []
    for comp in components:
        left = comp[:half].reshape(dim, dim)
        right = comp[half:].reshape(dim, dim)
        osis.append(max(orientation_selectivity(left),
                        orientation_selectivity(right)))

    osis = np.array(osis)
    return 100.0 * np.mean(osis > OSI_THRESHOLD)


# ---------------------------------------------------------------------------
# Run the sweep
# ---------------------------------------------------------------------------

# results[a] -> list of % structured, one per p
results = {a: [] for a in A_VALUES}

for a in A_VALUES:
    for p in P_VALUES:
        pct = fraction_structured(p, a)
        results[a].append(pct)
        print(f"p={p:.2f}  a={a:.2f}  ->  {pct:5.1f}% structured")

# ---------------------------------------------------------------------------
# Plot curves (IEEE single-column friendly)
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.6,
})

# Distinct marker + linestyle per a so curves read in grayscale
STYLES = [
    {"marker": "o", "linestyle": "-"},
    {"marker": "s", "linestyle": "--"},
    {"marker": "^", "linestyle": ":"},
    {"marker": "D", "linestyle": "-."},
    {"marker": "v", "linestyle": (0, (3, 1, 1, 1))},
]
# Grayscale ramp: light -> dark as a increases
GRAYS = np.linspace(0.75, 0.0, len(A_VALUES))

fig, ax = plt.subplots(figsize=(3.5, 2.7))

for i, a in enumerate(A_VALUES):
    ax.plot(
        P_VALUES, results[a],
        label=f"a = {a:.1f}",
        color=str(GRAYS[i]),
        markersize=4, linewidth=1.0,
        **STYLES[i % len(STYLES)],
    )

ax.set_xlabel(r"Recruitable fraction $p$")
ax.set_ylabel(r"\% structured filters" if plt.rcParams["text.usetex"]
              else "% structured filters")
ax.set_ylim(0, 105)
ax.grid(True, linewidth=0.3, alpha=0.4)
ax.legend(frameon=False, fontsize=7, title="transfer", ncol=2,
          loc="lower right")

fig.tight_layout(pad=0.4)
fig.savefig("figure4b_filter_quality_curves.pdf", bbox_inches="tight")
fig.savefig("figure4b_filter_quality_curves.png", dpi=600, bbox_inches="tight")
print("\nSaved figure4b_filter_quality_curves.{pdf,png}")