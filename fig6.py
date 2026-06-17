"""
Figure 6: Filter quality vs parameters.

For each (p, a) combination this script:
  1. Generates a set of ICA filters using the LGN-IBV forward path.
  2. Computes an orientation-selectivity index (OSI) for each filter from
     its 2D Fourier power spectrum.
  3. Classifies a filter as "structured" if OSI > OSI_THRESHOLD.
  4. Plots the percentage of structured filters as a p (x) by a (y) heatmap.

Orientation selectivity index
------------------------------
For one filter we take the 2D FFT power spectrum, bin the spectral energy by
angle, find the dominant orientation, and define
    OSI = energy within +/- ANGLE_WINDOW of the dominant axis / total energy.
A high OSI means the filter's energy is concentrated along one orientation
(an oriented, edge-like / Gabor-like receptive field).

A note on the threshold
-----------------------
ICA on binocular LGN activity tends to yield oriented filters across a wide
parameter range, so OSI values cluster fairly high. If every cell of the
heatmap comes out near 100%%, the OSI_THRESHOLD is below the whole OSI
distribution: print the per-filter OSI values, look at their min/mean/max,
and raise OSI_THRESHOLD into that range (or narrow ANGLE_WINDOW) so the
"%% structured" metric actually discriminates between (p, a) conditions.

Run from the same folder as lgn_ibv.py:
    python figure4_filter_quality.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lgn_ibv import (
    LGN,
    generate_patches,
    perform_ica,
    generate_ident_hash,
)

# ---------------------------------------------------------------------------
# Sweep + analysis settings  (edit here)
# ---------------------------------------------------------------------------

P_VALUES   = [0.04, 0.06, 0.08, 0.10]    # x-axis
A_VALUES   = [0.1, 0.2, 0.3, 0.4, 0.5]   # y-axis
R          = 4                           # fixed propagation radius
T          = 3                           # fixed activation threshold
LGN_WIDTH  = 256
PATCH_SIZE = 9
NUM_PATCHES    = 20000                    # patches per (p, a) cell (lower = faster)
NUM_COMPONENTS = 60                      # ICA components per fit (more = smoother %)
OSI_THRESHOLD  = 0.3                     # structured if OSI exceeds this
ANGLE_WINDOW   = 10.0                    # +/- degrees counted as "dominant axis"

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
    # Remove DC so a flat/constant filter doesn't dominate.
    f = filter_2d - filter_2d.mean()
    if np.allclose(f, 0):
        return 0.0

    power = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2

    h, w = power.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    # Angle of each frequency component (orientation is mod 180 degrees)
    angles = np.degrees(np.arctan2(yy - cy, xx - cx)) % 180.0

    # Ignore the DC bin itself
    mask = ~((yy == cy) & (xx == cx))
    angles = angles[mask]
    energy = power[mask]

    total = energy.sum()
    if total <= 0:
        return 0.0

    # Bin energy into 1-degree orientation bins, find the dominant orientation.
    bins = np.zeros(180)
    idx = np.clip(angles.astype(int), 0, 179)
    np.add.at(bins, idx, energy)
    dominant = int(np.argmax(bins))

    # Circular distance (mod 180) of every component to the dominant orientation.
    d = np.abs(angles - dominant)
    d = np.minimum(d, 180.0 - d)
    in_axis = energy[d <= angle_window].sum()

    return in_axis / total


def fraction_structured(p: float, a: float) -> float:
    """
    Generate ICA filters for one (p, a) point and return the percentage whose
    orientation selectivity exceeds OSI_THRESHOLD.
    """
    ident = generate_ident_hash(p, a, R, T)

    # generate_patches writes activity images; send them to a scratch folder.
    scratch = "_fig6_scratch"

    patches, _ = generate_patches(
        NUM_PATCHES, PATCH_SIZE, LGN_WIDTH,
        p, R, T, a, ident, scratch,
    )
    components = perform_ica(NUM_COMPONENTS, patches)

    # Each component is a concatenated [left-eye | right-eye] filter.
    half = components.shape[1] // 2
    dim = int(np.sqrt(half))

    osis = []
    for comp in components:
        left = comp[:half].reshape(dim, dim)
        right = comp[half:].reshape(dim, dim)
        # Score each eye, take the stronger of the two.
        osis.append(max(orientation_selectivity(left),
                        orientation_selectivity(right)))

    osis = np.array(osis)
    return 100.0 * np.mean(osis > OSI_THRESHOLD)


# ---------------------------------------------------------------------------
# Run the sweep
# ---------------------------------------------------------------------------

heatmap = np.zeros((len(A_VALUES), len(P_VALUES)))

for i, a in enumerate(A_VALUES):
    for j, p in enumerate(P_VALUES):
        pct = fraction_structured(p, a)
        heatmap[i, j] = pct
        print(f"p={p:.2f}  a={a:.2f}  ->  {pct:5.1f}% structured")

# ---------------------------------------------------------------------------
# Plot heatmap (IEEE single-column friendly)
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.6,
})

fig, ax = plt.subplots(figsize=(3.5, 2.8))

im = ax.imshow(
    heatmap, origin="lower", aspect="auto", cmap="viridis",
    vmin=0, vmax=100,
)

ax.set_xticks(range(len(P_VALUES)))
ax.set_xticklabels([f"{p:.2f}" for p in P_VALUES])
ax.set_yticks(range(len(A_VALUES)))
ax.set_yticklabels([f"{a:.1f}" for a in A_VALUES])
ax.set_xlabel(r"Recruitable fraction $p$")
ax.set_ylabel(r"Transfer parameter $a$")

# Annotate each cell with its value (readable on light & dark backgrounds)
for i in range(len(A_VALUES)):
    for j in range(len(P_VALUES)):
        val = heatmap[i, j]
        ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                fontsize=7,
                color="white" if val < 55 else "black")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("% structured filters")

fig.tight_layout(pad=0.4)
fig.savefig("figure6_filter_quality.pdf", bbox_inches="tight")
fig.savefig("figure6_filter_quality.png", dpi=600, bbox_inches="tight")
print("\nSaved figure6_filter_quality.{pdf,png}")