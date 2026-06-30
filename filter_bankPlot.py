"""
plot_filter_bank.py
-------------------
Generate a binocular ICA filter bank from the LGN spontaneous-activity model
and plot the best left-right filter pairs side by side.

This is the "show the filters" figure for the paper. It:
  1. Generates binocular spontaneous activity (left + right LGN layers).
  2. Extracts paired patches and runs FastICA -> binocular filters.
  3. Scores every filter by an orientation-selectivity index (the 2D-Fourier
     energy ratio described in the paper, Methods 2.5).
  4. Selects the top-N filters by that score and plots their left & right
     subfields as adjacent panels, annotated with the score.

USAGE
-----
Put this file in the SAME folder as your LGN model code (the file you pasted,
e.g. `lgn_ibv.py`). Then either:

    python plot_filter_bank.py

or import-free fallback: if the model import fails, a trimmed copy of the LGN
class + patch/ICA helpers is included below so the script still runs
standalone.

Tune the PARAMS block near the bottom to match the paper's biologically
plausible regime (intermediate p ~0.08-0.10, low a ~0.10-0.20).
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Try to reuse your existing model. Fall back to an embedded minimal copy
# so this file runs on its own if the import path differs.
# ---------------------------------------------------------------------------
try:
    # change "lgn_ibv" to whatever you named the pasted file (without .py)
    from lgn_ibv import LGN, perform_ica  # type: ignore
    _HAVE_MODEL = True
except Exception:
    _HAVE_MODEL = False

if not _HAVE_MODEL:
    import random
    from sklearn.decomposition import FastICA
    from sklearn.feature_extraction import image as skimage

    def pixel_distance(x0, y0, x1, y1):
        return np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

    class LGN:
        """Minimal copy of the binocular LGN percolation model."""

        def __init__(self, width=128, p=0.5, r=1.0, t=1, trans=0.0,
                     num_layers=2, make_wave=True, random_seed=0):
            random.seed(random_seed)
            self.width = width
            self.p = p
            self.r = r
            self.t = t
            self.trans = trans
            self.num_layers = num_layers
            if make_wave:
                self.reset_wave()

        def reset_wave(self):
            w = self.width
            self.recruitable = np.random.rand(self.num_layers, w, w) < self.p
            self.tot_recruitable = int(self.recruitable.sum())
            self.tot_recruitable_active = 0
            self.tot_active = 0
            self.active = np.zeros((self.num_layers, w, w), bool)
            self.active_neighbors = np.zeros((self.num_layers, w, w), int)
            self.activated = []
            if self.tot_recruitable > 0:
                while self.fraction_active() < 0.20:
                    self.activate()

        def fraction_active(self):
            if self.tot_recruitable > 0:
                return self.tot_recruitable_active / self.tot_recruitable
            return float("nan")

        def propagate(self):
            while self.activated:
                act_l, act_x, act_y = self.activated.pop()
                self.active[act_l, act_x, act_y] = True
                self.tot_active += 1
                self.tot_recruitable_active += 1
                for l in range(self.num_layers):
                    for x in range(int(act_x - self.r), int(act_x + self.r + 1)):
                        for y in range(int(act_y - self.r), int(act_y + self.r + 1)):
                            if pixel_distance(act_x, act_y, x, y) > self.r:
                                continue
                            xi, yi = x % self.width, y % self.width
                            if l != act_l:
                                if np.random.rand() < self.trans:
                                    self.active_neighbors[l, xi, yi] += 1
                            else:
                                self.active_neighbors[l, xi, yi] += 1
                            already_active = self.active[l, xi, yi]
                            threshold_reached = self.active_neighbors[l, xi, yi] == self.t
                            if threshold_reached and not already_active:
                                if self.recruitable[l, xi, yi]:
                                    self.activated.append([l, xi, yi])
                                else:
                                    self.active[l, xi, yi] = True
                                    self.tot_active += 1

        def activate(self):
            if self.fraction_active() > 0.95:
                return
            while True:
                l = np.random.randint(0, self.num_layers)
                x = np.random.randint(0, self.width)
                y = np.random.randint(0, self.width)
                if self.recruitable[l, x, y] and not self.active[l, x, y]:
                    break
            self.activated.append([l, x, y])
            self.propagate()

        def make_img_mat(self):
            w = self.width
            img_array = np.zeros((self.num_layers, w, w))
            for l in range(self.num_layers):
                for x in range(w - 1):
                    for y in range(w - 1):
                        if self.active[l, x, y]:
                            img_array[l, x, y] = 1.0
            return img_array

    def perform_ica(num_components, patches):
        ica = FastICA(n_components=num_components, random_state=1,
                      max_iter=1_000_000, whiten="unit-variance")
        return ica.fit(patches).components_


# ---------------------------------------------------------------------------
# Patch generation (kept local so the script is self-contained regardless of
# which save/IO helpers exist in your model file).
# ---------------------------------------------------------------------------
def generate_binocular_patches(num_patches, patch_size, lgn_width,
                               lgn_p, lgn_r, lgn_t, lgn_a, random_seed=0):
    """Generate >= num_patches paired left/right binocular patches."""
    from sklearn.feature_extraction import image as skimage
    half = patch_size ** 2
    patch_base = None
    seed = random_seed
    while patch_base is None or patch_base.shape[0] < num_patches:
        lgn = LGN(width=lgn_width, p=lgn_p, r=lgn_r, t=lgn_t,
                  trans=lgn_a, num_layers=2, random_seed=seed)
        seed += 1
        act = lgn.make_img_mat()
        # Extract ALL patches in deterministic raster order (no max_patches).
        # This is what guarantees left patch i and right patch i come from the
        # SAME (x, y) location -> a genuine binocular pair. This matches the
        # original lgn_ibv.generate_patches behaviour.
        pl = skimage.extract_patches_2d(act[0], (patch_size, patch_size))
        pr = skimage.extract_patches_2d(act[1], (patch_size, patch_size))
        fl = pl.reshape(len(pl), -1)
        fr = pr.reshape(len(pr), -1)
        comp = np.concatenate((fl, fr), axis=1)
        # Drop blank patches (zero variance in either eye).
        valid = [i for i in range(len(comp))
                 if comp[i, :half].std() > 0 and comp[i, half:].std() > 0]
        comp = comp[valid]
        # Subsample to keep memory sane, using ONE shared random draw so the
        # left and right halves of every kept row still correspond.
        if comp.shape[0] > num_patches:
            rng = np.random.default_rng(seed)
            pick = rng.choice(comp.shape[0], size=num_patches, replace=False)
            comp = comp[pick]
        patch_base = comp if patch_base is None else np.vstack((patch_base, comp))
    return patch_base[:num_patches]


# ---------------------------------------------------------------------------
# Orientation-selectivity index (paper Methods 2.5)
# ---------------------------------------------------------------------------
def orientation_selectivity(filt2d):
    """
    Energy concentrated along the dominant orientation axis / total energy,
    estimated from the 2D Fourier power spectrum. Returns a value in [0, 1];
    higher = more strongly oriented.
    """
    f = filt2d - filt2d.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
    n = power.shape[0]
    cy, cx = n // 2, n // 2
    ys, xs = np.nonzero(power >= 0)  # all coords
    # angle of each frequency component relative to center
    ang = np.arctan2(ys - cy, xs - cx)
    ang = np.mod(ang, np.pi)  # orientation is mod pi
    weights = power[ys, xs]
    total = weights.sum()
    if total <= 0:
        return 0.0
    # bin energy into orientation bins, take the dominant bin's fraction
    nbins = 18
    bins = np.minimum((ang / np.pi * nbins).astype(int), nbins - 1)
    energy = np.zeros(nbins)
    for b, wgt in zip(bins, weights):
        energy[b] += wgt
    return float(energy.max() / total)


def score_filters(first_eye, second_eye):
    """Combined orientation-selectivity score per binocular filter pair."""
    n = first_eye.shape[0]
    scores = np.zeros(n)
    for i in range(n):
        scores[i] = 0.5 * (orientation_selectivity(first_eye[i]) +
                           orientation_selectivity(second_eye[i]))
    return scores


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_filter_bank(first_eye, second_eye, scores, top_n=12,
                     out_path="filter_bank.png", title=None):
    """
    Plot the top_n binocular filters (by score) as left|right pairs.
    Layout: a grid where each cell is a [Left | Right] pair.
    """
    order = np.argsort(scores)[::-1][:top_n]

    ncols = 3                     # number of PAIRS per row
    nrows = int(np.ceil(top_n / ncols))

    # shared symmetric color scale so left/right are directly comparable
    vmax = max(np.abs(first_eye[order]).max(), np.abs(second_eye[order]).max())
    vmin = -vmax

    fig = plt.figure(figsize=(ncols * 3.4, nrows * 2.6 + (0.6 if title else 0)),
                     constrained_layout=True)

    if title:
        # Dedicated top band for the title so it can't overlap the first row.
        title_band, grid_area = fig.subfigures(
            2, 1, height_ratios=[0.5, nrows * 2.6])
        title_band.suptitle(title, fontsize=13)
        subfigs = grid_area.subfigures(nrows, ncols, wspace=0.08, hspace=0.18)
    else:
        subfigs = fig.subfigures(nrows, ncols, wspace=0.08, hspace=0.18)
    subfigs = np.atleast_2d(subfigs)

    for k, idx in enumerate(order):
        row, col = k // ncols, k % ncols
        sf = subfigs[row, col]

        # visible border + light fill so the pair reads as one unit
        sf.patch.set_edgecolor("0.55")
        sf.patch.set_linewidth(1.4)
        sf.patch.set_facecolor("0.97")

        # ONE header for the whole pair (component # + shared selectivity)
        sf.suptitle(f"Filter #{idx}   ·   selectivity {scores[idx]:.2f}",
                    fontsize=10, y=0.99)

        ax_l, ax_r = sf.subplots(1, 2)
        ax_l.imshow(first_eye[idx], cmap="gray", vmin=vmin, vmax=vmax)
        ax_r.imshow(second_eye[idx], cmap="gray", vmin=vmin, vmax=vmax)
        ax_l.set_title("Left eye", fontsize=8)
        ax_r.set_title("Right eye", fontsize=8)
        for a in (ax_l, ax_r):
            a.set_xticks([]); a.set_yticks([])

    # hide any leftover empty subfigures
    for k in range(top_n, nrows * ncols):
        row, col = k // ncols, k % ncols
        subfigs[row, col].patch.set_alpha(0)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure -> {out_path}")
    return order


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ---- PARAMS: biologically plausible regime from the paper ----
    PARAMS = dict(
        num_patches=120000,   # raise for cleaner filters (slower)
        num_components=30,   # ICA components = candidate filters
        patch_size=32,       # paper uses 32x32
        lgn_width=256,       # grid size (paper sweep used up to 512)
        lgn_p=0.09,          # intermediate p (paper: ~0.08-0.10)
        lgn_r=4,             # propagation radius
        lgn_t=3,             # activation threshold
        lgn_a=0.15,          # low inter-eye transfer (paper: ~0.10-0.20)
    )
    TOP_N = 15               # how many best pairs to show
    OUT = "filter_bank.png"

    print("Generating binocular patches...")
    patches = generate_binocular_patches(
        PARAMS["num_patches"], PARAMS["patch_size"], PARAMS["lgn_width"],
        PARAMS["lgn_p"], PARAMS["lgn_r"], PARAMS["lgn_t"], PARAMS["lgn_a"],
    )

    print("Running ICA...")
    filters = perform_ica(PARAMS["num_components"], patches)

    half = filters.shape[1] // 2
    dim = int(np.sqrt(half))
    first_eye = filters[:, :half].reshape(-1, dim, dim)
    second_eye = filters[:, half:].reshape(-1, dim, dim)

    print("Scoring filters by orientation selectivity...")
    scores = score_filters(first_eye, second_eye)
    print(f"  selectivity: mean={scores.mean():.3f}  max={scores.max():.3f}  "
          f"frac>0.3={(scores > 0.3).mean():.2f}")

    title = (f"Binocular filter bank  (p={PARAMS['lgn_p']}, "
             f"r={PARAMS['lgn_r']}, t={PARAMS['lgn_t']}, a={PARAMS['lgn_a']}, "
             f"num_patches={PARAMS['num_patches']}, num_components={PARAMS['num_components']}) "
             f" patch_size={PARAMS['patch_size']},"
             f" lgn_width={PARAMS['lgn_width']},"
             f"— top {TOP_N} by orientation selectivity")
    plot_filter_bank(first_eye, second_eye, scores,
                     top_n=TOP_N, out_path=OUT, title=title)


if __name__ == "__main__":
    main()