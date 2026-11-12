"""
Interocular patch correlation across alpha.

A small template from the left-eye map slides horizontally along the same rows of the
right-eye map. At each shift we compute the Pearson correlation between the template and
the right-eye window. The shift with the highest correlation is the preferred disparity.

Null: the same template is compared with the right-eye layer of an independent simulation
(different seed, same alpha), so any peak there is not due to a real left/right relationship.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from lgn_ibv import LGN, calculate_optimal_p

# --- model parameters
ALPHAS = [0.00, 0.25, 0.50, 0.75, 1.00]
R, T = 4, 3
WIDTH = 256
P_SHIFT = 0.01

# --- analysis parameters
PATCH = 16                       # template size (pixels)
MAX_SHIFT = 18                   # search range is -MAX_SHIFT .. +MAX_SHIFT
MIN_ACTIVE = round(0.04 * PATCH ** 2)   # windows with fewer active pixels are skipped
N_SEEDS = 10
N_PATCHES = 60                   # random template locations per simulation

SHIFTS = np.arange(-MAX_SHIFT, MAX_SHIFT + 1)


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return np.nan if denom == 0 else (a * b).sum() / denom


def correlation_curve(left, right, y, x):
    """Correlation between the left-eye template at (y, x) and right-eye windows at each shift.
    Shifts where either patch has too few active pixels are NaN (correlation undefined)."""
    template = left[y:y + PATCH, x:x + PATCH]
    curve = np.full(SHIFTS.shape, np.nan)
    if template.sum() < MIN_ACTIVE:
        return curve
    for i, shift in enumerate(SHIFTS):
        window = right[y:y + PATCH, x + shift:x + shift + PATCH]
        if window.sum() >= MIN_ACTIVE:
            curve[i] = pearson(template, window)
    return curve


def peak(curve):
    """(preferred shift, peak correlation), or None if the curve is entirely undefined."""
    if np.all(np.isnan(curve)):
        return None
    i = np.nanargmax(curve)
    return SHIFTS[i], curve[i]


def simulate(alpha, seed):
    np.random.seed(seed)
    p = calculate_optimal_p(T, R, alpha) + P_SHIFT
    return LGN(width=WIDTH, p=p, r=R, t=T, trans=alpha, num_layers=2).make_img_mat()


def analyze(alpha):
    """Return observed patch results, null peak correlations, and the simulated maps."""
    rng = np.random.default_rng(0)
    results, null_peaks, maps = [], [], []
    for seed in range(N_SEEDS):
        left, right = simulate(alpha, 1000 + seed)
        _, right_independent = simulate(alpha, 5000 + seed)
        maps.append((left, right))

        for _ in range(N_PATCHES):
            y = int(rng.integers(0, WIDTH - PATCH))
            x = int(rng.integers(MAX_SHIFT, WIDTH - PATCH - MAX_SHIFT))
            curve = correlation_curve(left, right, y, x)
            found = peak(curve)
            if found is None:
                continue
            results.append(dict(seed=seed, y=y, x=x, curve=curve,
                                shift=found[0], corr=found[1]))
            null = peak(correlation_curve(left, right_independent, y, x))
            if null is not None:
                null_peaks.append(null[1])

    valid_fraction = len(results) / (N_SEEDS * N_PATCHES)
    return results, np.array(null_peaks), maps, valid_fraction


def plot_example(fig, gs, row, alpha, example, maps):
    """Left: template above its right-eye strip (best window boxed). Right: correlation curve."""
    left, right = maps[example["seed"]]
    y, x, shift = example["y"], example["x"], example["shift"]
    gap = 3

    canvas = np.full((2 * PATCH + gap, PATCH + 2 * MAX_SHIFT), 0.5)
    canvas[:PATCH, MAX_SHIFT:MAX_SHIFT + PATCH] = left[y:y + PATCH, x:x + PATCH]
    canvas[PATCH + gap:, :] = right[y:y + PATCH, x - MAX_SHIFT:x + PATCH + MAX_SHIFT]

    ax = fig.add_subplot(gs[row, 0])
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.add_patch(Rectangle((MAX_SHIFT + shift - 0.5, PATCH + gap - 0.5), PATCH, PATCH,
                           fill=False, edgecolor="red", linewidth=1.5))
    p = calculate_optimal_p(T, R, alpha) + P_SHIFT
    ax.set_title(rf"$\alpha$={alpha:.2f}, p={p:.3f}", fontsize=11)
    ax.set_xlabel(f"top: left-eye template   bottom: right-eye strip\n"
                  f"(red box: best match, shift {shift:+d})", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[row, 1])
    ax.plot(SHIFTS, example["curve"], "k.-", linewidth=1.2)
    ax.plot(shift, example["corr"], "ro")
    ax.annotate(f"peak = {example['corr']:.2f}", (shift, example["corr"]),
                xytext=(6, -14), textcoords="offset points", fontsize=8, color="red")
    ax.axvline(0, color="0.5", linestyle="--", linewidth=0.7)
    ax.set_xlim(-MAX_SHIFT, MAX_SHIFT)
    ax.set_ylim(-0.5, 1.05)
    ax.set_ylabel("Correlation")
    ax.grid(alpha=0.3, linewidth=0.5)
    return ax


def plot_distributions(fig, gs, row, results, null_peaks):
    peaks = [r["corr"] for r in results]
    shifts = [r["shift"] for r in results]

    ax_peak = fig.add_subplot(gs[row, 2])
    bins = np.linspace(-0.5, 1, 31)
    ax_peak.hist(null_peaks, bins=bins, color="0.6", alpha=0.7, density=True, label="null (independent seed)")
    ax_peak.hist(peaks, bins=bins, color="C3", alpha=0.6, density=True, label="observed")
    ax_peak.set_yticks([])

    ax_shift = fig.add_subplot(gs[row, 3])
    ax_shift.hist(shifts, bins=np.arange(-MAX_SHIFT - 0.5, MAX_SHIFT + 1.5), color="C0", density=True)
    ax_shift.set_yticks([])
    return ax_peak, ax_shift


def main():
    fig = plt.figure(figsize=(14.5, 12))
    gs = fig.add_gridspec(len(ALPHAS), 4, width_ratios=[1.5, 1.6, 1.2, 1.2], wspace=0.28, hspace=0.5)
    last = len(ALPHAS) - 1

    for row, alpha in enumerate(ALPHAS):
        results, null_peaks, maps, valid_fraction = analyze(alpha)
        peaks = np.array([r["corr"] for r in results])
        shifts = np.array([r["shift"] for r in results])
        null_95 = np.percentile(null_peaks, 95)
        print(f"alpha={alpha:.2f}  valid={valid_fraction:.2f}  median peak={np.median(peaks):.2f}  "
              f"null 95th={null_95:.2f}  above null={np.mean(peaks > null_95):.2f}  "
              f"shift IQR=[{np.percentile(shifts, 25):+.0f}, {np.percentile(shifts, 75):+.0f}]")

        # example = patch whose peak correlation is closest to the median
        example = results[int(np.argmin(np.abs(peaks - np.median(peaks))))]
        ax_curve = plot_example(fig, gs, row, alpha, example, maps)
        ax_peak, ax_shift = plot_distributions(fig, gs, row, results, null_peaks)

        if row == 0:
            ax_curve.set_title("Example (median peak)")
            ax_peak.set_title("Peak correlation, all patches")
            ax_peak.legend(fontsize=7)
            ax_shift.set_title("Preferred disparity, all patches")
        for ax in (ax_curve, ax_peak, ax_shift):
            if row < last:
                ax.set_xticklabels([])
        if row == last:
            ax_curve.set_xlabel("Horizontal shift (px)")
            ax_peak.set_xlabel("Peak correlation")
            ax_shift.set_xlabel("Preferred disparity (px)")

    fig.suptitle("Interocular patch correlation across α", fontsize=14, y=0.94)
    fig.text(0.5, 0.035,
             f"r = {R}, t = {T}, LGN width = {WIDTH}, template = {PATCH}×{PATCH} px, "
             f"shift range ±{MAX_SHIFT} px, ≥{MIN_ACTIVE} active px per window, "
             f"{N_SEEDS} seeds × {N_PATCHES} random patches per α",
             ha="center", fontsize=9)
    plt.savefig("fig0_patch_pairs_with_stats.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()