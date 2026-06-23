import matplotlib.pyplot as plt
import numpy as np
from lgn_ibv import (
    LGN,
    compute_cluster_sizes,
    generate_patches,
    perform_ica,
    generate_ident_hash,
)

MIN_SIZE = 5  # cluster-size threshold, defined once


def cluster_analysis_example(lgn_width, r, t, a, p_values):
    results = {}
    for p in p_values:
        print(f"Running p = {p:.3f}")
        lgn = LGN(width=lgn_width, p=p, r=r, t=t, trans=a, num_layers=2)
        activity = lgn.make_img_mat()[0]
        results[p] = compute_cluster_sizes(activity)
    return results


def plot_cluster_summary(results, selected_ps, min_size=MIN_SIZE, p_labels=None):
    fig, (ax_hist, ax_mean) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    # distinct, well-separated hues that stay legible when overlapping
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd",
               "#ff7f0e", "#17becf", "#e377c2"]
    colors = {p: palette[i % len(palette)] for i, p in enumerate(selected_ps)}

    # ---- Panel A: distributions ----
    means = {}
    for p in selected_ps:
        sizes = [s for s in results[p] if s > min_size]
        m = np.mean(sizes)
        means[p] = m

        # legend entry: value + optional semantic label + sample size
        tag = f" ({p_labels[p]})" if p_labels and p in p_labels else ""
        label = f"p = {p:.2f}{tag}, n={len(sizes)}"

        ax_hist.hist(sizes, bins=30, alpha=0.45,
                     color=colors[p], edgecolor=colors[p], linewidth=1.1,
                     label=label)
        # dashed line at this distribution's mean (ties Panel B back to A)
        ax_hist.axvline(m, color=colors[p], linestyle="--",
                        linewidth=1, alpha=0.9)

    ax_hist.set_xlabel(f"Cluster size (> {min_size})")
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title("(A) Cluster size distributions")
    ax_hist.legend(frameon=False, fontsize=8)

    # ---- Panel B: mean vs p ----
    p_vals, avg_sizes = [], []
    for p in sorted(results):
        if p in means:
            p_vals.append(p)
            avg_sizes.append(means[p])

    ax_mean.plot(p_vals, avg_sizes, "-", color="0.5", zorder=1)
    ax_mean.scatter(p_vals, avg_sizes, c=[colors.get(p, "0.3") for p in p_vals],
                    s=40, zorder=2)
    ax_mean.set_xlabel("Recruitable fraction $p$")
    ax_mean.set_ylabel(f"Mean cluster size (> {min_size})")
    ax_mean.set_title("(B) Mean cluster size vs $p$")

    fig.tight_layout(pad=0.6)
    fig.savefig("figure7_cluster_summary.pdf", bbox_inches="tight")
    fig.savefig("figure7_cluster_summary.png", dpi=600, bbox_inches="tight")
    print("\nSaved figure7_cluster_summary.{pdf,png}")
    plt.show()


if __name__ == "__main__":
    lgn_width = 256
    r, t, a = 3, 3, 0.2
    p_values = [0.05, 0.08, 0.12]

    # semantic labels for this specific figure (drop if you densify p_values)
    p_labels = {0.05: "low", 0.08: "intermediate", 0.12: "high"}

    results = cluster_analysis_example(lgn_width, r, t, a, p_values)
    plot_cluster_summary(results, p_values, p_labels=p_labels)