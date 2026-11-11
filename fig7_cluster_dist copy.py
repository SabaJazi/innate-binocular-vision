import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
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
    fig, axes = plt.subplots(len(selected_ps), 1, figsize=(9.0, 2.0 * len(selected_ps)),
                            sharex=True, sharey=True)
    if len(selected_ps) == 1:
        axes = [axes]

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd",
               "#ff7f0e", "#17becf", "#e377c2"]
    hatch_styles = ["//", "\\\\", "--", "++", "xx", "oo", ".."]
    colors = {p: palette[i % len(palette)] for i, p in enumerate(selected_ps)}
    hatches = {p: hatch_styles[i % len(hatch_styles)] for i, p in enumerate(selected_ps)}
    max_count = 0

    for ax, p in zip(axes, selected_ps):
        sizes = [s for s in results[p] if min_size < s <= 150]
        if not sizes:
            sizes = [min_size + 1]

        mean_size = np.mean(sizes)
        tag = f" ({p_labels[p]})" if p_labels and p in p_labels else ""
        label = f"p = {p:.2f}{tag}"

        counts, bins = np.histogram(sizes, bins=np.linspace(min_size + 1, 150, 30))
        max_count = max(max_count, counts.max() if counts.size else 0)

        ax.hist(sizes, bins=bins, alpha=0.45, color=colors[p], edgecolor=colors[p],
                linewidth=1.1, hatch=hatches[p])
        ax.axvline(mean_size, color=colors[p], linestyle="--", linewidth=1.2)

        p_handle = Line2D([], [], color=colors[p], linewidth=1.5)
        n_handle = Line2D([], [], linestyle="None", marker="s", color=colors[p],
                          markerfacecolor=colors[p], alpha=0.45, markersize=6)
        mean_handle = Line2D([], [], color=colors[p], linestyle="--", linewidth=1.2)
        ax.legend(
            handles=[p_handle, n_handle, mean_handle],
            labels=[f"p = {p:.2f}{tag}", f"n = {len(sizes)}", f"mean = {mean_size:.1f}"],
            frameon=False, fontsize=7, loc="upper right", handlelength=1.8,
            handletextpad=0.6, borderpad=0.3
        )

        ax.set_ylabel("Frequency")
        ax.set_title(label)
        ax.set_xlim(min_size, 150)
        ax.yaxis.set_major_locator(MaxNLocator(4))

    y_max = max_count * 1.2 if max_count > 0 else 1
    for ax in axes:
        ax.set_ylim(0, y_max)

    axes[-1].set_xlabel("Cluster size")
    fig.tight_layout(pad=0.35)
    fig.savefig("figure7_cluster_summary.pdf")
    fig.savefig("figure7_cluster_summary.png", dpi=600)
    print("\nSaved figure7_cluster_summary.{pdf,png}")
    plt.show()


if __name__ == "__main__":
    lgn_width = 256
    r, t, a = 3, 3, 0.2
    p_values = [0.05, 0.08, 0.11]

    # semantic labels for this specific figure (drop if you densify p_values)
    p_labels = {0.05: "low", 0.08: "intermediate", 0.11: "high"}

    results = cluster_analysis_example(lgn_width, r, t, a, p_values)
    plot_cluster_summary(results, p_values, p_labels=p_labels)