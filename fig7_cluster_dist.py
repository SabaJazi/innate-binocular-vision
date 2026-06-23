
import matplotlib.pyplot as plt
import numpy as np
from lgn_ibv import (
    LGN,
    compute_cluster_sizes,
    generate_patches,
    perform_ica,
    generate_ident_hash,
)


def cluster_analysis_example(lgn_width, r, t, a, p_values):
    results = {}

    for p in p_values:
        print(f"Running p = {p:.3f}")

        lgn = LGN(width=lgn_width, p=p, r=r, t=t, trans=a, num_layers=2)
        activity = lgn.make_img_mat()[0]  # use one layer

        sizes = compute_cluster_sizes(activity)
        results[p] = sizes

    return results


def plot_cluster_distribution(results, selected_ps):
    fig, ax = plt.subplots(figsize=(3.5, 2.7))

    # plt.figure()

    for p in selected_ps:
        sizes = results[p]
        sizes = [s for s in sizes if s > 5]
        ax.hist(sizes, bins=30, alpha=0.5, label=f"p={p:.2f}")

    ax.set_xlabel("Cluster size")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.set_title("Cluster size distributions")
    plt.show()

    fig.tight_layout(pad=0.4)
    fig.savefig("figure7_cluster_distributions.pdf", bbox_inches="tight")
    fig.savefig("figure7_cluster_distributions.png", dpi=600, bbox_inches="tight")
    print("\nSaved figure7_cluster_distributions.{pdf,png}")

def plot_avg_cluster_size(results):
    p_vals = []
    avg_sizes = []

    for p, sizes in results.items():
        filtered = [s for s in sizes if s > 5]
        if len(filtered) > 0:
            p_vals.append(p)
            avg_sizes.append(np.mean(filtered))

    plt.figure(figsize=(3.5, 2.7))
    plt.plot(p_vals, avg_sizes, marker='o')
    plt.xlabel("Recruitable fraction p")
    plt.ylabel("Mean cluster size")
    plt.title("Mean cluster size vs p")
    plt.tight_layout()
    plt.savefig("figure7_mean_cluster_size.png", dpi=600)
    plt.show()

    
if __name__ == "__main__":
    # parameters (use something simple first)
    lgn_width = 256
    r = 3
    t = 3
    a = 0.2

    # pick p values (low, mid, high)
    p_values = [0.05, 0.08, 0.12]

    # run analysis
    results = cluster_analysis_example(lgn_width, r, t, a, p_values)

    # plot
    plot_cluster_distribution(results, p_values)
    plot_avg_cluster_size(results)



