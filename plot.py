"""
Plot the sensitivity of GCNMix to embedding size and layer count on Grocery & Gourmet Food (Test only).

Emb sizes: 32, 48, 64, 96, 128
Layer counts: 1, 2, 3, 4, 5
Metrics: HR@{5,10,20,50} and NDCG@{5,10,20,50}
Outputs:
  - emb_size_hr_test_curves.pdf
  - emb_size_ndcg_test_curves.pdf
  - emb_size_hr_test_facets.pdf
  - emb_size_ndcg_test_facets.pdf
  - n_layers_hr_test_facets.pdf
  - n_layers_ndcg_test_facets.pdf
  - lr_hr_test_facets.pdf
  - lr_ndcg_test_facets.pdf
  - l2_hr_test_facets.pdf
  - l2_ndcg_test_facets.pdf
  - mix_alpha_hr_test_facets.pdf
  - mix_alpha_ndcg_test_facets.pdf
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Global style
mpl.rcParams["font.sans-serif"] = ["Times New Roman", "SimHei"]
mpl.rcParams["axes.unicode_minus"] = False
plt.style.use("seaborn-v0_8-whitegrid")

EMB_SIZES = [32, 48, 64, 96, 128]
MODEL = "GCNMix"
DATASET = "Grocery_and_Gourmet_Food"

# Test results only (emb_size -> metrics by cutoff index)
TEST_HR = {
    32: [0.3892, 0.5144, 0.6271, 0.8268],
    48: [0.3887, 0.5138, 0.6280, 0.8324],
    64: [0.3908, 0.5186, 0.6319, 0.8326],
    96: [0.3982, 0.5194, 0.6345, 0.8349],
    128: [0.3948, 0.5203, 0.6344, 0.8343],
}
TEST_NDCG = {
    32: [0.2686, 0.3095, 0.3379, 0.3774],
    48: [0.2673, 0.3080, 0.3368, 0.3771],
    64: [0.2697, 0.3112, 0.3398, 0.3794],
    96: [0.2758, 0.3153, 0.3443, 0.3837],
    128: [0.2733, 0.3142, 0.3430, 0.3824],
}
# Layer depth sweep (n_layers -> metrics by cutoff index)
N_LAYERS = [1, 2, 3, 4, 5]
TEST_HR_LAYERS = {
    1: [0.3784, 0.4930, 0.6025, 0.8057],
    2: [0.3893, 0.5122, 0.6231, 0.8250],
    3: [0.3908, 0.5186, 0.6319, 0.8326],
    4: [0.3900, 0.5203, 0.6327, 0.8353],
    5: [0.3944, 0.5239, 0.6400, 0.8380],
}
TEST_NDCG_LAYERS = {
    1: [0.2630, 0.3003, 0.3279, 0.3679],
    2: [0.2706, 0.3106, 0.3387, 0.3785],
    3: [0.2697, 0.3112, 0.3398, 0.3794],
    4: [0.2679, 0.3103, 0.3387, 0.3787],
    5: [0.2712, 0.3134, 0.3428, 0.3819],
}
# Learning rate sweep
LRS = [5e-4, 8e-4, 1e-3, 1.5e-3, 2e-3]
TEST_HR_LR = {
    5e-4: [0.3771, 0.4963, 0.6066, 0.8116],
    8e-4: [0.3866, 0.5106, 0.6216, 0.8246],
    1e-3: [0.3908, 0.5186, 0.6319, 0.8326],
    1.5e-3: [0.3960, 0.5232, 0.6395, 0.8360],
    2e-3: [0.3912, 0.5209, 0.6316, 0.8286],
}
TEST_NDCG_LR = {
    5e-4: [0.2610, 0.2997, 0.3275, 0.3680],
    8e-4: [0.2683, 0.3088, 0.3368, 0.3770],
    1e-3: [0.2697, 0.3112, 0.3398, 0.3794],
    1.5e-3: [0.2728, 0.3143, 0.3436, 0.3824],
    2e-3: [0.2704, 0.3127, 0.3407, 0.3795],
}
# L2 sweep
L2S = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
TEST_HR_L2 = {
    1e-7: [0.3769, 0.4989, 0.6131, 0.8211],
    5e-7: [0.3911, 0.5111, 0.6229, 0.8262],
    1e-6: [0.3908, 0.5186, 0.6319, 0.8326],
    5e-6: [0.3679, 0.5030, 0.6248, 0.8283],
    1e-5: [0.3349, 0.4629, 0.5891, 0.7957],
}
TEST_NDCG_L2 = {
    1e-7: [0.2609, 0.3005, 0.3294, 0.3705],
    5e-7: [0.2718, 0.3108, 0.3391, 0.3792],
    1e-6: [0.2697, 0.3112, 0.3398, 0.3794],
    5e-6: [0.2504, 0.2944, 0.3252, 0.3652],
    1e-5: [0.2259, 0.2676, 0.2994, 0.3402],
}
# Mix alpha sweep (mix_prob fixed to 0.3 unless noted in source comments)
MIX_ALPHAS = [0.3, 0.4, 0.5, 0.6, 0.7]
TEST_HR_MIX_ALPHA = {
    0.3: [0.3948, 0.5226, 0.6335, 0.8316],
    0.4: [0.3910, 0.5193, 0.6312, 0.8343],
    0.5: [0.3908, 0.5186, 0.6319, 0.8326],
    0.6: [0.4692, 0.5871, 0.6913, 0.8619],  # reported on dev split
    0.7: [0.3911, 0.5176, 0.6301, 0.8302],
}
TEST_NDCG_MIX_ALPHA = {
    0.3: [0.2730, 0.3148, 0.3427, 0.3818],
    0.4: [0.2697, 0.3114, 0.3396, 0.3796],
    0.5: [0.2697, 0.3112, 0.3398, 0.3794],
    0.6: [0.3365, 0.3748, 0.4011, 0.4348],  # reported on dev split
    0.7: [0.2705, 0.3117, 0.3401, 0.3796],
}

CUTOFFS = [5, 10, 20, 50]
PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#59a14f"]


def _plot_facets(metric_name: str, data_dict, ylabel: str, filename: str, split_label: str, x_values, x_label: str, title_suffix=None):
    """Small-multiple plots per cutoff with tight y-limits to highlight small gaps (single split)."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), sharex=True)
    axes = axes.flatten()
    ncols = 2
    nrows = 2
    for idx, (ax, k) in enumerate(zip(axes, CUTOFFS)):
        vals = [data_dict[x_val][CUTOFFS.index(k)] for x_val in x_values]
        ax.plot(
            x_values,
            vals,
            marker="o",
            linewidth=2,
            markersize=7,
            color=PALETTE[idx % len(PALETTE)],
            label=f"{split_label} {metric_name}@{k}",
        )
        for x, y in zip(x_values, vals):
            ax.text(
                x,
                y + 0.002,
                f"{y:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
            )
        all_vals = vals
        span = max(all_vals) - min(all_vals)
        pad = max(0.005, span * 0.2)
        ax.set_ylim(min(all_vals) - pad, max(all_vals) + pad)
        ax.set_title(f"@{k}", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.5)
        ax.tick_params(labelsize=9)
        ax.set_xticks(x_values)
        ax.set_xticklabels([f"{x:g}" for x in x_values])
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        if idx // ncols == nrows - 1:
            ax.set_xlabel(x_label, fontsize=11, fontweight="bold")
    # common labels
    fig.text(0.02, 0.5, ylabel, va="center", rotation="vertical", fontsize=12, fontweight="bold")
    # single legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    title = title_suffix or split_label
    fig.suptitle(f"{MODEL} on {DATASET} ({title})", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0.06, 0.08, 0.96, 0.94))
    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    _plot_facets(
        "HR",
        TEST_HR,
        "Hit Ratio (HR@k)",
        "emb_size_hr_test_facets.pdf",
        "Test",
        EMB_SIZES,
        "Embedding Size",
    )
    _plot_facets(
        "NDCG",
        TEST_NDCG,
        "Normalized Discounted Cumulative Gain (NDCG@k)",
        "emb_size_ndcg_test_facets.pdf",
        "Test",
        EMB_SIZES,
        "Embedding Size",
    )
    _plot_facets(
        "HR",
        TEST_HR_LAYERS,
        "Hit Ratio (HR@k)",
        "n_layers_hr_test_facets.pdf",
        "Test",
        N_LAYERS,
        "Number of Layers",
    )
    _plot_facets(
        "NDCG",
        TEST_NDCG_LAYERS,
        "Normalized Discounted Cumulative Gain (NDCG@k)",
        "n_layers_ndcg_test_facets.pdf",
        "Test",
        N_LAYERS,
        "Number of Layers",
    )
    _plot_facets(
        "HR",
        TEST_HR_LR,
        "Hit Ratio (HR@k)",
        "lr_hr_test_facets.pdf",
        "Test",
        LRS,
        "Learning Rate",
    )
    _plot_facets(
        "NDCG",
        TEST_NDCG_LR,
        "Normalized Discounted Cumulative Gain (NDCG@k)",
        "lr_ndcg_test_facets.pdf",
        "Test",
        LRS,
        "Learning Rate",
    )
    _plot_facets(
        "HR",
        TEST_HR_L2,
        "Hit Ratio (HR@k)",
        "l2_hr_test_facets.pdf",
        "Test",
        L2S,
        "L2 Regularization",
    )
    _plot_facets(
        "NDCG",
        TEST_NDCG_L2,
        "Normalized Discounted Cumulative Gain (NDCG@k)",
        "l2_ndcg_test_facets.pdf",
        "Test",
        L2S,
        "L2 Regularization",
    )
    _plot_facets(
        "HR",
        TEST_HR_MIX_ALPHA,
        "Hit Ratio (HR@k)",
        "mix_alpha_hr_test_facets.pdf",
        "Test",
        MIX_ALPHAS,
        r"mix\_alpha (mix\_prob=0.3)",
    )
    _plot_facets(
        "NDCG",
        TEST_NDCG_MIX_ALPHA,
        "Normalized Discounted Cumulative Gain (NDCG@k)",
        "mix_alpha_ndcg_test_facets.pdf",
        "Test",
        MIX_ALPHAS,
        r"mix\_alpha (mix\_prob=0.3)",
    )
    print(
        "Saved: emb_size_hr_test_facets.pdf, emb_size_ndcg_test_facets.pdf, "
        "n_layers_hr_test_facets.pdf, n_layers_ndcg_test_facets.pdf, "
        "lr_hr_test_facets.pdf, lr_ndcg_test_facets.pdf, "
        "l2_hr_test_facets.pdf, l2_ndcg_test_facets.pdf, "
        "mix_alpha_hr_test_facets.pdf, mix_alpha_ndcg_test_facets.pdf"
    )
