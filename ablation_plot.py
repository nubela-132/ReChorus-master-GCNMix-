"""
Plot ablation study results for GCNMix on Grocery & Gourmet Food (Test split).

Experiments:
0) Baseline GCNMix (mix_alpha=0.5, mix_prob=0.3)
1) No IMix (mix_prob=0)
2) Low mix strength (mix_alpha=0.3)
3) High mix strength (mix_alpha=0.7)
4) Always mix (mix_prob=1.0)
5) Enable self_loop
6) Pure MF (n_layers=0, mix_prob=0)

Metrics: HR@{5,10,20,50} and NDCG@{5,10,20,50}
Outputs:
  - ablation_hr_lines.pdf (baseline + MF)
  - ablation_ndcg_lines.pdf (baseline + MF)
  - ablation_hr_bars.pdf (remaining variants)
  - ablation_ndcg_bars.pdf (remaining variants)
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Global style
mpl.rcParams["font.sans-serif"] = ["Times New Roman", "SimHei"]
mpl.rcParams["axes.unicode_minus"] = False
plt.style.use("seaborn-v0_8-whitegrid")

MODEL = "GCNMix"
DATASET = "Grocery_and_Gourmet_Food"
CUTOFFS = [5, 10, 20, 50]

# (label, HR list, NDCG list)
EXPERIMENTS = [
    ("Baseline (mix_prob=0.3)", [0.3908, 0.5186, 0.6319, 0.8326], [0.2697, 0.3112, 0.3398, 0.3794]),
    ("No IMix (prob=0)", [0.3918, 0.5160, 0.6265, 0.8288], [0.2718, 0.3124, 0.3403, 0.3802]),
    ("mix_alpha=0.3", [0.3948, 0.5226, 0.6335, 0.8316], [0.2730, 0.3148, 0.3427, 0.3818]),
    ("mix_alpha=0.7", [0.3911, 0.5176, 0.6301, 0.8302], [0.2705, 0.3117, 0.3401, 0.3796]),
    ("mix_prob=1.0", [0.3949, 0.5246, 0.6376, 0.8369], [0.2714, 0.3137, 0.3422, 0.3815]),
    ("self_loop=1", [0.3907, 0.5171, 0.6311, 0.8258], [0.2702, 0.3113, 0.3401, 0.3785]),
    ("MF (n_layers=0)", [0.3479, 0.4532, 0.5645, 0.7726], [0.2407, 0.2749, 0.3029, 0.3438]),
]

PALETTE = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#af7aa1",
]


def _plot_lines(metric_name: str, filename: str, ylabel: str, series):
    """Line plot across cutoffs (baseline + MF)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (name, values) in enumerate(series):
        color = PALETTE[idx % len(PALETTE)]
        ax.plot(
            CUTOFFS,
            values,
            marker="o",
            linewidth=2,
            markersize=7,
            label=name,
            color=color,
        )
        for x, y in zip(CUTOFFS, values):
            ax.text(
                x,
                y + 0.0015,
                f"{y:.4f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#333333",
            )

    ax.set_xlabel("Cutoff k", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_xticks(CUTOFFS)
    ax.set_xticklabels([f"@{c}" for c in CUTOFFS], fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=3,
        fontsize=9,
        frameon=False,
    )

    fig.suptitle(f"{MODEL} on {DATASET} (Ablation, {metric_name})", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0.05, 0.05, 0.98, 0.96))
    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


def _plot_bars(metric_name: str, filename: str, ylabel: str, values_at_cutoff):
    """Facet bar plots for remaining variants (excluding baseline and MF)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    axes = axes.flatten()

    labels = [name for name, _ in values_at_cutoff]
    x = range(len(labels))

    for ax, cutoff in zip(axes, CUTOFFS):
        vals = [series[CUTOFFS.index(cutoff)] for _, series in values_at_cutoff]
        colors = [PALETTE[(i + 1) % len(PALETTE)] for i in range(len(labels))]
        ax.bar(x, vals, color=colors, width=0.6, edgecolor="#333333", linewidth=0.6)
        for xi, yi in zip(x, vals):
            ax.text(xi, yi + 0.002, f"{yi:.4f}", ha="center", va="bottom", fontsize=8, color="#333333")
        span = max(vals) - min(vals)
        pad = max(0.01, span * 0.2)
        ax.set_ylim(min(vals) - pad, max(vals) + pad)
        ax.set_title(f"@{cutoff}", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.6)

    fig.text(0.02, 0.5, ylabel, va="center", rotation="vertical", fontsize=12, fontweight="bold")
    fig.suptitle(f"{MODEL} on {DATASET} (Ablation, {metric_name})", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0.06, 0.05, 0.98, 0.94))
    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # Separate baseline + MF (lines) vs. other variants (bars)
    baseline = EXPERIMENTS[0]
    mf = EXPERIMENTS[-1]
    others = EXPERIMENTS[1:-1]

    line_hr = [(baseline[0], baseline[1]), (mf[0], mf[1])]
    line_ndcg = [(baseline[0], baseline[2]), (mf[0], mf[2])]

    bar_hr = [(name, hr_list) for name, hr_list, _ in others]
    bar_ndcg = [(name, ndcg_list) for name, _, ndcg_list in others]

    _plot_lines("HR@k", "ablation_hr_lines.pdf", "Hit Ratio (HR@k)", line_hr)
    _plot_lines("NDCG@k", "ablation_ndcg_lines.pdf", "Normalized DCG (NDCG@k)", line_ndcg)
    _plot_bars("HR@k", "ablation_hr_bars.pdf", "Hit Ratio (HR@k)", bar_hr)
    _plot_bars("NDCG@k", "ablation_ndcg_bars.pdf", "Normalized DCG (NDCG@k)", bar_ndcg)
    print(
        "Saved: ablation_hr_lines.pdf, ablation_ndcg_lines.pdf, "
        "ablation_hr_bars.pdf, ablation_ndcg_bars.pdf"
    )
