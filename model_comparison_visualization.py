"""
Model Comparison Visualization Script
BPRMF, GCNMix, and SASRec comparisons on MovieLens-1M and Grocery & Gourmet Food datasets
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import pi
from typing import Dict, List

# Global plotting configuration
mpl.rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')

MODELS = ['BPRMF', 'GCNMix', 'SASRec']
SPLITS = ['Dev', 'Test']
CUTOFFS = [5, 10, 20, 50]
GROUP_COLORS = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']
MODEL_COLORS = {
    'BPRMF': '#4e79a7',
    'GCNMix': '#f28e2b',
    'SASRec': '#59a14f'
}

MOVIELENS_RESULTS = {
    'Dev': {
        'BPRMF': {
            'HR': [0.4016, 0.5687, 0.7564, 0.9461],
            'NDCG': [0.2695, 0.3230, 0.3705, 0.4090]
        },
        'GCNMix': {
            'HR': [0.4094, 0.5808, 0.7674, 0.9493],
            'NDCG': [0.2751, 0.3299, 0.3771, 0.4138]
        },
        'SASRec': {
            'HR': [0.5203, 0.6538, 0.7974, 0.9528],
            'NDCG': [0.3750, 0.4183, 0.4546, 0.4858]
        }
    },
    'Test': {
        'BPRMF': {
            'HR': [0.3612, 0.5327, 0.7220, 0.9294],
            'NDCG': [0.2425, 0.2978, 0.3458, 0.3873]
        },
        'GCNMix': {
            'HR': [0.3674, 0.5393, 0.7276, 0.9412],
            'NDCG': [0.2511, 0.3066, 0.3543, 0.3973]
        },
        'SASRec': {
            'HR': [0.5233, 0.6604, 0.8072, 0.9544],
            'NDCG': [0.3835, 0.4278, 0.4650, 0.4945]
        }
    }
}

GROCERY_RESULTS = {
    'Dev': {
        'BPRMF': {
            'HR': [0.4150, 0.5147, 0.6186, 0.8026],
            'NDCG': [0.3022, 0.3346, 0.3608, 0.3971]
        },
        'GCNMix': {
            'HR': [0.4650, 0.5843, 0.6906, 0.8614],
            'NDCG': [0.3336, 0.3726, 0.3994, 0.4332]
        },
        'SASRec': {
            'HR': [0.4109, 0.5059, 0.6157, 0.8080],
            'NDCG': [0.3117, 0.3425, 0.3701, 0.4080]
        }
    },
    'Test': {
        'BPRMF': {
            'HR': [0.3460, 0.4545, 0.5638, 0.7667],
            'NDCG': [0.2393, 0.2746, 0.3021, 0.3420]
        },
        'GCNMix': {
            'HR': [0.3908, 0.5186, 0.6319, 0.8326],
            'NDCG': [0.2697, 0.3112, 0.3398, 0.3794]
        },
        'SASRec': {
            'HR': [0.3690, 0.4628, 0.5706, 0.7803],
            'NDCG': [0.2722, 0.3026, 0.3297, 0.3711]
        }
    }
}

DATASETS = {
    'MovieLens-1M': {
        'label': 'MovieLens-1M',
        'prefix': 'movielens1m',
        'footnote': 'Note: Each group of bars corresponds to BPRMF, GCNMix, and SASRec evaluated on MovieLens-1M.',
        'results': MOVIELENS_RESULTS,
        'legacy_files': {}
    },
    'Grocery_and_Gourmet_Food': {
        'label': 'Grocery & Gourmet Food',
        'prefix': 'grocery',
        'footnote': 'Note: Each group of bars corresponds to BPRMF, GCNMix, and SASRec evaluated on Grocery & Gourmet Food.',
        'results': GROCERY_RESULTS,
        'legacy_files': {
            'Test': {
                'grouped_hr': 'model_comparison_HR.pdf',
                'grouped_ndcg': 'model_comparison_NDCG.pdf',
                'radar': 'model_comparison_radar.pdf',
                'curves': 'model_comparison_curves.pdf',
                'heatmap': 'model_comparison_heatmap.pdf'
            }
        }
    }
}


def build_metric_series(dataset_data: Dict[str, Dict[str, Dict[str, List[float]]]], dataset_split: str, metric: str) -> Dict[str, List[float]]:
    series = {}
    data = dataset_data[dataset_split]
    for idx, cutoff in enumerate(CUTOFFS):
        label = f'{metric}@{cutoff}'
        series[label] = [data[model][metric][idx] for model in MODELS]
    return series


def get_extra_paths(dataset_cfg: Dict, split: str, figure_key: str) -> List[str]:
    legacy = dataset_cfg.get('legacy_files', {})
    if split not in legacy:
        return []
    legacy_path = legacy[split].get(figure_key)
    return [legacy_path] if legacy_path else []


def finalize_figure(fig: plt.Figure, primary_path: str, extra_paths: List[str]) -> List[str]:
    saved_paths = []
    fig.savefig(primary_path, dpi=400, bbox_inches='tight')
    saved_paths.append(primary_path)
    for path in extra_paths or []:
        fig.savefig(path, dpi=400, bbox_inches='tight')
        saved_paths.append(path)
    plt.close(fig)
    return saved_paths


def plot_grouped_bar(dataset_cfg: Dict, dataset_split: str, metric: str, extra_paths: List[str]) -> List[str]:
    dataset_label = dataset_cfg['label']
    dataset_data = dataset_cfg['results']
    footnote = dataset_cfg['footnote']
    series = build_metric_series(dataset_data, dataset_split, metric)
    labels = list(series.keys())
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(MODELS))
    width = 0.18

    for i, label in enumerate(labels):
        values = series[label]
        ax.bar(
            x + (i - 1.5) * width,
            values,
            width,
            label=label,
            color=GROUP_COLORS[i],
            alpha=0.85
        )
        for j, v in enumerate(values):
            ax.text(j + (i - 1.5) * width, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ylabel = 'Hit Ratio (HR)' if metric == 'HR' else 'Normalized Discounted Cumulative Gain (NDCG)'
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    title_metric = 'HR Metric Comparison' if metric == 'HR' else 'NDCG Metric Comparison'
    ax.set_title(f'{dataset_split} Split - {title_metric} ({dataset_label})', fontsize=14, fontweight='bold', pad=18)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05 if metric == 'HR' else 0.55)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.5, 0.015, footnote, ha='center', fontsize=9, color='#444444')

    filename = f"{dataset_cfg['prefix']}_{dataset_split.lower()}_{metric.lower()}_grouped.pdf"
    return finalize_figure(fig, filename, extra_paths)


def plot_radar(dataset_cfg: Dict, dataset_split: str, extra_paths: List[str]) -> List[str]:
    dataset_label = dataset_cfg['label']
    dataset_data = dataset_cfg['results']
    categories = [f'HR@{k}' for k in CUTOFFS] + [f'NDCG@{k}' for k in CUTOFFS]
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    for model in MODELS:
        hr_values = dataset_data[dataset_split][model]['HR']
        ndcg_values = dataset_data[dataset_split][model]['NDCG']
        values = hr_values + ndcg_values
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=MODEL_COLORS[model])
        ax.fill(angles, values, alpha=0.15, color=MODEL_COLORS[model])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'{dataset_split} Split Radar View - {dataset_label}', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))

    fig.tight_layout()
    filename = f"{dataset_cfg['prefix']}_{dataset_split.lower()}_radar.pdf"
    return finalize_figure(fig, filename, extra_paths)


def plot_curves(dataset_cfg: Dict, dataset_split: str, extra_paths: List[str]) -> List[str]:
    dataset_label = dataset_cfg['label']
    dataset_data = dataset_cfg['results']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for model in MODELS:
        hr_values = dataset_data[dataset_split][model]['HR']
        ndcg_values = dataset_data[dataset_split][model]['NDCG']
        ax1.plot(CUTOFFS, hr_values, 'o-', linewidth=2.5, markersize=8, label=model, color=MODEL_COLORS[model])
        ax2.plot(CUTOFFS, ndcg_values, 'o-', linewidth=2.5, markersize=8, label=model, color=MODEL_COLORS[model])

    ax1.set_xlabel('Cutoff Value (k)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Hit Ratio (HR)', fontsize=11, fontweight='bold')
    ax1.set_title(f'{dataset_split} Split - HR Curve ({dataset_label})', fontsize=12, fontweight='bold')
    ax1.set_xticks(CUTOFFS)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Cutoff Value (k)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('NDCG', fontsize=11, fontweight='bold')
    ax2.set_title(f'{dataset_split} Split - NDCG Curve ({dataset_label})', fontsize=12, fontweight='bold')
    ax2.set_xticks(CUTOFFS)
    ax2.set_ylim(0, 0.55)
    ax2.grid(True, alpha=0.3)

    ax1.legend(fontsize=10)
    ax2.legend(fontsize=10)

    fig.tight_layout()
    filename = f"{dataset_cfg['prefix']}_{dataset_split.lower()}_curves.pdf"
    return finalize_figure(fig, filename, extra_paths)


def plot_heatmap(dataset_cfg: Dict, dataset_split: str, extra_paths: List[str]) -> List[str]:
    dataset_label = dataset_cfg['label']
    dataset_data = dataset_cfg['results']
    hr_matrix = np.array([
        [dataset_data[dataset_split][model]['HR'][idx] for model in MODELS]
        for idx in range(len(CUTOFFS))
    ])
    ndcg_matrix = np.array([
        [dataset_data[dataset_split][model]['NDCG'][idx] for model in MODELS]
        for idx in range(len(CUTOFFS))
    ])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    X = np.arange(len(MODELS) + 1)
    Y = np.arange(len(CUTOFFS) + 1)

    mesh1 = ax1.pcolormesh(X, Y, hr_matrix, cmap='YlOrRd', vmin=0.32, vmax=1.0, shading='auto')
    mesh1.set_rasterized(True)
    ax1.set_xticks(np.arange(len(MODELS)) + 0.5)
    ax1.set_xticklabels(MODELS, fontsize=11)
    ax1.set_yticks(np.arange(len(CUTOFFS)) + 0.5)
    ax1.set_yticklabels([f'HR@{k}' for k in CUTOFFS], fontsize=11)
    ax1.set_title(f'{dataset_split} Split - HR Heatmap ({dataset_label})', fontsize=12, fontweight='bold')

    thresh_hr = (0.32 + 1.0) / 2
    for i in range(hr_matrix.shape[0]):
        for j in range(hr_matrix.shape[1]):
            val = hr_matrix[i, j]
            color = 'white' if val > thresh_hr else 'black'
            ax1.text(j + 0.5, i + 0.5, f'{val:.4f}', ha='center', va='center', color=color, fontsize=9)

    cb1 = fig.colorbar(mesh1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_label('HR Score')

    mesh2 = ax2.pcolormesh(X, Y, ndcg_matrix, cmap='YlOrRd', vmin=0.24, vmax=0.5, shading='auto')
    mesh2.set_rasterized(True)
    ax2.set_xticks(np.arange(len(MODELS)) + 0.5)
    ax2.set_xticklabels(MODELS, fontsize=11)
    ax2.set_yticks(np.arange(len(CUTOFFS)) + 0.5)
    ax2.set_yticklabels([f'NDCG@{k}' for k in CUTOFFS], fontsize=11)
    ax2.set_title(f'{dataset_split} Split - NDCG Heatmap ({dataset_label})', fontsize=12, fontweight='bold')

    thresh_ndcg = (0.24 + 0.5) / 2
    for i in range(ndcg_matrix.shape[0]):
        for j in range(ndcg_matrix.shape[1]):
            val = ndcg_matrix[i, j]
            color = 'white' if val > thresh_ndcg else 'black'
            ax2.text(j + 0.5, i + 0.5, f'{val:.4f}', ha='center', va='center', color=color, fontsize=9)

    cb2 = fig.colorbar(mesh2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label('NDCG Score')

    fig.tight_layout()
    filename = f"{dataset_cfg['prefix']}_{dataset_split.lower()}_heatmap.pdf"
    return finalize_figure(fig, filename, extra_paths)


def main():
    generated_files = []
    summary_rows = []
    improvement_rows = []

    for dataset_name, dataset_cfg in DATASETS.items():
        dataset_data = dataset_cfg['results']
        for split in SPLITS:
            generated_files.extend(
                plot_grouped_bar(
                    dataset_cfg,
                    split,
                    'HR',
                    get_extra_paths(dataset_cfg, split, 'grouped_hr')
                )
            )
            generated_files.extend(
                plot_grouped_bar(
                    dataset_cfg,
                    split,
                    'NDCG',
                    get_extra_paths(dataset_cfg, split, 'grouped_ndcg')
                )
            )
            generated_files.extend(
                plot_radar(
                    dataset_cfg,
                    split,
                    get_extra_paths(dataset_cfg, split, 'radar')
                )
            )
            generated_files.extend(
                plot_curves(
                    dataset_cfg,
                    split,
                    get_extra_paths(dataset_cfg, split, 'curves')
                )
            )
            generated_files.extend(
                plot_heatmap(
                    dataset_cfg,
                    split,
                    get_extra_paths(dataset_cfg, split, 'heatmap')
                )
            )

            for model in MODELS:
                row = {
                    'Dataset': dataset_cfg['label'],
                    'Split': split,
                    'Model': model
                }
                for idx, cutoff in enumerate(CUTOFFS):
                    row[f'HR@{cutoff}'] = f"{dataset_data[split][model]['HR'][idx]:.4f}"
                    row[f'NDCG@{cutoff}'] = f"{dataset_data[split][model]['NDCG'][idx]:.4f}"
                summary_rows.append(row)

            base_hr = dataset_data[split]['BPRMF']['HR']
            base_ndcg = dataset_data[split]['BPRMF']['NDCG']
            for model in MODELS[1:]:
                row = {
                    'Dataset': dataset_cfg['label'],
                    'Split': split,
                    'Model': model
                }
                for idx, cutoff in enumerate(CUTOFFS):
                    hr_gain = (dataset_data[split][model]['HR'][idx] - base_hr[idx]) / base_hr[idx] * 100
                    ndcg_gain = (dataset_data[split][model]['NDCG'][idx] - base_ndcg[idx]) / base_ndcg[idx] * 100
                    row[f'HR@{cutoff}'] = f'{hr_gain:+.2f}%'
                    row[f'NDCG@{cutoff}'] = f'{ndcg_gain:+.2f}%'
                improvement_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    improvement_df = pd.DataFrame(improvement_rows)

    print('\n' + '=' * 110)
    print('MovieLens-1M & Grocery-and-Gourmet-Food Model Performance Summary (Dev/Test)')
    print('=' * 110)
    print(summary_df.to_string(index=False))

    print('\n' + '=' * 110)
    print('Relative Improvements (vs. BPRMF)')
    print('=' * 110)
    print(improvement_df.to_string(index=False))

    print('\n' + '=' * 110)
    print('Generated figure files:')
    for idx, path in enumerate(generated_files, 1):
        print(f'{idx:2d}. {path}')
    print('=' * 110)


if __name__ == '__main__':
    main()
