import itertools
from typing import List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from scipy.stats import ttest_ind

from adaptplm.core.default_path import DefaultPath
from adaptplm.core.package_version import get_package_major_version
from adaptplm.data.original_enz_activity_dense_screen_datasource import EnzActivityScreeningDataset
from adaptplm.viz.cpi_result.boxplot_utils import split_active_site, sort_key


def split_into_roc_and_pr_tables(df):
    df_mean_and_mlm, _ = split_active_site(df)
    df_roc = pd.concat([df_mean_and_mlm])
    df_pr = pd.concat([df_mean_and_mlm])
    df_roc = df_roc.sort_values(by="condition", key=lambda x: x.map(sort_key)).reset_index(drop=True)
    df_pr = df_pr.sort_values(by="condition", key=lambda x: x.map(sort_key)).reset_index(drop=True)
    return df_roc, df_pr


def run_t_test(df, column):
    print()
    print(df['dataset'][0])
    conditions = df['condition'].unique()
    for cond1, cond2 in itertools.combinations(conditions, 2):
        # collect data for each condition and concatenate into one array
        data1 = np.concatenate(df[df['condition'] == cond1][column].values)
        data2 = np.concatenate(df[df['condition'] == cond2][column].values)
        t_stat, p_value = ttest_ind(data1, data2, equal_var=False)
        result_emoji = "✅" if p_value < 0.05 else "❌"
        print(f"{result_emoji} T-test between '{cond1}' and '{cond2}': t = {t_stat:.4f}, p = {p_value:.4e}")

def draw(ax, df, column, y_label, y_lim, title_map, colors):
    # labels = df['label']
    dataset = df['dataset'][0]
    dataset = title_map[dataset]
    box = ax.boxplot(df[column], patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    for median in box['medians']:
        median.set(color='black', linewidth=1)
    ax.set_xticks([])
    # ax.set_xticks(range(len(labels)))
    # ax.set_xticklabels(labels, rotation=45, ha='right')
    # plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    # ax.set_title(f"{dataset}")
    # fofofo = 'IPAexGothic'
    ax.set_title(f"{dataset}", fontdict={'fontsize': 14})
    ax.set_ylabel(y_label)

    all_values = [v for sublist in df[column] for v in sublist]
    unit = 0.1
    # ymin = np.floor(min(all_values) / unit) * unit
    # ymax = np.ceil(max(all_values) / unit) * unit
    ymin = 0.25
    ymax = 0.90
    range = ymax - ymin
    y_tick_interval = 0.02 if range <= 0.1 else 0.05
    ax.set_ylim(ymin, ymax)
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(y_tick_interval))


def reshape_list(flat_list, n_rows, n_columns):
    result = []
    total_size = n_rows * n_columns
    padded_list = flat_list + [None] * (total_size - len(flat_list))

    for i in range(n_rows):
        row = padded_list[i * n_columns: (i + 1) * n_columns]
        result.append(row)

    return result


# output a pdf
def main(drawing_models: List[str]):
    ver = get_package_major_version()
    base_path = DefaultPath().build / 'fig' / ver

    results_roc = []
    results_pr = []
    for dataset in EnzActivityScreeningDataset:
        path = base_path / f"cv_trials_summary_{dataset.value}.pkl"
        if not path.exists():
            continue
        df = pd.read_pickle(path)
        try:
            df_roc, df_pr = split_into_roc_and_pr_tables(df)
            results_roc.append(df_roc)
            results_pr.append(df_pr)
        except Exception as e:
            pass

    n_rows = 2
    n_cols = 3
    dfs = reshape_list(results_pr, n_rows, n_cols)
    columns = ['pr_auc_mean_per_trial'] * 3
    # load_font_ipaexg()
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    # fixed_colors = ['#A6A6A6', '#2ba8ff', '#9467BD', '#FF7F0E']
    fixed_colors = ['#A6A6A6', '#2ba8ff', '#FF7F0E']
    colors = [fixed_colors[0], fixed_colors[1], *[fixed_colors[2]] * 100]
    title_map = {
        'duf': 'BKACE',
        'duf_filtered': 'BKACE',
        'esterase': 'Esterase',
        'esterase_filtered': 'Esterase',
        'gt_acceptors_chiral': 'Glycosyltransferase',
        'gt_acceptors_chiral_filtered': 'Glycosyltransferase',
        'halogenase_NaBr': 'Halogenase',
        'halogenase_NaBr_filtered': 'Halogenase',
        'olea': 'Thiolase',
        'olea_filtered': 'Thiolase',
        'phosphatase_chiral': 'Phosphatase',
        'phosphatase_chiral_filtered': 'Phosphatase',
    }
    for dfs_row, axs_row in zip(dfs, axes):
        for i in range(n_cols):
            df = dfs_row[i]
            if df is not None:
                df = df[df['condition'].isin(['Mean', 'Precomputed', *[f'Masked LM {m}' for m in drawing_models]])]
                draw(axs_row[i], df, columns[i], y_label='PR-AUC', y_lim=(0.2, 0.9),
                     title_map=title_map,
                     colors=colors)
                run_t_test(df, columns[i])
            else:
                axs_row[i].set_visible(False)
    data_categories = ["ESM-1b$_\\mathrm{{MEAN}}$", "ESM-1b$_\\mathrm{{ESP}}$", "ESM-1b$_\\mathrm{{DA}}$"]
    legend_patches = [mpatches.Patch(color=fixed_colors[i], label=data_categories[i]) for i in
                      range(len(data_categories))]
    fig.legend(handles=legend_patches, loc='lower center', ncol=len(data_categories), fontsize=14, frameon=False)
    plt.tight_layout(rect=[0, 0.08, 1, 1], h_pad=3, w_pad=3)

    # plt.tight_layout()
    # plt.show()
    save_dir = DefaultPath().build / 'fig' / ver
    plt.savefig(save_dir / f"cpi_summary_fixed_y.png")
    plt.savefig(save_dir / f"cpi_summary_fixed_y.pdf", format="pdf", dpi=300)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main(['250420_121652'])
