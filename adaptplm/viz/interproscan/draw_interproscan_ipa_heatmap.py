from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from adaptplm.core.default_path import DefaultPath
from adaptplm.core.package_version import get_package_version, get_package_major_version
from adaptplm.data.load_interproscan_tsv import load_interproscan_result_tsv_as_ipa_binary_feature
from adaptplm.data.original_enz_activity_dense_screen_datasource import EnzActivityScreeningDataset, \
    EnzActivityScreeningDatasource
from adaptplm.extension.bio_ext import calculate_crc64


def draw_interproscan_ipa_heatmap(dataset: EnzActivityScreeningDataset,
                                  screening_datasource: EnzActivityScreeningDatasource,
                                  interproscan_base_path, output_dir: Path):
    screening_df = screening_datasource.load_binary_dataset(dataset)
    target_seq_hashes = screening_df['SEQ'].apply(calculate_crc64).unique().tolist()

    path_screening_result = interproscan_base_path / f"{dataset.to_corresponding_not_filtered_dataset.value}_input.fasta.tsv"
    df_screening = load_interproscan_result_tsv_as_ipa_binary_feature(path_screening_result)
    len_before = len(df_screening)
    df_screening = df_screening[df_screening.index.isin(target_seq_hashes)]
    assert len(df_screening) < len_before
    df_screening = df_screening.loc[:, (df_screening != 0).any(axis=0)]
    screening_ipas = df_screening.columns.tolist()

    path_enzsrp_result = interproscan_base_path / 'enzsrp_full_train_input.fasta_filtered.tsv'
    df_enzsrp = load_interproscan_result_tsv_as_ipa_binary_feature(path_enzsrp_result, screening_ipas)

    counts1 = df_screening.sum(axis=0)
    counts2 = df_enzsrp.sum(axis=0)

    # rel_counts1 = counts1 # / counts1.sum()
    # rel_counts2 = counts2 # / counts2.sum()

    # 共通 IPR に揃える todo 逆？
    common_iprs = counts1.index.intersection(counts2.index)

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

    # 2つのデータセットを結合
    key1 = f"{title_map[dataset.value]} dataset"
    heatmap_df = pd.DataFrame({
        key1: counts1[common_iprs],
        "EnzSRP training set": counts2[common_iprs]
    })
    heatmap_df = heatmap_df.sort_values(by=key1, ascending=False)

    # heatmap_df = heatmap_df[~((heatmap_df.iloc[:, 0] <= 1) & (heatmap_df.iloc[:, 1] < 20))]

    # 行のラベルが多すぎると見づらいので、上位N個だけに絞る（例: 出現の多い20個）
    # top_iprs = heatmap_df.sum(axis=1).nlargest(20).index
    # heatmap_df = heatmap_df.loc[top_iprs]

    def normalize_column(col):
        if col.max() > col.min():
            return (col - col.min()) / (col.max() - col.min())
        else:
            return pd.Series(1, index=col.index)

    normalized_df = heatmap_df.apply(normalize_column, axis=0)

    annot_values = heatmap_df.apply(lambda col: col.map(lambda x: f"{x:.0f}"))

    annot_combined = normalized_df.copy()  # これだと数値の意味が理解しづらい

    for col in normalized_df.columns:
        annot_combined[col] = normalized_df[col].combine(
            heatmap_df[col],
            lambda norm_val, raw_val: f"{norm_val:.3f} ({int(round(raw_val))})"
        )

    ytick_labels = heatmap_df.index.astype(str).tolist()

    # ③ 描画（y軸ラベルを index から明示）
    fig_height = max(1.5, len(ytick_labels) * 0.2)
    plt.figure(figsize=(10, fig_height))

    ax = sns.heatmap(normalized_df,
                     annot=annot_values,
                     fmt='',
                     cmap="viridis",
                     xticklabels=True,  # x軸ラベル非表示
                     yticklabels=True)

    ax.set_yticklabels(heatmap_df.index.tolist(), rotation=0)

    ax.tick_params(axis='y', colors='black')

    # plt.title(f"Number of Sequences Containing Each IPR in the {dataset.value} dataset\nand the EnzSRP training set")
    # plt.ylabel("InterPro Accession")
    plt.xlabel("")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"interproscan_{dataset.value}.pdf", format="pdf")


if __name__ == '__main__':
    path = DefaultPath().data_original_dense_screen_processed
    path2 = DefaultPath().data_dataset_processed / 'cpi'
    source = EnzActivityScreeningDatasource(path, path2)
    output_dir = DefaultPath().build / 'fig' / get_package_major_version() / 'interproscan'

    interproscan_base = DefaultPath().data_dataset_dir / 'interproscan-5.75-106.0'
    for d in [EnzActivityScreeningDataset.DUF_FILTERED, EnzActivityScreeningDataset.ESTERASE_FILTERED,
              EnzActivityScreeningDataset.HALOGENASE_NABR_FILTERED, EnzActivityScreeningDataset.OLEA_FILTERED,
              EnzActivityScreeningDataset.PHOSPHATASE_CHIRAL_FILTERED]:
        draw_interproscan_ipa_heatmap(d, source, interproscan_base, output_dir)
