from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import variation

from enzrxnpred2.core.default_path import DefaultPath
from enzrxnpred2.data.original_enz_activity_dense_screen_datasource import EnzActivityScreeningDataset, \
    EnzActivityScreeningDatasource
from enzrxnpred2.extension.bio_ext import calculate_crc64


def load_blast_result_file(file_path: Path) -> pd.DataFrame:
    columns = ["query_id", "subject_id", "identity", "alignment_length", "mismatches", "gap_opens",
               "q_start", "q_end", "s_start", "s_end", "evalue", "bit_score"]
    return pd.read_csv(file_path, sep="\t", names=columns)


plt.figure()


def check_results(dataset, blast_result_file, e_value_threshold=0.0001):
    results_df = load_blast_result_file(blast_result_file)
    results_df = results_df[results_df["evalue"] <= e_value_threshold]

    source = EnzActivityScreeningDatasource(
        dense_screen_dir=DefaultPath().data_original_dense_screen_processed,
        additional_source_dir=DefaultPath().data_dataset_processed / 'cpi')
    df = source.load_binary_dataset(dataset)
    df['hash'] = df['SEQ'].apply(calculate_crc64)
    unique_hashes = df['hash'].tolist()

    filtered_results_df = results_df[results_df['query_id'].isin(unique_hashes)]
    assert len(filtered_results_df) < len(results_df)
    # print(len(filtered_results_df) , len(results_df))

    unique_queries = results_df["query_id"].unique()

    print()
    print(dataset.value)
    avg = len(filtered_results_df) / len(unique_queries)  # How many similar sequences per query (average)
    print(
        f"total: {len(filtered_results_df)}, unique_seq: {len(unique_queries)}, avg_per_seq: {len(filtered_results_df) / len(unique_queries)}")
    unique_b_counts = results_df.groupby('query_id')['subject_id'].nunique()
    var = unique_b_counts.var()
    std = unique_b_counts.std()
    print(f"number of unique B var.: {var}")
    print(f"number of unique B std.: {std}")
    cv = variation(unique_b_counts, ddof=1)

    threshold = 10
    num_below_threshold = (unique_b_counts < threshold).sum()
    total_queries = unique_b_counts.shape[0]
    rate_below_threshold = num_below_threshold / total_queries

    range_val = unique_b_counts.max() - unique_b_counts.min()
    normalized_std = unique_b_counts.std() / range_val
    normalized_var = unique_b_counts.var() / (range_val ** 2)
    print("normalized_std:", normalized_std)
    print("normalized_var:", normalized_var)

    plt.scatter(len(filtered_results_df), cv, color='blue')
    plt.text(len(filtered_results_df) + 0.001, cv + 0.001, dataset.value[:5])  # Move label slightly

    # unique_counts = results_df.groupby('query_id')['subject_id'].nunique()
    # plt.figure(figsize=(8, 6))
    # unique_counts.plot(kind='bar')
    # plt.xlabel('seq')
    # plt.ylabel('n unique')
    # plt.title(dataset.value)
    # plt.xticks(rotation=0)
    # plt.grid(axis='y')
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    base_path = DefaultPath().build.joinpath("blast")
    dataset_blast_result_sets = [
        (EnzActivityScreeningDataset.DUF_FILTERED, base_path / "duf_enzsrp_full_results.tsv"),
        (EnzActivityScreeningDataset.HALOGENASE_NABR_FILTERED, base_path / "halogenase_NaBr_enzsrp_full_results.tsv"),
        (EnzActivityScreeningDataset.PHOSPHATASE_CHIRAL_FILTERED,
         base_path / "phosphatase_chiral_enzsrp_full_results.tsv"),
        (EnzActivityScreeningDataset.OLEA_FILTERED, base_path / "olea_enzsrp_full_results.tsv"),
        (EnzActivityScreeningDataset.ESTERASE_FILTERED, base_path / "esterase_enzsrp_full_results.tsv"),
    ]
    for dataset, blast_result_path in dataset_blast_result_sets:
        check_results(dataset, blast_result_path, e_value_threshold=1)  # should be 0.001?
    plt.show()
