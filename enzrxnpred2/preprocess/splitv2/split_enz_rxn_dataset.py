import logging
from pathlib import Path

from sklearn.model_selection import train_test_split

from enzrxnpred2.data.cd_hit_result_datasource import load_clstr_file_as_dataframe
from enzrxnpred2.data.enz_seq_rxn_datasource import load_enz_seq_rxn_datasource
from enzrxnpred2.extension.bio_ext import calculate_crc64


# Split based on clusters.
# Assuming preprocessing has already done
def split_enz_rxn_dataset(
        enzsrp: Path,
        cd_hit_result: Path,
        output_dir: Path,
        out_file_prefix: str,
        val_ratio=0.05,
        test_ratio=0.05,
        seed=42,
):
    df_enz = load_enz_seq_rxn_datasource(enzsrp)
    df_enz['seq_crc64'] = df_enz['sequence'].apply(calculate_crc64)

    df_cluster = load_clstr_file_as_dataframe(cd_hit_result)
    unique_clusters = df_cluster['cluster_id'].drop_duplicates(keep='first').tolist()

    train_val_clusters, test_clusters = train_test_split(unique_clusters, test_size=test_ratio, random_state=seed)
    val_size_adjusted = val_ratio / (1 - test_ratio)
    train_clusters, val_clusters = train_test_split(train_val_clusters, test_size=val_size_adjusted, random_state=seed)

    train_hashes = df_cluster[df_cluster['cluster_id'].isin(train_clusters)]['sequence_name']
    val_hashes = df_cluster[df_cluster['cluster_id'].isin(val_clusters)]['sequence_name']
    test_hashes = df_cluster[df_cluster['cluster_id'].isin(test_clusters)]['sequence_name']

    df_train = df_enz[df_enz['seq_crc64'].isin(train_hashes)]
    df_val = df_enz[df_enz['seq_crc64'].isin(val_hashes)]
    df_test = df_enz[df_enz['seq_crc64'].isin(test_hashes)]

    output_dir.mkdir(parents=True, exist_ok=True)
    header = True

    assert len(df_train) + len(df_val) + len(df_test) == len(df_enz), (len(df_train) + len(df_val), len(df_enz))

    df_train.to_csv(output_dir.joinpath(f"{out_file_prefix}_train.csv"), index=False, header=header)
    df_val.to_csv(output_dir.joinpath(f"{out_file_prefix}_val.csv"), index=False, header=header)
    df_test.to_csv(output_dir.joinpath(f"{out_file_prefix}_test.csv"), index=False, header=header)
    logging.info(f"Process completed. File has been saved to: {output_dir}")
