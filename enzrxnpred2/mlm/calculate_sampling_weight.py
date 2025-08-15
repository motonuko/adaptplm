from typing import List

import pandas as pd

from enzrxnpred2.data.cd_hit_result_datasource import CdHitResultDatasource
from enzrxnpred2.extension.bio_ext import calculate_crc64


def calculate_sampling_weight(sequences: List[str], cd_hit_result: CdHitResultDatasource):
    df_cluster = cd_hit_result.get_as_dataframe()

    df_sequences = pd.DataFrame(sequences, columns=['sequence'])
    df_sequences['crc-64'] = df_sequences['sequence'].apply(calculate_crc64)

    assert set(df_sequences['crc-64']) == set(df_cluster[cd_hit_result.label_column_name]), \
        "Expected to be the same. CD-HIT results may differ if extra sequences exist."

    df_sequences_before = df_sequences['sequence'].tolist()
    df_sequences = pd.merge(df_sequences, df_cluster, how='left', left_on='crc-64',
                            right_on=cd_hit_result.label_column_name)
    assert df_sequences['sequence'].tolist() == df_sequences_before, (
        f"Unexpected change after merge! "
        f"Before: {len(df_sequences_before)}, After: {len(df_sequences)}"
    )
    cluster_counts = df_sequences[cd_hit_result.cluster_column_name].value_counts()
    cluster_weights = 1.0 / cluster_counts
    df_sequences['weight'] = df_sequences[cd_hit_result.cluster_column_name].map(cluster_weights)

    return df_sequences['weight'].tolist()
