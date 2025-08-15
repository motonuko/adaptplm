from typing import List

import pandas as pd

from enzrxnpred2.core.default_path import DefaultPath
from enzrxnpred2.data.cd_hit_result_datasource import load_clstr_file_as_dataframe

CLUSTER_HAS_SEQ_FROM = 'cluster_has_seq_from'


def add_cluster_prefix_flags(df: pd.DataFrame, prefixes: List[str]) -> pd.DataFrame:
    prefix_flags = {}
    for prefix in prefixes:
        col_name = f'is_{prefix}'
        df[col_name] = df['sequence_name'].str.startswith(prefix)
        prefix_flags[col_name] = 'any'
    cluster_flags = df.groupby('cluster_id').agg(prefix_flags).rename(
        columns={f'is_{prefix}': f"{CLUSTER_HAS_SEQ_FROM}_{prefix}" for prefix in prefixes}
    )
    df = df.drop(columns=[f'is_{prefix}' for prefix in prefixes])
    df = df.merge(cluster_flags, on='cluster_id')
    return df


def filter_out_overlap_sequences(key: str, filtering_target: str, similarity_threshold: int) -> List[str]:
    prefixes = key.split('__')
    assert filtering_target in prefixes
    clstr_file = DefaultPath().build / 'cdhit' / key / f"{key}_{similarity_threshold}.clstr"
    df = load_clstr_file_as_dataframe(clstr_file)
    df = add_cluster_prefix_flags(df, prefixes)
    df_target = df[df['sequence_name'].str.startswith(filtering_target)]
    print(f"original len: {len(df_target)}")
    for prefix in prefixes:
        if prefix == filtering_target:
            continue
        # keeps only clusters that does not have sequences from other datasets (removes similar sequences).
        df_target = df_target[df_target[f"{CLUSTER_HAS_SEQ_FROM}_{prefix}"] == False]
    print(f"filtered len: {len(df_target)}")
    seq_names = df_target['sequence_name'].tolist()
    seq_names = [name.removeprefix(filtering_target + '_') for name in seq_names]
    return seq_names
