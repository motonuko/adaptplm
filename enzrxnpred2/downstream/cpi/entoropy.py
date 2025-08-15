import numpy as np

from enzrxnpred2.core.default_path import DefaultPath
from enzrxnpred2.data.cd_hit_result_datasource import load_clstr_file_as_dataframe
from enzrxnpred2.data.original_enz_activity_dense_screen_datasource import EnzActivityScreeningDataset, \
    EnzActivityScreeningDatasource, datasets_used_in_paper
from enzrxnpred2.extension.bio_ext import calculate_crc64


def entropy(p):
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def row_entropy_score(df):
    entropies = []
    for x, row in df.iterrows():
        values = row.dropna().values
        if len(values) == 0:
            continue
        p = np.mean(values)
        entropies.append(entropy(p))
    return np.mean(entropies)


def row_variance_score(df):
    probs = []
    for x, row in df.iterrows():
        values = row.dropna().values
        if len(values) == 0:
            continue
        probs.append(np.mean(values))
    return np.var(probs)


def col_entropy_score(df):
    entropies = []
    for col in df.columns:
        values = df[col].dropna().values
        if len(values) == 0:
            continue
        p = np.mean(values)
        entropies.append(entropy(p))
    return np.mean(entropies)


def col_variance_score(df):
    probs = []
    for col in df.columns:
        values = df[col].dropna().values
        if len(values) == 0:
            continue
        probs.append(np.mean(values))
    return np.var(probs)


def summarize_predictability(df):
    return {
        'row_entropy': row_entropy_score(df),
        'col_entropy': col_entropy_score(df),
        'row_variance': row_variance_score(df),
        'col_variance': col_variance_score(df)
    }


def get_filtered_hashes(dataset):
    if dataset.value == EnzActivityScreeningDataset.DUF.value:
        filter_path = DefaultPath().build / 'cdhit_mixed_output' / 'enzsrp_mixed_duf_similar_seq_list.txt'
    elif dataset.value == EnzActivityScreeningDataset.ESTERASE.value:
        filter_path = DefaultPath().build / 'cdhit_mixed_output' / 'enzsrp_mixed_esterase_similar_seq_list.txt'
    elif dataset.value == EnzActivityScreeningDataset.GT_ACCEPTORS_CHIRAL.value:
        filter_path = DefaultPath().build / 'cdhit_mixed_output' / 'enzsrp_mixed_gt_acceptors_chiral_similar_seq_list.txt'
    elif dataset.value == EnzActivityScreeningDataset.HALOGENASE_NABR.value:
        filter_path = DefaultPath().build / 'cdhit_mixed_output' / 'enzsrp_mixed_halogenase_NaBr_similar_seq_list.txt'
    elif dataset.value == EnzActivityScreeningDataset.OLEA.value:
        filter_path = DefaultPath().build / 'cdhit_mixed_output' / 'enzsrp_mixed_olea_similar_seq_list.txt'
    elif dataset.value == EnzActivityScreeningDataset.PHOSPHATASE_CHIRAL.value:
        filter_path = DefaultPath().build / 'cdhit_mixed_output' / 'enzsrp_mixed_phosphatase_chiral_similar_seq_list.txt'
    else:
        raise ValueError('not ready')
    with open(filter_path, 'r', encoding='utf-8') as file:
        filter_seq_hashes = [line.strip() for line in file if line.strip()]
    return filter_seq_hashes


def compute_local_label_consistency(dataset: EnzActivityScreeningDataset):
    clstr_file = DefaultPath().build / 'cdhit_activity_screen_output' / f"activity_screen_{dataset.value}.clstr"
    df_clstr = load_clstr_file_as_dataframe(clstr_file)

    datasource = EnzActivityScreeningDatasource(DefaultPath().data_original_dense_screen_processed)
    df_cls = datasource.load_binary_dataset(dataset)
    df_cls['sequence_name'] = df_cls['SEQ'].apply(calculate_crc64)
    filtered_hashes = get_filtered_hashes(dataset)
    df_cls = df_cls[~df_cls['sequence_name'].isin(filtered_hashes)]

    df = df_cls.merge(df_clstr, on='sequence_name')
    assert len(df) == len(df_cls), dataset

    df_wide = df.pivot_table(index='SEQ', columns='SUBSTRATES', values='Activity')
    # result = row_entropy_score(df_wide)
    result = col_entropy_score(df_wide)
    # result = compute_weighted_local_label_consistency_1_only(df)

    # if True:
    #     # 1. Remove X that is only Z=0
    #     x_all_zero = df.groupby('SEQ')['Activity'].sum() == 0
    #     x_to_drop = x_all_zero[x_all_zero].index
    #     df = df[~df['SEQ'].isin(x_to_drop)]
    #
    #     # 2. Remove Y that is only Z=0
    #     y_all_zero = df.groupby('SUBSTRATES')['Activity'].sum() == 0
    #     y_to_drop = y_all_zero[y_all_zero].index
    #     df = df[~df['SUBSTRATES'].isin(y_to_drop)]

    print(dataset, result)
    print(f"#clusters {len(set(df['cluster_id'].tolist()))}, #sequence:  {len(set(df['sequence_name'].tolist()))}")
    print()


if __name__ == '__main__':
    for dat in datasets_used_in_paper:
        compute_local_label_consistency(dat)
