import pandas as pd

from enzrxnpred2.extension.bio_ext import calculate_crc64


# TODO: function name
def load_enz_seq_rxn_datasource(enz_seq_rxn_dataset_file, need_hash=False) -> pd.DataFrame:
    df = pd.read_csv(enz_seq_rxn_dataset_file, dtype={'rhea_id': str}, low_memory=False)
    if need_hash:
        df['seq_crc64'] = df['sequence'].apply(calculate_crc64)
    return df
