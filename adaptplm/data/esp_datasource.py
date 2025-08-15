from pathlib import Path

import pandas as pd

from adaptplm.core.default_path import DefaultPath
from adaptplm.extension.bio_ext import calculate_crc64


def load_esp_df(pkl_file: Path, need_hash: bool = False) -> pd.DataFrame:
    df = pd.read_pickle(pkl_file)
    if need_hash:
        df['seq_crc64'] = df['Sequence'].apply(calculate_crc64)
    return df


if __name__ == '__main__':
    df = load_esp_df(DefaultPath().original_esp_fine_tuning_pkl)
