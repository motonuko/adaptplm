from pathlib import Path

import pandas as pd

from adaptplm.core.default_path import DefaultPath


def load_kcat_df(pkl_file: Path) -> pd.DataFrame:
    return pd.read_pickle(pkl_file)


if __name__ == '__main__':
    df = load_kcat_df(DefaultPath().original_kcat_test_pkl)
    df
