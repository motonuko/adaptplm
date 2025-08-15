import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataframe(dataframe_file: Path, out_dir: Path, out_file_prefix: str, val_ratio=0.05, seed=42, header=True):
    df = pd.read_csv(dataframe_file)

    df_train, df_val = train_test_split(df, test_size=val_ratio, random_state=seed)
    combined_df = pd.concat([df_train, df_val])
    duplicates = combined_df[combined_df.duplicated(keep=False)]
    assert duplicates.empty, "Duplicates found"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(out_dir.joinpath(f"{out_file_prefix}_train.csv"), index=False, header=header)
    df_val.to_csv(out_dir.joinpath(f"{out_file_prefix}_val.csv"), index=False, header=header)
    logging.info(f"Data split completed. File has been saved to: {out_dir}")
