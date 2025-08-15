import logging
from pathlib import Path

import pandas as pd

from enzrxnpred2.core.default_path import DefaultPath

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_sequences_for_embedding_evaluation(path_enzsrp_full_test: Path, out_dir: Path):
    df = pd.read_csv(path_enzsrp_full_test)
    df = df.drop_duplicates(subset='sequence', keep='first')
    out_dir.mkdir(parents=True, exist_ok=True)
    df['sequence'].to_csv(out_dir / 'enzsrp_test_sequences.txt', index=False, header=False)


if __name__ == '__main__':
    extract_sequences_for_embedding_evaluation(
        path_enzsrp_full_test=DefaultPath().data_dataset_processed / 'enzsrp_full_cleaned' / 'enzsrp_full_cleaned_test.csv',
        out_dir=DefaultPath().data_dataset_processed / 'embedding')
