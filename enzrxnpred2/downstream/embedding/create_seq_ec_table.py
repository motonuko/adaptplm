import logging
from pathlib import Path

from enzrxnpred2.core.default_path import DefaultPath
from enzrxnpred2.data.enz_seq_rxn_datasource import load_enz_seq_rxn_datasource
from enzrxnpred2.domain.ec_number import to_ec_class

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_dataset_for_embedding_evaluation(path_enzsrp_full: Path, out_dir: Path,
                                            ):
    df = load_enz_seq_rxn_datasource(path_enzsrp_full, need_hash=True)
    df = df[['sequence', 'ec_number']]
    df = df.drop_duplicates(keep='first')  # NOTE: one seq can be linked with multiple ec numbers
    df['ec_class'] = df['ec_number'].apply(to_ec_class)
    df.to_csv(out_dir / f"embedding_evaluation_seq_ec.csv", index=False)


if __name__ == '__main__':
    out_dir = DefaultPath().build / 'embed'
    out_dir.mkdir(parents=True, exist_ok=True)
    create_dataset_for_embedding_evaluation(
        path_enzsrp_full=DefaultPath().data_dataset_processed / 'enzsrp_full_cleaned' / 'enzsrp_full_cleaned_test.csv',
        out_dir=out_dir)
