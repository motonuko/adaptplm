from pathlib import Path
from typing import List

from adaptplm.core.default_path import DefaultPath
from adaptplm.data.original_enz_activity_dense_screen_datasource import EnzActivityScreeningDatasource, \
    EnzActivityScreeningDataset


def create_filtered_dense_screen_dataset(dense_screen_dir: Path, out_file: Path, dataset: EnzActivityScreeningDataset,
                                         seqs_to_filter: List[str], substrates_to_filter: List[str]):
    datasource = EnzActivityScreeningDatasource(dense_screen_dir)
    df = datasource.load_binary_dataset(dataset)
    df = df[~df['SEQ'].isin(seqs_to_filter)]
    df = df[~df['SUBSTRATES'].isin(substrates_to_filter)]

    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)


if __name__ == '__main__':
    d_dir = DefaultPath().data_original_dense_screen_processed
    o_file = DefaultPath().data_dataset_processed / 'dense_screen' / 'halogenase_filtered.csv'
    data = EnzActivityScreeningDataset.HALOGENASE_NABR
    seqs_to_filter = []
    substrates_to_filter = []
    create_filtered_dense_screen_dataset(d_dir, o_file, data, seqs_to_filter, substrates_to_filter)
