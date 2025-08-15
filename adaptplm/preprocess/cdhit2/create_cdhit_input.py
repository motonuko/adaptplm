from enum import Enum
from functools import total_ordering, lru_cache
from pathlib import Path
from typing import List, Optional

import pandas as pd

from adaptplm.data.enz_seq_rxn_datasource import load_enz_seq_rxn_datasource
from adaptplm.data.esp_datasource import load_esp_df
from adaptplm.data.original_enz_activity_dense_screen_datasource import EnzActivityScreeningDatasource, \
    EnzActivityScreeningDataset
from adaptplm.data.turnup_datasource import load_kcat_df
from adaptplm.extension.bio_ext import calculate_crc64

KEY_SEQ_NAME = 'seq_name'
KEY_SEQ = 'sequence'


@total_ordering
class CDHitTargetDataset(Enum):
    ENZSRP_FULL = "enzsrp_full"
    ENZSRP_FULL_TRAIN = "enzsrp_full_train"
    ESP = "esp"
    KCAT = "kcat"
    DUF = "duf"
    ESTERASE = "esterase"
    GT_ACCEPTORS_CHIRAL = "gt_acceptors_chiral"
    HALOGENASE_NABR = "halogenase_NaBr"
    OLEA = "olea"
    PHOSPHATASE_CHIRAL = "phosphatase_chiral"

    def __init__(self, _):
        # give orders for consistent output
        self._order = {
            'ENZSRP_FULL': 2,
            'ENZSRP_FULL_TRAIN': 2,
            'ESP': 3,
            'KCAT': 4,
            'DUF': 5,
            'ESTERASE': 6,
            'GT_ACCEPTORS_CHIRAL': 7,
            'HALOGENASE_NABR': 8,
            'OLEA': 9,
            'PHOSPHATASE_CHIRAL': 10,
        }[self.name]

    @property
    def order(self):
        return self._order

    @property
    def label(self):
        return self.value

    def __lt__(self, other):
        if isinstance(other, CDHitTargetDataset):
            return self.order < other.order
        return NotImplemented

    @classmethod
    def from_string(cls, value: str) -> 'Enum':
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"{value} is not a valid {cls.__name__}")


CPI_CDHIT_DATASETS = [CDHitTargetDataset.DUF, CDHitTargetDataset.ESTERASE, CDHitTargetDataset.GT_ACCEPTORS_CHIRAL,
                      CDHitTargetDataset.HALOGENASE_NABR, CDHitTargetDataset.OLEA,
                      CDHitTargetDataset.PHOSPHATASE_CHIRAL]


def write_dataframe_in_fasta(in_df, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for index, row in in_df.iterrows():
            f.write(f">{row[KEY_SEQ_NAME]}\n")
            f.write(f"{row[KEY_SEQ]}\n")


@lru_cache
def cache_load_esp_df(path):
    return load_esp_df(path)


@lru_cache
def cache_load_turnup_df(path):
    return load_kcat_df(path)


class CDHitInputBuilder:

    def __init__(self,
                 enzsrp_full_path: Optional[Path] = None,
                 enzsrp_full_train_path: Optional[Path] = None,
                 esp_path: Optional[Path] = None, activity_screen_path: Optional[Path] = None,
                 kcat_path: Optional[Path] = None):
        self.enzsrp_full_path = enzsrp_full_path
        self.enzsrp_full_train_path = enzsrp_full_train_path
        self.esp_path = esp_path
        self.activity_screen_path = activity_screen_path
        self.turnup_path = kcat_path

    def load_and_format_dataset(self, dataset: CDHitTargetDataset):
        # returns dataframe(sequence, sequence_name(key+seq_hash))

        if dataset == CDHitTargetDataset.ENZSRP_FULL:
            assert self.enzsrp_full_path is not None
            df = load_enz_seq_rxn_datasource(self.enzsrp_full_path)
            df = df.drop_duplicates(subset=KEY_SEQ, keep='first')
            return df[[KEY_SEQ]]
        elif dataset == CDHitTargetDataset.ENZSRP_FULL_TRAIN:
            assert self.enzsrp_full_train_path is not None
            df = load_enz_seq_rxn_datasource(self.enzsrp_full_train_path)
            df = df.drop_duplicates(subset=KEY_SEQ, keep='first')
            return df[[KEY_SEQ]]
        elif dataset == CDHitTargetDataset.ESP:
            assert self.esp_path is not None
            df = cache_load_esp_df(self.esp_path)
            df = df.rename(columns={'Sequence': KEY_SEQ})
            df = df.drop_duplicates(subset=KEY_SEQ, keep='first')
            return df[[KEY_SEQ]]
        elif dataset == CDHitTargetDataset.KCAT:
            df = cache_load_turnup_df(self.turnup_path)
            df = df.rename(columns={'Sequence': KEY_SEQ})
            df = df.drop_duplicates(subset=KEY_SEQ, keep='first')
            return df[[KEY_SEQ]]
        elif dataset in [CDHitTargetDataset.DUF, CDHitTargetDataset.ESTERASE, CDHitTargetDataset.GT_ACCEPTORS_CHIRAL,
                         CDHitTargetDataset.HALOGENASE_NABR, CDHitTargetDataset.OLEA,
                         CDHitTargetDataset.PHOSPHATASE_CHIRAL]:
            assert self.activity_screen_path is not None
            datasource = EnzActivityScreeningDatasource(self.activity_screen_path)
            if dataset == CDHitTargetDataset.DUF:
                df = datasource.load_binary_dataset(EnzActivityScreeningDataset.DUF)
            elif dataset == CDHitTargetDataset.ESTERASE:
                df = datasource.load_binary_dataset(EnzActivityScreeningDataset.ESTERASE)
            elif dataset == CDHitTargetDataset.GT_ACCEPTORS_CHIRAL:
                df = datasource.load_binary_dataset(EnzActivityScreeningDataset.GT_ACCEPTORS_CHIRAL)
            elif dataset == CDHitTargetDataset.HALOGENASE_NABR:
                df = datasource.load_binary_dataset(EnzActivityScreeningDataset.HALOGENASE_NABR)
            elif dataset == CDHitTargetDataset.OLEA:
                df = datasource.load_binary_dataset(EnzActivityScreeningDataset.OLEA)
            elif dataset == CDHitTargetDataset.PHOSPHATASE_CHIRAL:
                df = datasource.load_binary_dataset(EnzActivityScreeningDataset.PHOSPHATASE_CHIRAL)
            else:
                raise ValueError('undefined')
            df = df.rename(columns={'SEQ': KEY_SEQ})
            df = df.drop_duplicates(subset=KEY_SEQ, keep='first')
            return df[[KEY_SEQ]]
        raise ValueError('Undefined dataset')

    def create_cdhit_input(self, incoming_datasets: List[CDHitTargetDataset], output_parent_dir: Path):
        datasets = list(sorted(incoming_datasets))
        df_list = []
        if len(datasets) == 1:
            df = self.load_and_format_dataset(datasets[0])
            assert df[KEY_SEQ].is_unique, f"Duplicates found in '{KEY_SEQ}' column!"
            df[KEY_SEQ_NAME] = df[KEY_SEQ].apply(calculate_crc64)
            df_list.append(df[[KEY_SEQ, KEY_SEQ_NAME]])
        else:
            for dataset in datasets:
                df = self.load_and_format_dataset(dataset)
                assert df[KEY_SEQ].is_unique, f"Duplicates found in '{KEY_SEQ}' column!"
                df[KEY_SEQ_NAME] = df[KEY_SEQ].apply(lambda x: f"{dataset.value}_{calculate_crc64(x)}")
                df_list.append(df[[KEY_SEQ, KEY_SEQ_NAME]])

        if len(df_list) > 1:
            df_concat = pd.concat(df_list, ignore_index=True)
            key = '__'.join([d.value for d in datasets])
            file_name = f"{key}_input.fasta"
            output_dir = output_parent_dir / key
            write_dataframe_in_fasta(df_concat, output_dir / file_name)
        elif len(df_list) == 1:
            key = datasets[0].value
            file_name = f"{key}_input.fasta"
            output_dir = output_parent_dir / key
            write_dataframe_in_fasta(df, output_dir / file_name)
        else:
            raise ValueError('unexpected')
        # format and save as fasta


def create_sequence_inputs_for_splitting(output_dir: Path, enzsrp_full_path: Path, enzsrp_full_train_path: Path):
    builder = CDHitInputBuilder(enzsrp_full_path=enzsrp_full_path, enzsrp_full_train_path=enzsrp_full_train_path)
    # builder.create_cdhit_input([CDHitTargetDataset.ENZSRP], output_dir)
    builder.create_cdhit_input([CDHitTargetDataset.ENZSRP_FULL], output_dir)


def create_sequence_inputs_for_analysis(enzsrp_full_train_path: Path, esp_path: Path,
                                        activity_screen_path: Path, kcat_path: Path, output_dir: Path):
    builder = CDHitInputBuilder(enzsrp_full_train_path=enzsrp_full_train_path, esp_path=esp_path,
                                activity_screen_path=activity_screen_path, kcat_path=kcat_path)
    # For CPI analysis
    builder.create_cdhit_input([CDHitTargetDataset.ENZSRP_FULL_TRAIN, CDHitTargetDataset.DUF], output_dir)
    builder.create_cdhit_input([CDHitTargetDataset.ENZSRP_FULL_TRAIN, CDHitTargetDataset.ESTERASE], output_dir)
    builder.create_cdhit_input([CDHitTargetDataset.ENZSRP_FULL_TRAIN, CDHitTargetDataset.GT_ACCEPTORS_CHIRAL],
                               output_dir)
    builder.create_cdhit_input([CDHitTargetDataset.ENZSRP_FULL_TRAIN, CDHitTargetDataset.HALOGENASE_NABR], output_dir)
    builder.create_cdhit_input([CDHitTargetDataset.ENZSRP_FULL_TRAIN, CDHitTargetDataset.OLEA], output_dir)
    builder.create_cdhit_input([CDHitTargetDataset.ENZSRP_FULL_TRAIN, CDHitTargetDataset.PHOSPHATASE_CHIRAL],
                               output_dir)

    # For splitting CPI datasets
    for dataset in CPI_CDHIT_DATASETS:
        builder.create_cdhit_input([CDHitTargetDataset.ENZSRP_FULL_TRAIN, CDHitTargetDataset.ESP, dataset], output_dir)

    # For kcat clustering
    builder.create_cdhit_input(
        [CDHitTargetDataset.ENZSRP_FULL_TRAIN, CDHitTargetDataset.ESP, CDHitTargetDataset.KCAT], output_dir)

    # For checking diversity
    # builder.create_cdhit_input([CDHitTargetDataset.ENZSRP_TRAIN, CDHitTargetDataset.ESP], output_dir)
    # builder.create_cdhit_input([CDHitTargetDataset.ENZSRP_FULL_TRAIN, CDHitTargetDataset.ESP], output_dir)

    # For CPI analysis
    builder.create_cdhit_input([CDHitTargetDataset.ENZSRP_FULL_TRAIN], output_dir)
    builder.create_cdhit_input([CDHitTargetDataset.DUF], output_dir)
    builder.create_cdhit_input([CDHitTargetDataset.ESTERASE], output_dir)
    builder.create_cdhit_input([CDHitTargetDataset.GT_ACCEPTORS_CHIRAL], output_dir)
    builder.create_cdhit_input([CDHitTargetDataset.HALOGENASE_NABR], output_dir)
    builder.create_cdhit_input([CDHitTargetDataset.OLEA], output_dir)
    builder.create_cdhit_input([CDHitTargetDataset.PHOSPHATASE_CHIRAL], output_dir)
