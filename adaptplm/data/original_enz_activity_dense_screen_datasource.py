from enum import Enum
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
from rdkit import Chem


class EnzActivityScreeningDataset(Enum):
    # See the following link to check the corresponding files
    # https://github.com/samgoldman97/enz-pred/blob/main/make_figs/dataset_summary.py#L36
    DUF = "duf"  # BKACE  # 2,737
    ESTERASE = "esterase"
    GT_ACCEPTORS_CHIRAL = "gt_acceptors_chiral"  # 4,298 (not all combinations）
    HALOGENASE_NABR = "halogenase_NaBr"  # 2604
    OLEA = "olea"  # thiolase, 1095
    PHOSPHATASE_CHIRAL = "phosphatase_chiral"

    # NOTE: Chiral data is used in ref. [3]
    # AMINOTRANSFERASE = "aminotransferase"  # seems not used in paper, but exists in GitHub repo [2]
    # GT_ACCEPTORS_ACHIRAL = "gt_acceptors_achiral"
    # GT_DONORS_ACHIRAL = "gt_donors_achiral"
    # GT_DONORS_CHIRAL = "gt_donors_chiral"  # Not used. See S1 of ref. [1]
    # HALOGENASE_NACL = "halogenase_NaCl" # Not used. See S1 of ref. [1]
    # NITRILASE = "nitrilase"    # seems not used in paper, but exists in GitHub repo [2]
    # PHOSPHATASE_ACHIRAL = "phosphatase_achiral"

    # Instead of using MSA result, here, we create renewed dataset with using sequences fetched from UniParc .
    # ([1] uses MSA result -> https://github.com/samgoldman97/enzyme-datasets/blob/main/bin/reformat_duf.py#L37 )
    DUF_SEQ_RETRIEVED = "duf_seq_retrieved"

    DUF_FILTERED = "duf_filtered"
    ESTERASE_FILTERED = "esterase_filtered"
    GT_ACCEPTORS_CHIRAL_FILTERED = "gt_acceptors_chiral_filtered"
    HALOGENASE_NABR_FILTERED = "halogenase_NaBr_filtered"
    OLEA_FILTERED = "olea_filtered"
    PHOSPHATASE_CHIRAL_FILTERED = "phosphatase_chiral_filtered"

    # *** REFERENCE ***
    # [1] Goldman, Samuel, Ria Das, Kevin K. Yang, and Connor W. Coley. "Machine learning modeling of family wide
    #     enzyme-substrate specificity screens." PLoS computational biology 18, no. 2 (2022): e1009853.
    # [2] https://github.com/samgoldman97/enzyme-datasets
    # [3] https://github.com/samgoldman97/enz-pred

    @staticmethod
    def from_label(label: str):
        for dataset_name in EnzActivityScreeningDataset:
            if dataset_name.value == label:
                return dataset_name
        raise ValueError(f"No matching ModelParamFreezeStrategy found for label: {label}")

    @property
    def short_name_for_original_files(self):
        if self == EnzActivityScreeningDataset.DUF:
            return "duf"
        elif self == EnzActivityScreeningDataset.ESTERASE:
            return "esterase"
        elif self == EnzActivityScreeningDataset.GT_ACCEPTORS_CHIRAL:
            return "gt"
        elif self == EnzActivityScreeningDataset.HALOGENASE_NABR:
            return "halogenase"
        elif self == EnzActivityScreeningDataset.OLEA:
            return "olea"
        elif self == EnzActivityScreeningDataset.PHOSPHATASE_CHIRAL:
            return "phosphatase"
        raise ValueError()

    @property
    def to_corresponding_not_filtered_dataset(self):
        if self == EnzActivityScreeningDataset.DUF_FILTERED:
            return EnzActivityScreeningDataset.DUF
        elif self == EnzActivityScreeningDataset.ESTERASE_FILTERED:
            return EnzActivityScreeningDataset.ESTERASE
        elif self == EnzActivityScreeningDataset.GT_ACCEPTORS_CHIRAL_FILTERED:
            return EnzActivityScreeningDataset.GT_ACCEPTORS_CHIRAL
        elif self == EnzActivityScreeningDataset.HALOGENASE_NABR_FILTERED:
            return EnzActivityScreeningDataset.HALOGENASE_NABR
        elif self == EnzActivityScreeningDataset.OLEA_FILTERED:
            return EnzActivityScreeningDataset.OLEA
        elif self == EnzActivityScreeningDataset.PHOSPHATASE_CHIRAL_FILTERED:
            return EnzActivityScreeningDataset.PHOSPHATASE_CHIRAL
        raise ValueError()


# except inhibitor
datasets_used_in_paper: Tuple[EnzActivityScreeningDataset, ...] = (
    EnzActivityScreeningDataset.HALOGENASE_NABR,
    EnzActivityScreeningDataset.GT_ACCEPTORS_CHIRAL,
    EnzActivityScreeningDataset.OLEA,
    EnzActivityScreeningDataset.DUF,
    EnzActivityScreeningDataset.PHOSPHATASE_CHIRAL,
    EnzActivityScreeningDataset.ESTERASE
)

datasets_used_in_our_paper: Tuple[EnzActivityScreeningDataset, ...] = (
    EnzActivityScreeningDataset.HALOGENASE_NABR_FILTERED,
    EnzActivityScreeningDataset.OLEA_FILTERED,
    EnzActivityScreeningDataset.DUF_FILTERED,
    EnzActivityScreeningDataset.PHOSPHATASE_CHIRAL_FILTERED,
    EnzActivityScreeningDataset.ESTERASE_FILTERED
)

class EnzActivityScreeningDatasource:
    sequence_key = 'SEQ'
    substrates_key = 'SUBSTRATES'
    activity_key = 'Activity'

    def __init__(self, dense_screen_dir: Path, additional_source_dir: Optional[Path] = None):
        self.dense_screen_dir = dense_screen_dir
        self.additional_source_dir = additional_source_dir

    def _get_path(self, dataset: EnzActivityScreeningDataset):
        if dataset in datasets_used_in_paper:
            return self.dense_screen_dir.joinpath(f"{dataset.value}_binary.csv")
        else:
            return self.additional_source_dir.joinpath(f"{dataset.value}_binary.csv")

    def load_binary_dataset(self, dataset: EnzActivityScreeningDataset):
        df = pd.read_csv(self._get_path(dataset))
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
        # Standardize the name as 'Activity' for easier processing.
        if dataset == EnzActivityScreeningDataset.HALOGENASE_NABR:
            df.rename(columns={'Conversion_NaBr': self.activity_key}, inplace=True)
        if dataset == EnzActivityScreeningDataset.PHOSPHATASE_CHIRAL:
            df.rename(columns={'Conversion': self.activity_key}, inplace=True)
        if dataset == EnzActivityScreeningDataset.ESTERASE:
            df.rename(columns={'activity': self.activity_key}, inplace=True)
        assert df[self.activity_key].isin([0, 1]).all()
        df[self.activity_key] = df[self.activity_key].astype(int)
        df[self.substrates_key] = df[self.substrates_key].apply(Chem.CanonSmiles)
        return df

    # def load_binary_datasets(self, datasets: Union[
    #     List[EnzActivityDenseScreenDataset], Tuple[EnzActivityDenseScreenDataset, ...]] = datasets_used_in_paper):
    #     dfs = []
    #     for dataset in datasets:
    #         df_sub, _ = self.load_binary_dataset(dataset)
    #         df_sub['dataset'] = dataset.value
    #         dfs.append(df_sub)
    #     return pd.concat(dfs, axis=0)

    def get_all_dense_screen_seqs(self) -> List[str]:
        seqs = set()
        for dataset in datasets_used_in_paper:
            df_sub = self.load_binary_dataset(dataset)
            seqs.update(df_sub["SEQ"].tolist())
        seqs = sorted(seqs)  # To make the output content always same
        return seqs

    # NOTE: For regression (not tested)
    # def load_regression_dataset(self, dataset: EnzActivityDenseScreenDataset):
    #     df = pd.read_csv(self.goldman_dir.joinpath(f"{dataset.value}.csv"))
    #     df.drop(columns=["Unnamed: 0"], inplace=True)
    #
    #     if dataset == EnzActivityDenseScreenDataset.HALOGENASE_NABR:
    #         df.rename(columns={'Conversion_NaBr': 'Activity'}, inplace=True)
    #     if dataset == EnzActivityDenseScreenDataset.PHOSPHATASE_CHIRAL:
    #         df.rename(columns={'Conversion': 'Activity'}, inplace=True)
    #     if dataset == EnzActivityDenseScreenDataset.ESTERASE:
    #         df.rename(columns={'activity': 'Activity'}, inplace=True)
    #
    #     # assert df['Activity'].isin([0, 1]).all()
    #     df['Activity'] = df['Activity']
    #
    #     df['SUBSTRATES'] = df['SUBSTRATES'].apply(Chem.CanonSmiles)
    #
    #     return df
