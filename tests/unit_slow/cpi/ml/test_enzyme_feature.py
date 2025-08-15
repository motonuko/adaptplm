import pickle
import unittest

import numpy as np
from tqdm import tqdm

from adaptplm.core.constants import ESM1B_T33_650M_UR50S
from adaptplm.core.default_path import DefaultPath
from adaptplm.data.original_enz_activity_dense_screen_datasource import EnzActivityScreeningDatasource, datasets_used_in_paper
from adaptplm.downstream.cpi.domain.exp_config import ProteinFeatConfig
from adaptplm.downstream.cpi.ml.enzyme_feature import EsmFeatureConstructor


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.datasource = EnzActivityScreeningDatasource(DefaultPath().data_original_dense_screen_processed)
        self.test_data_dir = DefaultPath().test_data_dir / 'enzpred_protein_feature'

    def _load_data(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def test_something(self):
        for dataset in datasets_used_in_paper:
            expected_data_path = self.test_data_dir / f"embedding_{dataset.short_name_for_original_files}_mean_.pkl"
            expected = self._load_data(expected_data_path)
            df = self.datasource.load_binary_dataset(dataset)
            config = ProteinFeatConfig.from_dict({
                'name': ESM1B_T33_650M_UR50S,
                'pooling_strategy': 'average',
                'pooling_params': {}
            })
            builder = EsmFeatureConstructor(
                config=config,
                dataset=dataset,
                device='cpu')
            result = {}
            for seq in tqdm(list(df['SEQ'].unique())):
                result[seq] = builder.construct_feature(seq)

            if set(result.keys()) != set(expected.keys()):
                return False
            for key in result.keys():
                if not np.allclose(result[key], expected[key], atol=1e-5):
                    self.fail('not close')

    def test_something2(self):
        for dataset in datasets_used_in_paper:
            dists = [3, 12, 20]
            for dist in dists:
                expected_data_path = self.test_data_dir / f"embedding_{dataset.short_name_for_original_files}_hard_{dist}.pkl"
                expected = self._load_data(expected_data_path)
                df = self.datasource.load_binary_dataset(dataset)
                config = ProteinFeatConfig.from_dict({
                    'name': ESM1B_T33_650M_UR50S,
                    'pooling_strategy': 'active_site',
                    'pooling_params': {
                        'distance': dist
                    }
                })
                builder = EsmFeatureConstructor(
                    config=config,
                    dataset=dataset,
                    device='cpu')
                result = {}
                for seq in tqdm(list(df['SEQ'].unique())):
                    result[seq] = builder.construct_feature(seq)

                if set(result.keys()) != set(expected.keys()):
                    return False
                for key in result.keys():
                    if not np.allclose(result[key], expected[key], atol=1e-5):
                        self.fail('not close')
