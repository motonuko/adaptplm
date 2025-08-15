import unittest

from tqdm import tqdm

from enzrxnpred2.core.default_path import DefaultPath
from enzrxnpred2.data.original_enz_activity_dense_screen_datasource import EnzActivityScreeningDatasource, \
    datasets_used_in_paper
from enzrxnpred2.downstream.cpi.domain.exp_config import ProteinFeatConfig
from enzrxnpred2.downstream.cpi.ml.enzyme_feature import EsmFeatureConstructor


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.datasource = EnzActivityScreeningDatasource(DefaultPath().data_original_dense_screen_processed)
        self.data_dir = DefaultPath().data_dataset_dir / 'precomputed'

    def test_something(self):
        for dataset in datasets_used_in_paper:
            load_path = self.data_dir / 'dense_screen_esp_embeddings.npz'
            df = self.datasource.load_binary_dataset(dataset)
            config = ProteinFeatConfig.from_dict({
                'model_name': "?",
                'embedding_type': 'precomputed',
                'embedding_params': {'precomputed_embeddings_npy': load_path}
            })
            builder = EsmFeatureConstructor(
                config=config,
                dataset=dataset,
                device='cpu')
            result = {}
            for seq in tqdm(list(df['SEQ'].unique())):
                result[seq] = builder.construct_feature(seq)
