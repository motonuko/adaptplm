import unittest

from adaptplm.core.default_path import DefaultPath
from adaptplm.data.enz_seq_rxn_datasource import load_enz_seq_rxn_datasource


class EnzSeqRxnDatasourceTestCase(unittest.TestCase):
    def setUp(self):
        self.data_path = DefaultPath().data_dataset_raw.joinpath('enzsrp_full.csv')

    def test_load(self):
        df = load_enz_seq_rxn_datasource(self.data_path)
        self.assertTrue(df['rhea_id'].apply(lambda x: isinstance(x, str)).all(), "rhea_id is not str")

    def test_load_with_hash(self):
        df = load_enz_seq_rxn_datasource(self.data_path, need_hash=True)
        self.assertTrue(df['rhea_id'].apply(lambda x: isinstance(x, str)).all(), "rhea_id is not str")
        self.assertIn('seq_crc64', df.columns, "seq_crc64 column is missing")
        self.assertTrue(df['seq_crc64'].notnull().all(), "seq_crc64 has null values")


if __name__ == '__main__':
    unittest.main()
