import unittest

from enzrxnpred2.core.default_path import DefaultPath
from enzrxnpred2.data.original_enz_activity_dense_screen_datasource import EnzActivityScreeningDatasource, \
    EnzActivityScreeningDataset


class DenseScreenDatasourceTestCase(unittest.TestCase):
    def test_load_binary_duf(self):
        source = EnzActivityScreeningDatasource(DefaultPath().data_original_dense_screen_processed)
        df = source.load_binary_dataset(EnzActivityScreeningDataset.DUF)
        self.assertIn(EnzActivityScreeningDatasource.sequence_key, df.columns, "A column is missing")
        self.assertIn(EnzActivityScreeningDatasource.substrates_key, df.columns, "A column is missing")
        self.assertIn(EnzActivityScreeningDatasource.activity_key, df.columns, "A column is missing")
        self.assertEqual(len(df), 2737)

    def test_load_binary_esterase(self):
        source = EnzActivityScreeningDatasource(DefaultPath().data_original_dense_screen_processed)
        df = source.load_binary_dataset(EnzActivityScreeningDataset.ESTERASE)
        self.assertIn(EnzActivityScreeningDatasource.sequence_key, df.columns, "A column is missing")
        self.assertIn(EnzActivityScreeningDatasource.substrates_key, df.columns, "A column is missing")
        self.assertIn(EnzActivityScreeningDatasource.activity_key, df.columns, "A column is missing")
        self.assertEqual(len(df), 14016)

    def test_load_binary_gt_acceptors(self):
        source = EnzActivityScreeningDatasource(DefaultPath().data_original_dense_screen_processed)
        df = source.load_binary_dataset(EnzActivityScreeningDataset.GT_ACCEPTORS_CHIRAL)
        self.assertIn(EnzActivityScreeningDatasource.sequence_key, df.columns, "A column is missing")
        self.assertIn(EnzActivityScreeningDatasource.substrates_key, df.columns, "A column is missing")
        self.assertIn(EnzActivityScreeningDatasource.activity_key, df.columns, "A column is missing")
        self.assertEqual(len(df), 4347)

    def test_load_binary_halogenase(self):
        source = EnzActivityScreeningDatasource(DefaultPath().data_original_dense_screen_processed)
        df = source.load_binary_dataset(EnzActivityScreeningDataset.HALOGENASE_NABR)
        self.assertIn(EnzActivityScreeningDatasource.sequence_key, df.columns, "A column is missing")
        self.assertIn(EnzActivityScreeningDatasource.substrates_key, df.columns, "A column is missing")
        self.assertIn(EnzActivityScreeningDatasource.activity_key, df.columns, "A column is missing")
        self.assertEqual(len(df), 2604)

    def test_load_binary_olea(self):
        source = EnzActivityScreeningDatasource(DefaultPath().data_original_dense_screen_processed)
        df = source.load_binary_dataset(EnzActivityScreeningDataset.OLEA)
        self.assertIn(EnzActivityScreeningDatasource.sequence_key, df.columns, "A column is missing")
        self.assertIn(EnzActivityScreeningDatasource.substrates_key, df.columns, "A column is missing")
        self.assertIn(EnzActivityScreeningDatasource.activity_key, df.columns, "A column is missing")
        self.assertEqual(len(df), 1095)

    def test_load_binary_phosphatase(self):
        source = EnzActivityScreeningDatasource(DefaultPath().data_original_dense_screen_processed)
        df = source.load_binary_dataset(EnzActivityScreeningDataset.PHOSPHATASE_CHIRAL)
        self.assertIn(EnzActivityScreeningDatasource.sequence_key, df.columns, "A column is missing")
        self.assertIn(EnzActivityScreeningDatasource.substrates_key, df.columns, "A column is missing")
        self.assertIn(EnzActivityScreeningDatasource.activity_key, df.columns, "A column is missing")
        self.assertEqual(len(df), 35970)

    def test_all_seq(self):
        source = EnzActivityScreeningDatasource(DefaultPath().data_original_dense_screen_processed)
        seqs = source.get_all_dense_screen_seqs()
        self.assertEqual(len(seqs), 694)


if __name__ == '__main__':
    unittest.main()
