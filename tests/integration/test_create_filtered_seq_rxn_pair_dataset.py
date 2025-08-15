import tempfile
from pathlib import Path
from unittest import TestCase


class CreateFilteredSeqRxnPairDatasetTest(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_out_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    # def test_create_filtered_seq_rxn_pair_dataset(self):
    #     output_file = self.temp_out_dir / 'enzsrp_cleaned_filtered.csv'
    #     _create_filtered_seq_rxn_pair_dataset(
    #         seq_rxn_pair_file=DataPath().data_dataset_dir.joinpath('processed', "enzsrp_cleaned.csv"),
    #         output_file=output_file,
    #         exclude_seq_hashes=DataPath().build.joinpath('blastp_high_score_seq_list.txt'), )
    #     expected = '070a6222cc657d81d08d9f29cccd59a331d8ac7b2a2d1d4c3dca8f5bb403e3a2'
    #     self.assertEqual(calculate_file_hash(output_file), expected)
