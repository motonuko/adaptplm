import tempfile
from pathlib import Path
from unittest import TestCase

from click.testing import CliRunner
from dotenv import load_dotenv

from adaptplm.cli.commands.preprocess import clean_enzyme_reaction_pair_full_dataset_cli

load_dotenv()

from tests.test_utils.hash import calculate_file_hash


class TestCleanDatasets(TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_out_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()


    def test_clean_enzyme_reaction_pair_dataset(self):
        output_file = self.temp_out_dir / 'test_seq_rxn_pair_cleaned.csv'
        result = self.runner.invoke(clean_enzyme_reaction_pair_full_dataset_cli, ['--output-file', output_file])
        self.assertEqual(result.exit_code, 0, result)
        expected = '43f077b47a83be6d7a564b92c0a3d2b0d306b4247b332515c23a095fe2a31ca1'
        self.assertEqual(calculate_file_hash(output_file), expected)
