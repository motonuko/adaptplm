# test_app.py
import tempfile
import unittest
from pathlib import Path

from click.testing import CliRunner
from dotenv import load_dotenv

load_dotenv()


class TestGreetCommand(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_out_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    # def test_custom_greeting(self):
    #     output_file = self.temp_out_dir / 'temp.fasta'
    #     result = self.runner.invoke(create_uniprot_fasta, ['--output-fasta', output_file])
    #     if result.exit_code == 1:
    #         print(result)
    #     self.assertEqual(result.exit_code, 0)
    #     self.assertEqual(calculate_file_hash(output_file),
    #                      '01c2569e90a8360f1062e90d9b7d8ce9429e6419f41ffa2d85ef23e6f3fc6319')
    #
    # def test_custom_greeting2(self):
    #     output_file = self.temp_out_dir / 'temp.fasta'
    #     result = self.runner.invoke(create_dense_screen_fasta, ['--output-fasta', output_file])
    #     if result.exit_code == 1:
    #         print(result)
    #     self.assertEqual(result.exit_code, 0)
    #
    #     all_seqs = DenseScreenDatasource(DataPath().data_original_dense_screen_processed).get_all_dense_screen_seqs()
    #
    #     with open(output_file) as f:
    #         lines = f.readlines()
    #         seqs_in_fasta = [line for line in lines if not line.startswith('>')]
    #
    #     self.assertEqual(len(all_seqs), len(seqs_in_fasta))
    #     self.assertEqual(calculate_file_hash(output_file),
    #                      '9bdffe95b58032fd4bf8ab40ac2ba596b3a8ea9b4d214f050f853afb22691d2e')


if __name__ == '__main__':
    unittest.main()
