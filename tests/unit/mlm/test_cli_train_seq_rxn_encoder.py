# import unittest
# from unittest.mock import patch
#
# from click.testing import CliRunner
#
# from adaptplm.cli.commands.mlm import train_seq_rxn_encoder_with_mlm_cli
# from adaptplm.domain.constants import ESM2_T6_8M_UR50D
# from adaptplm.utils.default_path import DefaultPath
#
#
# class TestTrainPretrainMaskedLM(unittest.TestCase):
#     def setUp(self):
#         self.runner = CliRunner()
#
#     @patch('adaptplm.mlm.train_seq_rxn_encoder.train_seq_rxn_encoder_with_mlm')  # TODO: Update package name on publishing code
#     def test_default_parameters(self, mock_external_function):
#         mock_external_function.return_value = None
#         config_path = DefaultPath().test_data_dir / 'model_configs' / 'seq_rxn_encoder_config.json'
#         result = self.runner.invoke(train_seq_rxn_encoder_with_mlm_cli, [
#             '--bert-pretrained', DefaultPath().test_data_dir.as_posix(),  # whatever
#             '--esm-pretrained', ESM2_T6_8M_UR50D,
#             '--seq-rxn-encoder-config-file', config_path,
#         ])
#         if result.exit_code != 0:
#             print(result)
#         self.assertEqual(0, result.exit_code)
#
#
# if __name__ == '__main__':
#     unittest.main()
