# import unittest
# from unittest.mock import patch
#
# from click.testing import CliRunner
#
# from adaptplm.cli.commands.mlm import train_rxn_encoder_cli
# from adaptplm.utils.default_path import DefaultPath
#
#
# class TestTrainPretrainMaskedLM(unittest.TestCase):
#     def setUp(self):
#         self.runner = CliRunner()
#
#     @patch('adaptplm.mlm.train_rxn_encoder.train_rxn_encoder')  # TODO: Update package name on publishing code
#     def test_default_parameters(self, mock_external_function):
#         mock_external_function.return_value = None
#         config_path = DefaultPath().test_data_dir / 'model_configs' / 'bert.json'
#         result = self.runner.invoke(train_rxn_encoder_cli, ['--model-config-file', config_path])
#         if result.exit_code != 0:
#             print(result)
#         self.assertEqual(0, result.exit_code)
#
#
# if __name__ == '__main__':
#     unittest.main()
