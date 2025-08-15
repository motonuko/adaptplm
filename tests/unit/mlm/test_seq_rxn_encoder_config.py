from unittest import TestCase

from enzrxnpred2.core.default_path import DefaultPath
from enzrxnpred2.mlm.model.seq_rxn_encoder_config import SeqRxnEncoderConfig


class TestSeqRxnEncoderConfig(TestCase):

    def test_from_json_file(self):
        input_file = DefaultPath().test_data_dir / 'model_configs' / 'seq_rxn_encoder_config3.json'
        loaded = SeqRxnEncoderConfig.from_json_file(input_file)
        expected = SeqRxnEncoderConfig(n_additional_layers=4, n_trainable_esm_layers=4)
        self.assertEqual(loaded, expected)
