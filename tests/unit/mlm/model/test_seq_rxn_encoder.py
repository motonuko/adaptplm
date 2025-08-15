import unittest

import torch
from transformers import EsmModel, BertModel

from adaptplm.core.constants import ESM2_T6_8M_UR50D
from adaptplm.core.default_path import DefaultPath
from adaptplm.mlm.model.seq_rxn_encoder import SeqRxnEncoderForMaskedLM
from adaptplm.mlm.model.seq_rxn_encoder_config import SeqRxnEncoderConfig


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.bert_path = DefaultPath().test_data_dir.joinpath('model', 'rxn_pretrained').as_posix()
        self.esm = ESM2_T6_8M_UR50D
        self.model = SeqRxnEncoderForMaskedLM.from_pretrained_models(
            esm_pretrained=self.esm,
            bert_pretrained=self.bert_path,
            config=SeqRxnEncoderConfig()
        )

    def test_esm_parameters_are_loaded(self):
        esm_model = self.model.esm
        original = EsmModel.from_pretrained(self.esm)

        # NOTE: pooler is not pretrained and initialized randomly
        state_dict1 = {k: v for k, v in esm_model.state_dict().items() if not k.startswith('pooler')}
        state_dict2 = {k: v for k, v in original.state_dict().items() if not k.startswith('pooler')}

        self.assertEqual(set(state_dict1.keys()), set(state_dict2.keys()), "State dict keys do not match.")

        failures = []
        for key in state_dict1.keys():
            if not torch.equal(state_dict1[key], state_dict2[key]):
                failures.append(f"Mismatch in parameter: {key}")
        self.assertTrue(
            len(failures) == 0,
            f"The following parameters did not match:\n" + "\n".join(failures)
        )

    def test_bert_parameters_are_loaded(self):
        bert_model = self.model.bert
        original = BertModel.from_pretrained(self.bert_path, add_pooling_layer=False)

        state_dict1 = bert_model.state_dict()
        state_dict2 = original.state_dict()

        self.assertEqual(set(state_dict1.keys()), set(state_dict2.keys()), "State dict keys do not match.")

        failures = []
        for key in state_dict1.keys():
            if not torch.equal(state_dict1[key], state_dict2[key]):
                failures.append(f"Mismatch in parameter: {key}")
        self.assertTrue(
            len(failures) == 0,
            f"The following parameters did not match:\n" + "\n".join(failures)
        )

    # def test_get_trained_and_untrained_parameters(self):
    #     trained_params, untrained_params, esm_upper_layer_params = self.model.get_grouped_parameters()
    #
    #     for param in trained_params:
    #         param.param_type = 1
    #     for param in untrained_params:
    #         param.param_type = 2
    #     for param in esm_upper_layer_params:
    #         param.param_type = 3
    #
    #     failures = []
    #     for name, param in self.model.named_parameters():
    #         # The order is of if elif is important: child -> parent
    #         if name.startswith('esm.pooler'):
    #             if param.param_type != 2:
    #                 failures.append(name)
    #         elif name.startswith('bert.embeddings.word_embeddings') or name.startswith('cls.predictions'):
    #             # embeddings will be retrained since additional tokens exist
    #             if param.param_type != 2 :
    #                 failures.append(name)
    #         elif name.startswith('esm') or name.startswith('bert') or name.startswith('cls'):
    #             if not param.param_type:
    #                 failures.append(name)
    #         else:
    #             if param.is_trained:
    #                 failures.append(name)
    #     self.assertTrue(len(failures) == 0, f"failed: {failures}")

    def test_all_modules_initialized(self):
        failures = []
        for name, module in self.model.named_modules():
            if hasattr(module, "hf_initialized") and not module.hf_initialized:
                failures.append(name)
        if failures:
            self.fail(f"The following modules are not initialized: {', '.join(failures)}")

    def test_word_embeddings_are_tied(self):
        self.assertIs(
            self.model.bert.embeddings.word_embeddings.weight,
            self.model.cls.predictions.decoder.weight,
            "Word embedding weights are not tied with decoder weights.")


class ResizedModelTestCase(unittest.TestCase):
    def setUp(self):
        self.bert_path = DefaultPath().test_data_dir.joinpath('model', 'rxn_pretrained').as_posix()
        self.esm = ESM2_T6_8M_UR50D
        self.model = SeqRxnEncoderForMaskedLM.from_pretrained_models(
            esm_pretrained=self.esm,
            bert_pretrained=self.bert_path,
            config=SeqRxnEncoderConfig(),
            resized_token_embedding_size=500,
        )

    def test_esm_parameters_are_loaded(self):
        esm_model = self.model.esm
        original = EsmModel.from_pretrained(self.esm)

        # NOTE: pooler is not pretrained and initialized randomly
        state_dict1 = {k: v for k, v in esm_model.state_dict().items() if not k.startswith('pooler')}
        state_dict2 = {k: v for k, v in original.state_dict().items() if not k.startswith('pooler')}

        self.assertEqual(set(state_dict1.keys()), set(state_dict2.keys()), "State dict keys do not match.")

        failures = []
        for key in state_dict1.keys():
            if not torch.equal(state_dict1[key], state_dict2[key]):
                failures.append(f"Mismatch in parameter: {key}")
        self.assertTrue(
            len(failures) == 0,
            f"The following parameters did not match:\n" + "\n".join(failures)
        )

    def test_bert_parameters_are_loaded(self):
        bert_model = self.model.bert
        original = BertModel.from_pretrained(self.bert_path, add_pooling_layer=False)

        state_dict1 = bert_model.state_dict()
        state_dict2 = original.state_dict()

        self.assertEqual(set(state_dict1.keys()), set(state_dict2.keys()), "State dict keys do not match.")

        failures = []
        for key in state_dict1.keys():
            if key.startswith('embeddings.word_embeddings'):
                continue
            if not torch.equal(state_dict1[key], state_dict2[key]):
                failures.append(f"Mismatch in parameter: {key}")
        self.assertTrue(
            len(failures) == 0,
            f"The following parameters did not match:\n" + "\n".join(failures)
        )

    # def test_get_trained_and_untrained_parameters(self):
    #     trained_params, untrained_params = self.model.get_grouped_parameters()
    #     for param in trained_params:
    #         param.is_trained = True
    #     for param in untrained_params:
    #         param.is_trained = False
    #
    #     failures = []
    #     for name, param in self.model.named_parameters():
    #         # This order is important: child -> parent
    #         if name.startswith('esm.pooler'):
    #             if param.is_trained:
    #                 failures.append(name)
    #         elif name.startswith('bert.embeddings.word_embeddings') or name.startswith(
    #                 'cls.predictions') or name.startswith('cls.predictions'):  #  cls.predictions duplicated?
    #             # embeddings will be retrained since additional tokens exist
    #             if param.is_trained:
    #                 failures.append(name)
    #         elif name.startswith('esm') or name.startswith('bert') or name.startswith('cls'):
    #             if not param.is_trained:
    #                 failures.append(name)
    #         else:
    #             if param.is_trained:
    #                 failures.append(name)
    #     self.assertTrue(len(failures) == 0, f"failed: {failures}")

    def test_all_modules_initialized(self):
        failures = []
        for name, module in self.model.named_modules():
            if hasattr(module, "hf_initialized") and not module.hf_initialized:
                failures.append(name)
        if failures:
            self.fail(f"The following modules are not initialized: {', '.join(failures)}")

    def test_word_embeddings_are_tied(self):
        self.assertIs(
            self.model.bert.embeddings.word_embeddings.weight,
            self.model.cls.predictions.decoder.weight,
            "Word embedding weights are not tied with decoder weights.")


if __name__ == '__main__':
    unittest.main()
