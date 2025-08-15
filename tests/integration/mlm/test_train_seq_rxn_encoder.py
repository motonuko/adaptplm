import os
import tempfile
import unittest
from pathlib import Path

from enzrxnpred2.core.constants import ESM2_T6_8M_UR50D
from enzrxnpred2.core.default_path import DefaultPath
from enzrxnpred2.mlm.train_seq_rxn_encoder import train_seq_rxn_encoder_with_mlm, \
    SeqRxnEncoderTrainingConfig
from tests.test_utils.hash import calculate_file_hash


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)
        print(self.temp_dir_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_something(self):
        output_parent_dir = self.temp_dir_path
        train_seq_rxn_encoder_with_mlm(
            SeqRxnEncoderTrainingConfig(
                train_csv=DefaultPath().test_data_dir.joinpath("dataset", 'enzsrp_cleaned_mini.csv'),
                # train_csv=DefaultPath().data_dataset_processed.joinpath("enzsrp_cleaned", 'enzsrp_cleaned_train.csv'),
                val_csv=DefaultPath().test_data_dir.joinpath("dataset", 'enzsrp_cleaned_mini.csv'),
                vocab_file=DefaultPath().test_data_dir / 'dataset' / 'seqrxn_vocab.txt',
                # vocab_file=DefaultPath().data_dataset_processed / 'vocab' / 'enzsrp_cleand_train_vocab.txt',
                out_parent_dir=output_parent_dir,
                mlm_probability=0.15,
                n_training_steps=6,
                batch_size=4,
                save_steps=2,
                eval_steps=3,
                untrained_lr=2e-4,
                trained_lr=5e-5,
                esm_upper_layer_lr=5e-5,  # to check consistency with older models
                seq_rxn_encoder_config_file=DefaultPath().test_data_dir.joinpath("model_configs",
                                                                                 "seq_rxn_encoder_config.json"),
                # seq_rxn_encoder_config_file=DefaultPath().data_exp_configs_dir.joinpath('sample2025Mar',
                #                                                                  "seq_rxn_encoder.json"),
                esm_pretrained=ESM2_T6_8M_UR50D,
                bert_pretrained=DefaultPath().local_exp_rxn_encoder_train.joinpath('241204_152116').as_posix(),
                # bert_pretrained=None,
                bert_model_config_file=None,
                # bert_model_config_file=DefaultPath().data_exp_configs_dir.joinpath('sample2025Mar',
                #                                                                  "bert_enzsrp.json"),
                max_checkpoints=1,
                early_stopping_patience=10,
                early_stopping_min_delta=0.001,
                seed=42,
                mixed_precision='no',
                gradient_accumulation_steps=1,
                randomize_rxn_smiles=False,
                weighted_sampling=False,
                clstr_for_training_seq=None,
                use_cpu=True,
            ),
            is_debug_mode=True
        )

        found = False
        target_file_name = 'final_model.pt'
        for root, dirs, files in os.walk(output_parent_dir):
            if target_file_name in files:
                self.assertEqual('103557f063afaa08a8b886eccb8b34f492f2451492c014bfa3f97f75600c6381',
                                 calculate_file_hash(Path(root, target_file_name))),
                found = True
        if not found:
            self.fail(f"Target file not found.")

    def test_something2(self):
        output_parent_dir = self.temp_dir_path
        train_seq_rxn_encoder_with_mlm(
            SeqRxnEncoderTrainingConfig(
                train_csv=DefaultPath().test_data_dir.joinpath("dataset", 'enzsrp_cleaned_mini.csv'),
                val_csv=DefaultPath().test_data_dir.joinpath("dataset", 'enzsrp_cleaned_mini.csv'),
                vocab_file=DefaultPath().test_data_dir / 'dataset' / 'seqrxn_vocab.txt',
                out_parent_dir=output_parent_dir,
                mlm_probability=0.15,
                n_training_steps=6,
                batch_size=4,
                save_steps=2,
                eval_steps=3,
                untrained_lr=2e-4,
                trained_lr=5e-5,
                esm_upper_layer_lr=5e-5,  # to check consistency with older models
                seq_rxn_encoder_config_file=DefaultPath().test_data_dir.joinpath("model_configs",
                                                                                 "seq_rxn_encoder_config2.json"),
                esm_pretrained=ESM2_T6_8M_UR50D,
                # TODO: bert_pretrained -> should be replaced with shared model.
                bert_pretrained=DefaultPath().local_exp_rxn_encoder_train.joinpath('241204_152116').as_posix(),
                bert_model_config_file=None,
                max_checkpoints=1,
                early_stopping_patience=10,
                early_stopping_min_delta=0.001,
                seed=42,
                mixed_precision='no',
                gradient_accumulation_steps=1,
                randomize_rxn_smiles=False,
                weighted_sampling=False,
                clstr_for_training_seq=None,
                use_cpu=True,
            ),
            is_debug_mode=True
        )

        found = False
        target_file_name = 'final_model.pt'
        for root, dirs, files in os.walk(output_parent_dir):
            if target_file_name in files:
                self.assertEqual('dea962a7272c0b9568feaf9e49d1df5648ca4382deab2f31aa212b0a50dbb382',
                                 calculate_file_hash(Path(root, target_file_name))),
                found = True
        if not found:
            self.fail(f"Target file not found.")

    def test_something_bert_scratch(self):
        output_parent_dir = self.temp_dir_path
        train_seq_rxn_encoder_with_mlm(
            SeqRxnEncoderTrainingConfig(
                train_csv=DefaultPath().test_data_dir.joinpath("dataset", 'enzsrp_cleaned_mini.csv'),
                val_csv=DefaultPath().test_data_dir.joinpath("dataset", 'enzsrp_cleaned_mini.csv'),
                vocab_file=DefaultPath().test_data_dir / 'dataset' / 'seqrxn_vocab.txt',
                out_parent_dir=output_parent_dir,
                mlm_probability=0.15,
                n_training_steps=6,
                batch_size=4,
                save_steps=2,
                eval_steps=3,
                untrained_lr=2e-4,
                trained_lr=5e-5,
                esm_upper_layer_lr=5e-5,  # to check consistency with older models
                seq_rxn_encoder_config_file=DefaultPath().test_data_dir.joinpath("model_configs",
                                                                                 "seq_rxn_encoder_config.json"),
                esm_pretrained=ESM2_T6_8M_UR50D,
                bert_pretrained=None,
                bert_model_config_file=DefaultPath().local_exp_rxn_encoder_train.joinpath('241204_152116',
                                                                                          'config.json'),
                max_checkpoints=1,
                early_stopping_patience=10,
                early_stopping_min_delta=0.001,
                seed=42,
                mixed_precision='no',
                gradient_accumulation_steps=1,
                randomize_rxn_smiles=False,
                weighted_sampling=False,
                clstr_for_training_seq=None,
                use_cpu=True,
            ),
            is_debug_mode=True
        )

        found = False
        target_file_name = 'final_model.pt'
        for root, dirs, files in os.walk(output_parent_dir):
            if target_file_name in files:
                self.assertEqual('dd04815af99b3a120334dd5561b399d1ee8b7cb1bd63a14c42a685d6deb44158',
                                 calculate_file_hash(Path(root, target_file_name))),
                found = True
        if not found:
            self.fail(f"Target file not found.")

    def test_something_bert_scratch2(self):
        output_parent_dir = self.temp_dir_path
        train_seq_rxn_encoder_with_mlm(
            SeqRxnEncoderTrainingConfig(
                train_csv=DefaultPath().test_data_dir.joinpath("dataset", 'enzsrp_cleaned_mini.csv'),
                val_csv=DefaultPath().test_data_dir.joinpath("dataset", 'enzsrp_cleaned_mini.csv'),
                vocab_file=DefaultPath().test_data_dir / 'dataset' / 'seqrxn_vocab.txt',
                out_parent_dir=output_parent_dir,
                mlm_probability=0.15,
                n_training_steps=6,
                batch_size=4,
                save_steps=2,
                eval_steps=3,
                untrained_lr=2e-4,
                trained_lr=5e-5,
                esm_upper_layer_lr=5e-5,  # to check consistency with older models
                seq_rxn_encoder_config_file=DefaultPath().test_data_dir.joinpath("model_configs",
                                                                                 "seq_rxn_encoder_config3.json"),
                esm_pretrained=ESM2_T6_8M_UR50D,
                bert_pretrained=None,
                bert_model_config_file=DefaultPath().local_exp_rxn_encoder_train.joinpath('241204_152116',
                                                                                          'config.json'),
                max_checkpoints=1,
                early_stopping_patience=10,
                early_stopping_min_delta=0.001,
                seed=42,
                mixed_precision='no',
                gradient_accumulation_steps=1,
                randomize_rxn_smiles=False,
                weighted_sampling=False,
                clstr_for_training_seq=None,
                use_cpu=True,
            ),
            is_debug_mode=True
        )

    def test_something_bert_4(self):
        output_parent_dir = self.temp_dir_path
        train_seq_rxn_encoder_with_mlm(
            SeqRxnEncoderTrainingConfig(
                train_csv=DefaultPath().test_data_dir.joinpath("dataset", 'enzsrp_cleaned_mini.csv'),
                val_csv=DefaultPath().test_data_dir.joinpath("dataset", 'enzsrp_cleaned_mini.csv'),
                vocab_file=DefaultPath().test_data_dir / 'dataset' / 'seqrxn_vocab.txt',
                out_parent_dir=output_parent_dir,
                mlm_probability=0.15,
                n_training_steps=6,
                batch_size=4,
                save_steps=2,
                eval_steps=3,
                untrained_lr=2e-4,
                trained_lr=5e-5,
                esm_upper_layer_lr=5e-5,  # to check consistency with older models
                seq_rxn_encoder_config_file=DefaultPath().test_data_dir.joinpath("model_configs",
                                                                                 "seq_rxn_encoder_config3.json"),
                esm_pretrained=ESM2_T6_8M_UR50D,
                bert_pretrained=DefaultPath().local_exp_rxn_encoder_train.joinpath('241204_152116').as_posix(),
                bert_model_config_file=None,
                max_checkpoints=1,
                early_stopping_patience=10,
                early_stopping_min_delta=0.001,
                seed=42,
                mixed_precision='no',
                gradient_accumulation_steps=1,
                randomize_rxn_smiles=True,
                weighted_sampling=False,
                clstr_for_training_seq=None,
                use_cpu=True,
            ),
            is_debug_mode=True
        )

        found = False
        target_file_name = 'final_model.pt'
        for root, dirs, files in os.walk(output_parent_dir):
            if target_file_name in files:
                self.assertEqual('f0a3167351c09a0b719073d6a415a52e0e447d1474a11d8fa4740c9c0d0a5c27',
                                 calculate_file_hash(Path(root, target_file_name))),
                found = True
        if not found:
            self.fail(f"Target file not found.")
