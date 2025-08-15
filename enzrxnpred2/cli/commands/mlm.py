from pathlib import Path
from typing import Optional

import click

from adaptplm.core.default_path import DefaultPath
from adaptplm.mlm.construct_vocab import build_vocab, SPECIAL_TOKENS_FOR_BERT_MODEL
from adaptplm.mlm.train_seq_rxn_encoder import SeqRxnEncoderTrainingConfig, train_seq_rxn_encoder_with_mlm


@click.command()
@click.argument('data-file', type=click.Path(exists=True, path_type=Path),
                default=DefaultPath().data_dataset_processed and
                        DefaultPath().data_dataset_processed / 'enzsrp_full_cleaned' / 'enzsrp_full_cleaned_train.csv')
@click.argument('output-file', type=click.Path(path_type=Path),
                default=DefaultPath().data_dataset_processed and
                        DefaultPath().data_dataset_processed.joinpath("vocab", "enzsrp_full_cleand_train_vocab.txt"))
def build_vocab_enzsrp_full_cli(data_file: Path, output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    return build_vocab(data_file, output_file, ignore_first_line=True, min_freq=5,
                       base_tokens=SPECIAL_TOKENS_FOR_BERT_MODEL)


# gradient checkpointing might be useful
# https://github.com/facebookresearch/esm/issues/606?utm_source=chatgpt.com
@click.command()
@click.option('--train-data-path', help='',
              default=DefaultPath().data_dataset_processed and
                      DefaultPath().data_dataset_processed / 'enzsrp_cleaned' / 'enzsrp_full_cleaned_train.csv')
@click.option('--eval-data-path', help='',
              default=DefaultPath().data_dataset_processed and
                      DefaultPath().data_dataset_processed / 'enzsrp_cleaned' / 'enzsrp_full_cleaned_val.csv')
@click.option('--vocab-path', help='',
              default=DefaultPath().data_dataset_processed and
                      DefaultPath().data_dataset_processed / 'vocab' / 'enzsrp_full_cleand_train_vocab.txt')
@click.option('--out-parent-dir', type=click.Path(),
              default=DefaultPath().local_exp_seqrxn_encoder_train,
              help='')
@click.option('--mlm-probability', type=float, help='', default=0.3)
@click.option('--n-training-steps', type=int, help='', default=200_000)
@click.option('--batch-size', type=int, help='', default=8)  # assuming using gradient accumulation
@click.option('--save-steps', type=int, help='', default=1000)
@click.option('--eval-steps', type=int, help='', default=2000)
@click.option('--gradient-accumulation_steps', type=int, help='', default=2)
@click.option('--untrained-lr', type=float, default=2e-4)
@click.option('--trained-lr', type=float,
              default=5e-5)  # default learning_rate for AdamW defined in TrainingArguments class
@click.option('--esm-upper-layer-lr', type=float, default=5e-5)
@click.option('--bert-pretrained', help='', type=click.Path(exists=True), default=None)
@click.option('--bert-model-config-file', help='', type=click.Path(exists=True), default=None)
@click.option('--esm-pretrained', help='', type=str)
@click.option('--seq-rxn-encoder-config-file', help='', type=click.Path(exists=True))
@click.option('--seed', help='', type=int, default=42)
@click.option('--max-checkpoints', type=int, default=0,
              help='The maximum number of saved models to keep.')
@click.option('--early-stopping-patience', type=int, default=5,
              help='Number of eval steps with no improvement after which training will be stopped.')
@click.option('--early-stopping-min-delta', type=float, default=0.0001,
              help='The minimum change in the monitored quantity to qualify as an improvement.')
@click.option('--mixed-precision', type=str,
              default='fp16')
@click.option('--randomize-rxn-smiles', is_flag=True, help='')
@click.option('--weighted-sampling', is_flag=True, help='')
@click.option('--clstr-for-training-seq', type=click.Path(exists=True), default=None, help='')
@click.option('--use-cpu', is_flag=True, help='')
@click.option('--debug', is_flag=True, help='Enable debug mode')
def train_seq_rxn_encoder_with_mlm_cli(train_data_path: str, eval_data_path: str, vocab_path: str,
                                       out_parent_dir: str, mlm_probability: float, n_training_steps: int,
                                       batch_size: int,
                                       save_steps: int, eval_steps: int, gradient_accumulation_steps: int,
                                       untrained_lr: float, trained_lr: float, esm_upper_layer_lr: float,
                                       seq_rxn_encoder_config_file: str, bert_pretrained: Optional[str],
                                       bert_model_config_file: Optional[str],
                                       esm_pretrained: str, seed: int, max_checkpoints: int,
                                       early_stopping_patience: int, early_stopping_min_delta: int,
                                       mixed_precision: str,
                                       randomize_rxn_smiles: bool, weighted_sampling: bool,
                                       clstr_for_training_seq: str,
                                       use_cpu: bool, debug: bool):
    assert (bert_pretrained is None) != (bert_model_config_file is None)
    training_config = SeqRxnEncoderTrainingConfig(
        train_csv=Path(train_data_path),
        val_csv=Path(eval_data_path),
        vocab_file=Path(vocab_path),
        out_parent_dir=Path(out_parent_dir) if out_parent_dir is not None else None,
        mlm_probability=mlm_probability,
        n_training_steps=n_training_steps,
        batch_size=batch_size,
        save_steps=save_steps,
        eval_steps=eval_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        untrained_lr=untrained_lr,
        trained_lr=trained_lr,
        esm_upper_layer_lr=esm_upper_layer_lr,
        seq_rxn_encoder_config_file=Path(seq_rxn_encoder_config_file),
        bert_pretrained=bert_pretrained,
        bert_model_config_file=Path(bert_model_config_file) if bert_model_config_file else None,
        esm_pretrained=esm_pretrained,
        seed=seed,
        max_checkpoints=max_checkpoints,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        mixed_precision=mixed_precision,
        randomize_rxn_smiles=randomize_rxn_smiles,
        weighted_sampling=weighted_sampling,
        clstr_for_training_seq=Path(clstr_for_training_seq) if clstr_for_training_seq else None,
        use_cpu=use_cpu,
    )
    train_seq_rxn_encoder_with_mlm(training_config, is_debug_mode=debug)
