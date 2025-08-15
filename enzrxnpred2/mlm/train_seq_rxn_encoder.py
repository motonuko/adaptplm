import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import EsmTokenizer, PreTrainedModel

from enzrxnpred2.data.cd_hit_result_datasource import CdHitResultDatasource
from enzrxnpred2.extension.seed import set_random_seed
from enzrxnpred2.mlm.calculate_sampling_weight import calculate_sampling_weight
from enzrxnpred2.mlm.model.seq_rxn_encoder import SeqRxnEncoderForMaskedLM
from enzrxnpred2.mlm.model.seq_rxn_encoder_config import SeqRxnEncoderConfig
from enzrxnpred2.mlm.tokenizer.smile_bert_tokenizer import SmilesBertTokenizer
from enzrxnpred2.mlm.train_utils.custom_data_collator import MyCustomDataCollator, CustomTokenizedTextDataset
from enzrxnpred2.mlm.train_utils.custom_train_loop import CustomTrainLoop, LoopConfig
from enzrxnpred2.mlm.train_utils.random_custom_tokenized_text_dataset import RandomCustomTokenizedTextDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeqRxnEncoderTrainingConfig(LoopConfig):
    train_csv: Path
    val_csv: Path
    vocab_file: Path
    esm_pretrained: Union[str, os.PathLike]
    bert_pretrained: Optional[Union[str, os.PathLike]]
    bert_model_config_file: Optional[Path]
    seq_rxn_encoder_config_file: Path
    mlm_probability: float
    out_parent_dir: Optional[Path]
    batch_size: int
    n_training_steps: int
    save_steps: int
    eval_steps: int
    untrained_lr: float
    trained_lr: float
    esm_upper_layer_lr: float
    seed: int
    weighted_sampling: bool
    clstr_for_training_seq: Optional[Path]
    use_cpu: bool

    @property
    def train_bert_from_scratch(self):
        return self.bert_model_config_file is not None

    def __post_init__(self):
        assert (self.bert_pretrained is None) != (self.bert_model_config_file is None)
        assert self.weighted_sampling == (self.clstr_for_training_seq is not None)


def train_seq_rxn_encoder_with_mlm(t_config: SeqRxnEncoderTrainingConfig, is_debug_mode=False):
    if is_debug_mode:
        logger.setLevel(logging.DEBUG)
        logger.debug("debug mode")
    set_random_seed(t_config.seed)
    out_dir = t_config.out_parent_dir.joinpath(datetime.now().strftime('%y%m%d_%H%M%S'))
    out_dir.mkdir(parents=True, exist_ok=True)
    esm_save_dir = out_dir / 'esm'
    bert_save_dir = out_dir / 'bert'
    t_config.to_json_file(out_dir.joinpath("training_config.json"))

    aa_seq_tokenizer = EsmTokenizer.from_pretrained(t_config.esm_pretrained)

    if t_config.train_bert_from_scratch:
        rxn_smiles_tokenizer = SmilesBertTokenizer(t_config.vocab_file.as_posix())
        model = SeqRxnEncoderForMaskedLM.from_pretrained_esm(esm_pretrained=t_config.esm_pretrained,
                                                             bert_config=t_config.bert_model_config_file,
                                                             config=SeqRxnEncoderConfig.from_json_file(
                                                                 t_config.seq_rxn_encoder_config_file),
                                                             debug=is_debug_mode)
    else:
        rxn_smiles_tokenizer = SmilesBertTokenizer.from_pretrained(t_config.bert_pretrained)
        with open(t_config.vocab_file) as f:
            new_vocabs = [line.strip() for line in f]
        added = rxn_smiles_tokenizer.add_tokens(new_vocabs)
        logging.info(f"Added {added} tokens")
        new_size_embeddings = len(rxn_smiles_tokenizer)
        model = SeqRxnEncoderForMaskedLM.from_pretrained_models(esm_pretrained=t_config.esm_pretrained,
                                                                bert_pretrained=t_config.bert_pretrained,
                                                                config=SeqRxnEncoderConfig.from_json_file(
                                                                    t_config.seq_rxn_encoder_config_file),
                                                                resized_token_embedding_size=new_size_embeddings,
                                                                debug=is_debug_mode)

    aa_seq_tokenizer.save_pretrained(save_directory=esm_save_dir)
    rxn_smiles_tokenizer.save_pretrained(save_directory=bert_save_dir)

    collator = MyCustomDataCollator(aa_sequence_tokenizer=aa_seq_tokenizer, rxn_smiles_tokenizer=rxn_smiles_tokenizer,
                                    rxn_smiles_mlm_probability=t_config.mlm_probability)
    esm_max_position_embeddings = model.esm.config.max_position_embeddings
    bert_max_position_embeddings = model.bert.config.max_position_embeddings
    logging.info(f"esm_max_position_embedding {esm_max_position_embeddings}")
    logging.info(f"bert_max_position_embedding {bert_max_position_embeddings}")

    # NOTE: ESM1b crashes max_seq_token_length == 1026 somehow, even  ESM1b max_position_embeddings == 1026
    max_seq_token_length = 1024
    if t_config.randomize_rxn_smiles:
        train_dataset = RandomCustomTokenizedTextDataset(t_config.train_csv, smi_tokenizer=rxn_smiles_tokenizer,
                                                         seq_tokenizer=aa_seq_tokenizer,
                                                         max_seq_token_length=max_seq_token_length,
                                                         max_rxn_token_length=bert_max_position_embeddings)
        eval_dataset = RandomCustomTokenizedTextDataset(t_config.val_csv, smi_tokenizer=rxn_smiles_tokenizer,
                                                        seq_tokenizer=aa_seq_tokenizer,
                                                        max_seq_token_length=max_seq_token_length,
                                                        max_rxn_token_length=bert_max_position_embeddings)
    else:
        train_dataset = CustomTokenizedTextDataset(t_config.train_csv, smi_tokenizer=rxn_smiles_tokenizer,
                                                   seq_tokenizer=aa_seq_tokenizer,
                                                   max_seq_token_length=max_seq_token_length,
                                                   max_rxn_token_length=bert_max_position_embeddings)

        eval_dataset = CustomTokenizedTextDataset(t_config.val_csv, smi_tokenizer=rxn_smiles_tokenizer,
                                                  seq_tokenizer=aa_seq_tokenizer,
                                                  max_seq_token_length=max_seq_token_length,
                                                  max_rxn_token_length=bert_max_position_embeddings)

    # pin_memory = device == "cuda"  # https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723/2
    num_worker = 0 if is_debug_mode else 2
    if t_config.weighted_sampling:
        df = pd.read_csv(t_config.train_csv)
        cd_hit_result = CdHitResultDatasource(t_config.clstr_for_training_seq)
        sample_weights = calculate_sampling_weight(df['sequence'].tolist(), cd_hit_result)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_data_loader = DataLoader(train_dataset, batch_size=t_config.batch_size, collate_fn=collator,
                                       num_workers=num_worker, pin_memory=False, shuffle=False, sampler=sampler)
    else:
        train_data_loader = DataLoader(train_dataset, batch_size=t_config.batch_size, collate_fn=collator,
                                       num_workers=num_worker, pin_memory=False, shuffle=True)
    eval_data_loader = DataLoader(eval_dataset, batch_size=t_config.batch_size, collate_fn=collator,
                                  num_workers=num_worker, pin_memory=False)

    if t_config.train_bert_from_scratch:
        trained_params, untrained_params, esm_upper_layer_params = model.get_parameters_for_untrained_bert()
    else:
        trained_params, untrained_params, esm_upper_layer_params = model.get_parameters()

    optimizer = AdamW([
        {'params': trained_params, 'lr': t_config.trained_lr},
        {'params': untrained_params, 'lr': t_config.untrained_lr},
        {'params': esm_upper_layer_params, 'lr': t_config.esm_upper_layer_lr},
    ])

    train_loop = CustomTrainLoop(model=model, optimizer=optimizer, train_data_loader=train_data_loader,
                                 eval_data_loader=eval_data_loader, loop_config=t_config, use_cpu=t_config.use_cpu,
                                 out_dir=out_dir)

    def save_model(pretrained: PreTrainedModel):
        # To easy to load trained protein encoder, save ESM model separately.
        pretrained.esm.save_pretrained(esm_save_dir)
        pretrained.esm.config.save_pretrained(save_directory=esm_save_dir)

        # Skip saving bert model (no use-case)
        pretrained.bert.config.save_pretrained(save_directory=bert_save_dir)

        torch.save(model.state_dict(), out_dir / 'final_model.pt')

    def save_checkpoint(pretrained: PreTrainedModel, steps: int) -> Path:
        esm_save_dir = out_dir / f"esm_step{steps}"
        pretrained.esm.save_pretrained(esm_save_dir)
        pretrained.esm.config.save_pretrained(save_directory=esm_save_dir)
        aa_seq_tokenizer.save_pretrained(save_directory=esm_save_dir)
        return esm_save_dir

    train_loop.train(save_model, save_checkpoint, label_key='smiles_labels')
