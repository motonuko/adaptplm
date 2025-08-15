from dataclasses import dataclass
from typing import List, Any, Mapping, Optional, Tuple

import pandas as pd
import torch.utils.data as data
from transformers import PreTrainedTokenizer, BatchEncoding, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin, _torch_collate_batch
from transformers.utils import PaddingStrategy


@dataclass
class CustomDataInput:
    seq_batch_encoding: dict
    smiles_batch_encoding: dict


class CustomTokenizedTextDataset(data.Dataset):

    def __init__(self, file_path, seq_tokenizer: PreTrainedTokenizer, smi_tokenizer: PreTrainedTokenizer,
                 max_seq_token_length: int,
                 max_rxn_token_length: int):
        print("Loading and Tokenizing data ...")
        df = pd.read_csv(file_path)
        print("Read", len(df), "lines from input and target files.")

        self.items = []
        for _, row in df.iterrows():
            # padding=False, return_attention_mask=False since padding is applied later.
            aa_seq_tokenized_inputs = seq_tokenizer(row['sequence'], add_special_tokens=True,
                                                    max_length=max_seq_token_length,
                                                    padding=False, truncation=True, return_tensors='pt',
                                                    return_attention_mask=False)
            rxn_smiles_tokenized_inputs = smi_tokenizer(row['rxn'], add_special_tokens=True,
                                                        max_length=max_rxn_token_length,
                                                        padding=False, truncation=True, return_tensors='pt',
                                                        return_attention_mask=False)
            aa_seq_tokenized_inputs.data = {k: v.view(-1) for k, v in aa_seq_tokenized_inputs.data.items()}
            rxn_smiles_tokenized_inputs.data = {k: v.view(-1) for k, v in rxn_smiles_tokenized_inputs.data.items()}
            self.items.append((aa_seq_tokenized_inputs, rxn_smiles_tokenized_inputs))

    def __getitem__(self, idx):
        # seq = {key: value for key, value in self.items[idx][0].items()}
        # smiles = {key: value for key, value in self.items[idx][0].items()}
        return CustomDataInput(seq_batch_encoding=self.items[idx][0], smiles_batch_encoding=self.items[idx][1])

    def __len__(self):
        return len(self.items)


@dataclass
class MyCustomDataCollator(DataCollatorMixin):
    aa_sequence_tokenizer: PreTrainedTokenizerBase
    rxn_smiles_tokenizer: PreTrainedTokenizerBase
    aa_seq_mlm: bool = False
    aa_seq_mlm_probability: float = 0.15
    rxn_smiles_mlm: bool = True
    rxn_smiles_mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.aa_seq_mlm and self.aa_sequence_tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )
        if self.rxn_smiles_mlm and self.rxn_smiles_tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    # override
    def torch_call(self, examples: List[CustomDataInput]) -> BatchEncoding:
        seq_encoding = [example.seq_batch_encoding for example in examples]
        aa_seq_encodings = self.construct_batch(seq_encoding, self.aa_sequence_tokenizer, self.aa_seq_mlm,
                                                self.aa_seq_mlm_probability)
        smi_encoding = [example.smiles_batch_encoding for example in examples]
        smiles_encodings = self.construct_batch(smi_encoding, self.rxn_smiles_tokenizer, self.rxn_smiles_mlm,
                                                self.rxn_smiles_mlm_probability)
        return BatchEncoding({
            "aaseq_input_ids": aa_seq_encodings["input_ids"],
            "aaseq_attention_mask": aa_seq_encodings["attention_mask"],
            "aaseq_labels": aa_seq_encodings.get("labels"),
            "smiles_input_ids": smiles_encodings["input_ids"],
            "smiles_attention_mask": smiles_encodings["attention_mask"],
            "smiles_labels": smiles_encodings.get("labels")})

    # source: transformers.DataCollatorForLanguageModeling
    def construct_batch(self, examples: List[dict], tokenizer: PreTrainedTokenizerBase, mlm: bool,
                        mlm_probability: float):
        if isinstance(examples[0], Mapping):
            batch = tokenizer.pad(examples, return_tensors="pt", padding=PaddingStrategy.LONGEST,
                                  pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], tokenizer, mlm_probability=mlm_probability,
                special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if tokenizer.pad_token_id is not None:
                labels[labels == tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    # source: transformers.DataCollatorForLanguageModeling
    @staticmethod
    def torch_mask_tokens(inputs: Any, tokenizer: PreTrainedTokenizerBase, mlm_probability: float,
                          special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
