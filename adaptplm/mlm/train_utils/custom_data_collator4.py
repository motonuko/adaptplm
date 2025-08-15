from dataclasses import dataclass
from typing import List, Mapping, Optional

import pandas as pd
import torch.utils.data as data
from transformers import PreTrainedTokenizer, BatchEncoding, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin, _torch_collate_batch
from transformers.utils import PaddingStrategy

from adaptplm.core.constants import MAX_SEQUENCE_LENGTH
from adaptplm.extension.bio_ext import calculate_crc64


@dataclass
class CustomDataInput:
    seq_batch_encoding: dict
    data_id: str


class CustomTokenizedTextDataset4(data.Dataset):

    def __init__(self, file_path, tokenizer: PreTrainedTokenizer, max_seq_token_length: int = MAX_SEQUENCE_LENGTH):
        print("Loading and Tokenizing data ...")
        df = pd.read_csv(file_path, names=['sequence'])
        df['seq_crc64'] = df['sequence'].apply(calculate_crc64)
        df['len_seq'] = df['sequence'].apply(len)
        print("Read", len(df), "lines from input and target files.")

        self.items = []
        for _, row in df.iterrows():
            # padding=False, return_attention_mask=False since padding is applied later.
            aa_seq_tokenized_inputs = tokenizer(row['sequence'], add_special_tokens=True,
                                                max_length=max_seq_token_length,
                                                padding=False, truncation=True, return_tensors='pt',
                                                return_attention_mask=False)
            aa_seq_tokenized_inputs.data = {k: v.view(-1) for k, v in aa_seq_tokenized_inputs.data.items()}
            data_id = row['seq_crc64']
            self.items.append((aa_seq_tokenized_inputs, data_id))

    def __getitem__(self, idx):
        return CustomDataInput(seq_batch_encoding=self.items[idx][0], data_id=self.items[idx][1])

    def __len__(self):
        return len(self.items)


@dataclass
class MyCustomDataCollator4(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    # override
    def torch_call(self, examples: List[CustomDataInput]) -> BatchEncoding:
        seq_encoding = [example.seq_batch_encoding for example in examples]
        aa_seq_encodings = self.construct_batch(seq_encoding, self.tokenizer)
        data_ids = [example.data_id for example in examples]
        return BatchEncoding({
            "input_ids": aa_seq_encodings["input_ids"],
            "attention_mask": aa_seq_encodings["attention_mask"],
            "labels": aa_seq_encodings.get("labels"),
            "data_ids": data_ids,
        })

    def construct_batch(self, examples: List[dict], tokenizer: PreTrainedTokenizerBase):
        if isinstance(examples[0], Mapping):
            batch = tokenizer.pad(examples, return_tensors="pt", padding=PaddingStrategy.LONGEST,
                                  max_length=1024,
                                  pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }
        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        labels = batch["input_ids"].clone()
        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch
