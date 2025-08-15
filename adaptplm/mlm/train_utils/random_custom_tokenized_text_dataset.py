import pandas as pd
import torch.utils.data as data
from transformers import PreTrainedTokenizer

from adaptplm.extension.rdkit_ext import randomize_reaction_smiles
from adaptplm.mlm.train_utils.custom_data_collator import CustomDataInput


class RandomCustomTokenizedTextDataset(data.Dataset):

    def __init__(self, file_path, seq_tokenizer: PreTrainedTokenizer, smi_tokenizer: PreTrainedTokenizer,
                 max_seq_token_length: int, max_rxn_token_length: int):
        self.smi_tokenizer = smi_tokenizer
        self.max_rxn_token_length = max_rxn_token_length
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
            aa_seq_tokenized_inputs.data = {k: v.view(-1) for k, v in aa_seq_tokenized_inputs.data.items()}
            self.items.append((aa_seq_tokenized_inputs, row['rxn']))

    # NOTE: Training runtime tokenization may increase overhead. Adjust num_workers and prefetch_factor of DataLoader.
    def __getitem__(self, idx):
        rxn = self.items[idx][1]
        rxn = randomize_reaction_smiles(rxn)
        rxn_smiles_tokenized_inputs = self.smi_tokenizer(rxn, add_special_tokens=True,
                                                         max_length=self.max_rxn_token_length,
                                                         padding=False, truncation=True, return_tensors='pt',
                                                         return_attention_mask=False)
        rxn_smiles_tokenized_inputs.data = {k: v.view(-1) for k, v in rxn_smiles_tokenized_inputs.data.items()}
        return CustomDataInput(seq_batch_encoding=self.items[idx][0], smiles_batch_encoding=rxn_smiles_tokenized_inputs)

    def __len__(self):
        return len(self.items)
