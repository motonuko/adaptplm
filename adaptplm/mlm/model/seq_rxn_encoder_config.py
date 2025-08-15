import json
from dataclasses import dataclass


@dataclass
class SeqRxnEncoderConfig:

    def __init__(self,
                 n_additional_layers=1,
                 n_trainable_esm_layers=1,
                 initializer_range=0.02,  # same as ESM and BERT default parameter
                 # tie_word_embeddings=True  # not used
                 ):
        self.n_additional_layers = n_additional_layers
        self.n_trainable_esm_layers = n_trainable_esm_layers
        self.initializer_range = initializer_range
        # self.tie_word_embeddings = tie_word_embeddings
        # self.bert_config = BertConfig.from_dict(kwargs['bert'])

    @classmethod
    def from_json_file(cls, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            return cls(**data)
