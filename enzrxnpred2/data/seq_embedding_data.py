from typing import Dict

import numpy as np
import pandas as pd
from pandas import DataFrame


class SequenceEmbeddingData:

    def __init__(self, label_embedding_dict: Dict[str, np.array]):
        self._label_embedding_dict = label_embedding_dict

    @classmethod
    def from_npy(cls, npy_path):
        loaded_data = np.load(npy_path, allow_pickle=False)
        loaded_labels = loaded_data["sequence"]
        loaded_embeddings = loaded_data["embeddings"]
        final_dict = {}
        for label, embedding in zip(loaded_labels, loaded_embeddings):
            final_dict[label] = embedding
        return SequenceEmbeddingData(final_dict)

    def to_npy(self, npy_path):  # TODO: rename
        labels = np.array(list(self._label_embedding_dict.keys()))
        embeddings = np.array(list(self._label_embedding_dict.values()))
        np.savez(npy_path, sequence=labels, embeddings=embeddings)

    @property
    def df(self) -> DataFrame:
        return pd.DataFrame(list(self._label_embedding_dict.items()), columns=['sequence', 'embedding'])

    def get_one_embedding_by_hash(self, crc64: str):
        return self._label_embedding_dict[crc64]
