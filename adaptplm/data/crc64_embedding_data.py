from typing import Dict

import numpy as np
import pandas as pd
from pandas import DataFrame


class CRC64EmbeddingData:

    def __init__(self, label_embedding_dict: Dict[str, np.array]):
        self._label_embedding_dict = label_embedding_dict

    @classmethod
    def from_npy(cls, npy_path):
        loaded_data = np.load(npy_path, allow_pickle=False)
        loaded_labels = loaded_data["labels"]
        loaded_embeddings = loaded_data["embeddings"]
        final_dict = {}
        for label, embedding in zip(loaded_labels, loaded_embeddings):
            final_dict[label] = embedding
        return CRC64EmbeddingData(final_dict)

    @property
    def df(self) -> DataFrame:
        return pd.DataFrame(self._label_embedding_dict)

    def get_one_embedding_by_hash(self, crc64: str):
        return self._label_embedding_dict[crc64]
