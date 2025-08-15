import copy

import numpy as np
import torch
from numpy import float32
from numpy.typing import NDArray
from transformers import EsmTokenizer, EsmModel

from adaptplm.core.default_path import DefaultPath
from adaptplm.data.crc64_embedding_data import CRC64EmbeddingData
from adaptplm.data.original_enz_activity_dense_screen_datasource import EnzActivityScreeningDataset
from adaptplm.downstream.cpi.domain.exp_config import ProteinFeatConfig
from adaptplm.downstream.cpi.domain.protein_lm_embedding import CoveragePoolEmbedding, \
    ConservationPoolEmbedding, \
    AveragePoolEmbedding, ActiveSitePoolEmbedding, EnzRxnMLMEmbedding, PrecomputedEmbedding
from adaptplm.downstream.cpi.external import parse_utils
from adaptplm.extension.bio_ext import calculate_crc64


def _get_mapping(pooling_strategy, dataset: EnzActivityScreeningDataset):
    if isinstance(pooling_strategy, AveragePoolEmbedding):
        return None
    elif isinstance(pooling_strategy, ActiveSitePoolEmbedding):
        ssa_ref_file = DefaultPath().data_original_dense_screen_processed_structure.joinpath(
            f"{dataset.short_name_for_original_files}_reference_{pooling_strategy.distance}.txt")
        ref_seq, pool_residues = parse_utils.parse_ssa_reference(ssa_ref_file.as_posix())
        msa_file = DefaultPath().data_original_dense_screen_processed_alignments.joinpath(
            f"{dataset.short_name_for_original_files}_alignment.fasta")
        pool_residues = sorted(pool_residues)
        return parse_utils.extract_pool_residue_dict(msa_file.as_posix(), ref_seq, pool_residues)
    elif isinstance(pooling_strategy, CoveragePoolEmbedding):
        msa_file = DefaultPath().data_original_dense_screen_processed_alignments.joinpath(
            f"{dataset.short_name_for_original_files}_alignment.fasta")
        return parse_utils.extract_coverage_residue_dict(msa_file.as_posix(), pool_num=pooling_strategy.top_k)
    elif isinstance(pooling_strategy, ConservationPoolEmbedding):
        msa_file = DefaultPath().data_original_dense_screen_processed_alignments.joinpath(
            f"{dataset.short_name_for_original_files}_alignment.fasta")
        return parse_utils.extract_conserve_residue_dict(msa_file.as_posix(), pool_num=pooling_strategy.top_k)
    raise ValueError("unexpected")


class PrecomputedEmbeddingLoader:

    def __init__(self, npy_path):
        self._data: CRC64EmbeddingData = CRC64EmbeddingData.from_npy(npy_path)

    def get_embedding(self, sequence: str) -> np.array:
        crc64_hash = calculate_crc64(sequence)
        return self._data.get_one_embedding_by_hash(crc64_hash)


class EsmFeatureConstructor:
    def __init__(self, config: ProteinFeatConfig, dataset: EnzActivityScreeningDataset, device):
        self._config = config
        self._device = device
        self._tokenizer = None
        self._model = None
        self._cache = {}

        embedding_config = self._config.embedding_config
        if isinstance(embedding_config, EnzRxnMLMEmbedding):
            model = EsmModel.from_pretrained(config.model_name)
            self._model = model.to(self._device)
            self._tokenizer = EsmTokenizer.from_pretrained(config.model_name)
            self._construct_feat_func = self._construct_feature_esm_load
        elif isinstance(embedding_config, PrecomputedEmbedding):
            self._model = PrecomputedEmbeddingLoader(embedding_config.precomputed_embeddings_npy)
            self._construct_feat_func = self._construct_feature_from_precomputed_embedding
        else:
            self._model = EsmModel.from_pretrained(config.model_name).to(self._device)
            self._tokenizer = EsmTokenizer.from_pretrained(config.model_name)
            self._construct_feat_func = self._construct_feature_esm_fixed
            self._pool_residues_mapping = _get_mapping(config.embedding_config, dataset)

    def construct_feature(self, sequence: str) -> NDArray[float32]:
        if sequence in self._cache.keys():
            return self._cache[sequence]
        feat = self._construct_feat_func(sequence)
        self._cache[sequence] = copy.deepcopy(feat)
        return feat

    def _construct_feature_esm_fixed(self, sequence):
        inputs = self._tokenizer(sequence, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
        # hidden_states = outputs.last_hidden_state
        hidden_states = outputs.last_hidden_state[:, 1:-1, :]
        assert len(sequence) == hidden_states.size(1)
        if self._pool_residues_mapping is None:
            feature_vector = hidden_states.mean(dim=1).squeeze().cpu().numpy()
        else:
            selected_indexes = self._pool_residues_mapping[sequence]
            selected_tokens = hidden_states[:, selected_indexes, :]
            feature_vector = selected_tokens.mean(dim=1).squeeze().cpu().numpy()
        return feature_vector

    #  Our Model
    def _construct_feature_esm_load(self, sequence):
        inputs = self._tokenizer(sequence, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        feature_vector = outputs.pooler_output.flatten().cpu().numpy()
        return feature_vector

    def _construct_feature_from_precomputed_embedding(self, sequence):
        return self._model.get_embedding(sequence)

    def clear(self):
        if self._model is not None:
            del self._model
            torch.cuda.empty_cache()
            self._model = None
            print("cache cleared")

    def __del__(self):
        self.clear()
