from dataclasses import dataclass
from enum import Enum
from typing import ClassVar


class EmbeddingType(Enum):
    AVERAGE = "average"
    ACTIVE_SITE = "active_site"  # pooling residues near active site in reference structure
    COVERAGE = "coverage"  # pooling columns with the fewest gaps
    CONSERVATION = "conservation"  # pooling columns that have the highest frequency of any single amino acid type
    RXN_MLM = 'rxn_mlm'
    PRECOMPUTED = 'precomputed'

    @staticmethod
    def from_label(label: str):
        for embedding_type in EmbeddingType:
            if embedding_type.value == label:
                return embedding_type
        raise ValueError(f"No matching ModelParamFreezeStrategy found for label: {label}")


@dataclass(frozen=True)
class ProteinLMEmbedding:
    embedding_type: ClassVar[EmbeddingType]

    @staticmethod
    def from_embedding_type(embedding_type: EmbeddingType, **kwargs):
        for embed_cls in [AveragePoolEmbedding, ActiveSitePoolEmbedding, CoveragePoolEmbedding,
                          ConservationPoolEmbedding, EnzRxnMLMEmbedding, PrecomputedEmbedding]:
            if embedding_type == embed_cls.embedding_type:
                return embed_cls(**kwargs)
        raise ValueError('undefined embedding type')


class AveragePoolEmbedding(ProteinLMEmbedding):
    embedding_type: ClassVar[EmbeddingType] = EmbeddingType.AVERAGE

    def __init__(self, **kwargs):
        pass


# Not used in the experiment.
@dataclass(frozen=True)
class ActiveSitePoolEmbedding(ProteinLMEmbedding):
    embedding_type: ClassVar[EmbeddingType] = EmbeddingType.ACTIVE_SITE
    distance: float


# Not used in the experiment.
@dataclass(frozen=True)
class CoveragePoolEmbedding(ProteinLMEmbedding):
    embedding_type: ClassVar[EmbeddingType] = EmbeddingType.COVERAGE
    top_k: int


# Not used in the experiment.
@dataclass(frozen=True)
class ConservationPoolEmbedding(ProteinLMEmbedding):
    embedding_type: ClassVar[EmbeddingType] = EmbeddingType.CONSERVATION
    top_k: int


class EnzRxnMLMEmbedding(ProteinLMEmbedding):
    embedding_type: ClassVar[EmbeddingType] = EmbeddingType.RXN_MLM

    def __init__(self, **kwargs):
        pass


@dataclass(frozen=True)
class PrecomputedEmbedding(ProteinLMEmbedding):
    embedding_type: ClassVar[EmbeddingType] = EmbeddingType.PRECOMPUTED
    precomputed_embeddings_npy: str
