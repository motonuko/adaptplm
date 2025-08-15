from dataclasses import dataclass, asdict
from typing import Dict, Any, List

from enzrxnpred2.data.original_enz_activity_dense_screen_datasource import EnzActivityScreeningDataset
from enzrxnpred2.downstream.cpi.domain.protein_lm_embedding import EmbeddingType, ProteinLMEmbedding


@dataclass(frozen=True)
class ProteinFeatConfig:
    model_name: str
    embedding_config: ProteinLMEmbedding

    @staticmethod
    def from_dict(obj: Dict[str, Any]) -> 'ProteinFeatConfig':
        # embedding_type_name = EmbeddingType.from_label(obj['embedding_type'])
        # return ProteinFeatConfig(
        #     model_name=obj['model_name'],
        #     embedding_config=ProteinLMEmbedding.from_embedding_type(embedding_type_name, **obj['embedding_params'])
        # )
        # NOTE: for backward compatibility
        embedding_type = obj.get('embedding_type', None) or obj.get('pooling_strategy', None)
        model_name = obj.get('model_name', None) or obj.get('name', None)
        embedding_type_name = EmbeddingType.from_label(embedding_type)
        if 'embedding_params' in obj.keys():
            embedding_params = obj['embedding_params']
        else:
            embedding_params = obj['pooling_params']
        return ProteinFeatConfig(
            model_name=model_name,
            embedding_config=ProteinLMEmbedding.from_embedding_type(embedding_type_name, **embedding_params)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'embedding_type': self.embedding_config.embedding_type.value,
            'embedding_params': asdict(self.embedding_config)
        }


@dataclass(frozen=True)
class MoleculeFeatConfig:
    name: str
    params: Dict[str, Any]

    @staticmethod
    def from_dict(obj: Dict[str, Any]) -> 'MoleculeFeatConfig':
        return MoleculeFeatConfig(
            name=obj['name'],
            params=obj['params']
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'params': self.params
        }


@dataclass(frozen=True)
class OptimParams:
    name: str

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError()


@dataclass(frozen=True)
class IntOptimParams(OptimParams):
    min: int
    max: int
    step: int
    type: str = "int"

    @staticmethod
    def from_dict(obj: Dict[str, Any]) -> 'IntOptimParams':
        return IntOptimParams(
            name=obj['name'],
            min=obj['min'],
            max=obj['max'],
            step=obj['step'],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.type,
            'min': self.min,
            'max': self.max,
            'step': self.step
        }


@dataclass(frozen=True)
class FloatOptimParams(OptimParams):
    min: float
    max: float
    type: str = "float"

    @staticmethod
    def from_dict(obj: Dict[str, Any]) -> 'FloatOptimParams':
        return FloatOptimParams(
            name=obj['name'],
            min=obj['min'],
            max=obj['max'],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.type,
            'min': self.min,
            'max': self.max,
        }


@dataclass(frozen=True)
class CategoricalOptimParams(OptimParams):
    name: str
    candidates: List
    type: str = "categorical"

    @staticmethod
    def from_dict(obj: Dict[str, Any]) -> 'CategoricalOptimParams':
        return CategoricalOptimParams(
            name=obj['name'],
            candidates=obj['candidates'],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.type,
            'candidates': self.candidates,
        }


def construct_optim_param_obj(optim_type: str, optim_data: dict):
    if optim_type == IntOptimParams.type:
        return IntOptimParams(name=optim_data["name"], min=optim_data["min"], max=optim_data["max"],
                              step=optim_data["step"])
    if optim_type == FloatOptimParams.type:
        return FloatOptimParams(name=optim_data["name"], min=optim_data["min"], max=optim_data["max"])
    if optim_type == CategoricalOptimParams.type:
        return CategoricalOptimParams(name=optim_data["name"], candidates=optim_data["candidates"])
    else:
        raise ValueError("unexpected optim type")


@dataclass(frozen=True)
class ExpConfig:
    protein_feat_config: ProteinFeatConfig
    molecule_feat_config: MoleculeFeatConfig
    model: str
    class_weight: str
    n_outer_folds: int
    n_inner_folds: int
    optim_params: List[OptimParams]
    optim_metric: str
    optuna_n_trials: int
    optuna_n_startup_trials: int
    random_seed: int
    inner_cv_seed: int
    outer_cv_seed: int
    data: EnzActivityScreeningDataset
    filter_seq_list: List[str]
    n_steps: int
    eval_steps: int
    use_cpu: bool
    eval_metrics: List[str]
    ensemble_top_k: int

    @staticmethod
    def from_dict(obj: Dict[str, Any]) -> 'ExpConfig':
        return ExpConfig(
            protein_feat_config=ProteinFeatConfig.from_dict(obj['protein_feat_config']),
            molecule_feat_config=MoleculeFeatConfig.from_dict(obj['molecule_feat_config']),
            model=obj['model'],
            class_weight=obj['class_weight'],
            data=EnzActivityScreeningDataset.from_label(obj["data"]),
            filter_seq_list=obj['filter_seq_list'],
            n_inner_folds=obj['n_inner_folds'],
            n_outer_folds=obj['n_outer_folds'],
            optim_params=[construct_optim_param_obj(i["type"], i) for i in obj['optim_params']],
            optuna_n_trials=obj['optuna_n_trials'],
            optuna_n_startup_trials=obj['optuna_n_startup_trials'],
            optim_metric=obj['optim_metric'],
            random_seed=obj['random_seed'],
            inner_cv_seed=obj['inner_cv_seed'],
            outer_cv_seed=obj['outer_cv_seed'],
            n_steps=obj['n_steps'],
            eval_steps=obj['eval_steps'],
            use_cpu=obj['use_cpu'],
            eval_metrics=obj['eval_metrics'],
            ensemble_top_k=obj['ensemble_top_k']
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'protein_feat_config': self.protein_feat_config.to_dict(),
            'molecule_feat_config': self.molecule_feat_config.to_dict(),
            'model': self.model,
            'class_weight': self.class_weight,
            'n_inner_folds': self.n_inner_folds,
            'n_outer_folds': self.n_outer_folds,
            'optim_params': [param.to_dict() for param in self.optim_params],
            'optuna_n_trials': self.optuna_n_trials,
            'optuna_n_startup_trials': self.optuna_n_startup_trials,
            'optim_metric': self.optim_metric,
            'random_seed': self.random_seed,
            'inner_cv_seed': self.inner_cv_seed,
            'outer_cv_seed': self.outer_cv_seed,
            'n_steps': self.n_steps,
            'eval_steps': self.eval_steps,
            'data': self.data.value,
            'filter_seq_list': self.filter_seq_list,
            'use_cpu': self.use_cpu,
            'eval_metrics': self.eval_metrics,
            'ensemble_top_k': self.ensemble_top_k
        }
