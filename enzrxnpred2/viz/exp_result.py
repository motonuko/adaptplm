from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from enzrxnpred2.data.original_enz_activity_dense_screen_datasource import EnzActivityScreeningDataset
from enzrxnpred2.downstream.cpi.domain.exp_config import ProteinFeatConfig, MoleculeFeatConfig, OptimParams, \
    construct_optim_param_obj
from enzrxnpred2.downstream.cpi.domain.protein_lm_embedding import EmbeddingType


@dataclass(frozen=True)
class FoldResult:
    fold_idx: int
    best_params: Dict
    roc_auc: float
    pr_auc: float
    # optional for backward compatibility
    test_seq_crc64: Optional[List[str]]

    @staticmethod
    def from_dict(obj: Dict[str, Any]) -> 'FoldResult':
        return FoldResult(
            fold_idx=obj['fold_idx'],
            best_params=obj['best_params'],
            roc_auc=obj.get('roc_auc', None) or obj.get('avg_roc_auc', None),
            pr_auc=obj.get('pr_auc', None) or obj.get('avg_pr_auc', None),
            test_seq_crc64=obj.get('test_seq_crc64', [])
        )

    @property
    def hash_key(self):
        return hash((self.fold_idx, tuple(self.test_seq_crc64)))


@dataclass(frozen=True)
class ExpResult:
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
    n_steps: int
    eval_steps: int
    eval_metrics: List[str]
    #
    averaged_fold_results_avg_roc_auc_mean: float
    averaged_fold_results_avg_roc_auc_std: float
    averaged_fold_results_avg_pr_auc_mean: float
    averaged_fold_results_avg_pr_auc_std: float
    #
    fold_results: List[FoldResult]

    @staticmethod
    def from_dict(obj: Dict[str, Any]) -> 'ExpResult':
        config = obj['config']
        avg_fold_result = obj['averaged_fold_results']
        return ExpResult(
            protein_feat_config=ProteinFeatConfig.from_dict(config['protein_feat_config']),
            molecule_feat_config=MoleculeFeatConfig.from_dict(config['molecule_feat_config']),
            model=config['model'],
            class_weight=config['class_weight'],
            data=EnzActivityScreeningDataset.from_label(config["data"]),
            n_inner_folds=config['n_inner_folds'],
            n_outer_folds=config['n_outer_folds'],
            optim_params=[construct_optim_param_obj(i["type"], i) for i in config['optim_params']],
            optuna_n_trials=config['optuna_n_trials'],
            optuna_n_startup_trials=config['optuna_n_startup_trials'],
            optim_metric=config['optim_metric'],
            random_seed=config['random_seed'],
            inner_cv_seed=config['inner_cv_seed'],
            outer_cv_seed=config['outer_cv_seed'],
            n_steps=config['n_steps'],
            eval_steps=config['eval_steps'],
            eval_metrics=config['eval_metrics'],
            averaged_fold_results_avg_roc_auc_mean=
            (avg_fold_result.get('roc_auc', None) or avg_fold_result.get('avg_roc_auc', None))['mean'],
            averaged_fold_results_avg_roc_auc_std=
            (avg_fold_result.get('roc_auc', None) or avg_fold_result.get('avg_roc_auc', None))['std'],
            averaged_fold_results_avg_pr_auc_mean=
            (avg_fold_result.get('pr_auc', None) or avg_fold_result.get('avg_pr_auc', None))['mean'],
            averaged_fold_results_avg_pr_auc_std=
            (avg_fold_result.get('pr_auc', None) or avg_fold_result.get('avg_pr_auc', None))['std'],
            fold_results=[FoldResult.from_dict(fold) for fold in obj['fold_result']]
        )

    @property
    def group_key(self):
        config = self.protein_feat_config.to_dict()
        if config['embedding_type'] == EmbeddingType.ACTIVE_SITE.value:
            dist = config['embedding_params']['distance']
            return f"Active Site\n(dist. {dist})"
        elif config['embedding_type'] == EmbeddingType.AVERAGE.value:
            return f"Mean"
        elif config['embedding_type'] == EmbeddingType.RXN_MLM.value:
            short_name = self.protein_feat_config.model_name.split('/')[-2]
            return f"Masked LM {short_name}"
        elif config['embedding_type'] == EmbeddingType.PRECOMPUTED.value:
            # embedding_config: PrecomputedEmbedding = self.protein_feat_config.embedding_config
            # short_name = embedding_config.precomputed_embeddings_npy.split('/')[-2]
            return f"Precomputed"
        raise ValueError('unexpected')

    @property
    def hash_key(self):
        return hash(
            str(self.molecule_feat_config.to_dict()) +
            str(self.model) +
            str(self.class_weight) +
            str(self.data.value) +
            str(self.n_inner_folds) +
            str(self.n_outer_folds) +
            str([param.to_dict() for param in self.optim_params]) +
            str(self.optuna_n_trials) +
            str(self.optuna_n_startup_trials) +
            str(self.optim_metric) +
            str(self.n_steps) +
            str(self.eval_steps) +
            str(self.eval_metrics) +
            str((fold_result.hash_key for fold_result in self.fold_results))
        )
