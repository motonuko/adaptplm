from dataclasses import dataclass
from typing import Dict, Any, List

from enzrxnpred2.data.original_enz_activity_dense_screen_datasource import EnzActivityScreeningDataset
from enzrxnpred2.downstream.cpi.domain.exp_config import ProteinFeatConfig, MoleculeFeatConfig, OptimParams, \
    construct_optim_param_obj, ExpConfig
from enzrxnpred2.downstream.cpi.domain.generate_seeds import generate_nested_cv_seeds


@dataclass(frozen=True)
class ExpMasterConfig:
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
    random_seeds: List[int]
    data: EnzActivityScreeningDataset
    filter_seq_list: List[str]
    n_steps: int
    eval_steps: int
    n_jobs: int  # NOTE: n_jobs = 1 behaves like running without Parallel
    use_cpu: bool
    eval_metrics: List[str]
    ensemble_top_k: int

    def __post_init__(self):
        assert len(self.random_seeds) == len(set(self.random_seeds))  # no duplicated items

    @staticmethod
    def from_dict(obj: Dict[str, Any]) -> 'ExpMasterConfig':
        return ExpMasterConfig(
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
            random_seeds=obj['random_seeds'],
            n_steps=obj['n_steps'],
            eval_steps=obj['eval_steps'],
            n_jobs=obj['n_jobs'],
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
            'random_seeds': self.random_seeds,
            'n_steps': self.n_steps,
            'eval_steps': self.eval_steps,
            'n_jobs': self.n_jobs,
            'data': self.data.value,
            'filter_seq_list': self.filter_seq_list,
            'use_cpu': self.use_cpu,
            'eval_metrics': self.eval_metrics,
            'ensemble_top_k': self.ensemble_top_k
        }

    def to_exp_config_for_single_trial(self) -> List[ExpConfig]:
        base_config = self.to_dict()
        base_config.pop('random_seeds')
        base_config.pop('n_jobs')

        seed_list = generate_nested_cv_seeds(self.random_seeds)
        configs = []
        for seed_dict in seed_list:
            config = ExpConfig.from_dict({
                'random_seed': seed_dict['base_seed'],
                'outer_cv_seed': seed_dict['outer_cv_seed'],
                'inner_cv_seed': seed_dict['inner_cv_seed'],
                **base_config
            })
            configs.append(config)
        return configs
