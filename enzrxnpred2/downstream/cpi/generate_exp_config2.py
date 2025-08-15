import json
from typing import Optional

from adaptplm.core.constants import ESM1B_T33_650M_UR50S
from adaptplm.core.default_path import DefaultPath
from adaptplm.core.package_version import get_package_version
from adaptplm.data.original_enz_activity_dense_screen_datasource import EnzActivityScreeningDataset
from adaptplm.downstream.cpi.domain.exp_config import ProteinFeatConfig, MoleculeFeatConfig, FloatOptimParams, \
    construct_optim_param_obj
from adaptplm.downstream.cpi.domain.exp_master_config import ExpMasterConfig
from adaptplm.downstream.cpi.domain.protein_lm_embedding import ProteinLMEmbedding, EmbeddingType


def main(model: str, dataset: EnzActivityScreeningDataset, distance: Optional[int], optim_metric,
         pooling_strategy: EmbeddingType, n_steps: int, batch_size: int,
         esm_model_name=ESM1B_T33_650M_UR50S, file_name_suffix=""):
    mol_feat_config = MoleculeFeatConfig(name="morgan", params={"length": 1024})

    if pooling_strategy == EmbeddingType.ACTIVE_SITE:
        protein_feat_config = ProteinFeatConfig(model_name=esm_model_name,
                                                embedding_config=ProteinLMEmbedding.from_embedding_type(
                                                    pooling_strategy,
                                                    distance=distance))
        file_name = f"{model}_{protein_feat_config.embedding_config.embedding_type.value}_{mol_feat_config.name}_{optim_metric}_dist{distance}.json"
    elif pooling_strategy == EmbeddingType.AVERAGE:
        protein_feat_config = ProteinFeatConfig(model_name=esm_model_name,
                                                embedding_config=ProteinLMEmbedding.from_embedding_type(
                                                    pooling_strategy))
        file_name = f"{model}_{protein_feat_config.embedding_config.embedding_type.value}_{mol_feat_config.name}_{optim_metric}.json"

    elif pooling_strategy == EmbeddingType.RXN_MLM:
        protein_feat_config = ProteinFeatConfig(model_name=esm_model_name,
                                                embedding_config=ProteinLMEmbedding.from_embedding_type(
                                                    pooling_strategy,
                                                ))
        file_name = f"{model}_{protein_feat_config.embedding_config.embedding_type.value}_{mol_feat_config.name}_{optim_metric}_{file_name_suffix}.json"
    elif pooling_strategy == EmbeddingType.PRECOMPUTED:
        protein_feat_config = ProteinFeatConfig(model_name=esm_model_name,
                                                embedding_config=ProteinLMEmbedding.from_embedding_type(
                                                    pooling_strategy,
                                                    precomputed_embeddings_npy='./build/dense_screen_esp_embeddings.npz'
                                                ))
        file_name = f"{model}_{protein_feat_config.embedding_config.embedding_type.value}_{mol_feat_config.name}_{optim_metric}.json"
    else:
        raise ValueError()

    n_trial = 60
    if model == 'ridge':
        optim_params = [
            FloatOptimParams(name='alpha', min=0.01, max=100),
        ]
    elif model == 'fnn':
        optim_params = [
            {
                "name": "n_hidden_layers",
                "type": "int",
                "min": 1,
                "max": 2,
                "step": 1
            },
            {
                "name": "hidden_dim",
                "type": "int",
                "min": 16,
                "max": 192,
                "step": 16
            },
            {
                "name": "learning_rate",
                "type": "float",
                "min": 1e-4,
                "max": 2e-2,
            },
            {
                "name": "dropout_rate",
                "type": "float",
                "min": 0.0,
                "max": 0.2
            },
            {
                "name": "weight_decay",
                "type": "float",
                "min": 1e-4,
                "max": 1e-2
            },
            {
                "name": "batch_size",
                "type": "categorical",
                "candidates": [
                    batch_size,
                ]
            }]
        optim_params = [construct_optim_param_obj(param["type"], param) for param in optim_params]
    else:
        raise ValueError('unexpected_model name')

    filter_seq_hashes = []

    n_outer_folds = 4
    config = ExpMasterConfig(
        model=model,
        protein_feat_config=protein_feat_config,
        molecule_feat_config=mol_feat_config,
        class_weight="balanced",
        n_outer_folds=n_outer_folds,
        n_inner_folds=3,
        random_seeds=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
        # random_seeds=[45, 46, 47, 48, 49, 50, 51],
        n_jobs=n_outer_folds,
        n_steps=n_steps,
        eval_steps=100,
        use_cpu=False,
        optim_metric=optim_metric,
        optim_params=optim_params,
        optuna_n_trials=n_trial,
        optuna_n_startup_trials=10,
        data=dataset,
        filter_seq_list=filter_seq_hashes,
        eval_metrics=[
            "roc_auc",
            "pr_auc"
        ],
        ensemble_top_k=5,
    )

    version = get_package_version()
    exp_key = f"{version}"

    dic = config.to_dict()
    path = DefaultPath().data_exp_configs_dir.joinpath(
        exp_key, dataset.value,
        f"{model}_{protein_feat_config.embedding_config.embedding_type.value}_{mol_feat_config.name}_{optim_metric}",
        file_name,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(dic, f, ensure_ascii=False, indent=2)


def generate_all_config():
    datasets = [
        (EnzActivityScreeningDataset.DUF_FILTERED, 64, (3, 12)),
        (EnzActivityScreeningDataset.ESTERASE_FILTERED, 512, (1, 10)),
        # (EnzActivityScreeningDataset.GT_ACCEPTORS_CHIRAL_FILTERED, 32, (3, 8)),  # too low number of sequences
        (EnzActivityScreeningDataset.HALOGENASE_NABR_FILTERED, 64, (3, 11)),
        (EnzActivityScreeningDataset.OLEA_FILTERED, 32, (1, 12)),
        (EnzActivityScreeningDataset.PHOSPHATASE_CHIRAL_FILTERED, 512, (3, 14)),
    ]
    our_models = ['250407_182715']
    our_models = ['250415_192945', '250415_221757']
    our_models = ['250420_121652']
    metric = "pr_auc"
    pred_model = "fnn"
    n_steps = 2000

    for dataset, batch_size, (dist_s, dist_e) in datasets:
        main(pred_model, dataset, None, metric,
             EmbeddingType.AVERAGE, n_steps, batch_size)
        for model in our_models:
            main(pred_model, dataset, None, metric,
                 EmbeddingType.RXN_MLM, n_steps, batch_size,
                 esm_model_name=f"./local/exp/seqrxn_encoder_train/{model}/esm",
                 file_name_suffix=model)
        main(pred_model, dataset, None, metric,
             EmbeddingType.PRECOMPUTED, n_steps, batch_size,
             esm_model_name="-")
        for i in range(dist_s, dist_e):
            main(pred_model, dataset, i, metric,
                 EmbeddingType.ACTIVE_SITE, n_steps, batch_size)


if __name__ == '__main__':
    generate_all_config()
