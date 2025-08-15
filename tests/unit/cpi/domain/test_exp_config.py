import unittest

from enzrxnpred2.core.constants import ESM2_T6_8M_UR50D
from enzrxnpred2.data.original_enz_activity_dense_screen_datasource import EnzActivityScreeningDataset
from enzrxnpred2.downstream.cpi.domain.exp_config import construct_optim_param_obj, ExpConfig, ProteinFeatConfig, \
    MoleculeFeatConfig
from enzrxnpred2.downstream.cpi.domain.protein_lm_embedding import AveragePoolEmbedding


class ExpConfigTestCase(unittest.TestCase):
    # def setUp(self):
    #     self.temp_dir = tempfile.TemporaryDirectory()
    #
    # def tearDown(self):
    #     self.temp_dir.cleanup()

    def test_something(self):
        optim_params = [
            {
                "name": "n_hidden_layers",
                "type": "int",
                "min": 1,
                "max": 3,
                "step": 1
            },
            {
                "name": "hidden_dim",
                "type": "int",
                "min": 16,
                "max": 256,
                "step": 16
            },
            {
                "name": "learning_rate",
                "type": "float",
                "min": 1e-4,
                "max": 0.02
            },
            {
                "name": "dropout_rate",
                "type": "float",
                "min": 0.1,
                "max": 0.4
            },
            {
                "name": "weight_decay",
                "type": "float",
                "min": 1e-6,
                "max": 1e-2
            },
            {
                "name": "batch_size",
                "type": "categorical",
                "candidates": [
                    32,
                    64,
                    128,
                    256
                ]
            }]
        optim_params = [construct_optim_param_obj(param["type"], param) for param in optim_params]
        config = ExpConfig(
            protein_feat_config=ProteinFeatConfig(
                model_name=ESM2_T6_8M_UR50D,
                embedding_config=AveragePoolEmbedding()
            ),
            molecule_feat_config=MoleculeFeatConfig(name="morgan", params={"length": 1024}),
            model="fnn",
            class_weight="balanced",
            n_outer_folds=5,
            n_inner_folds=3,
            optim_params=optim_params,
            optim_metric='roc_auc',
            optuna_n_trials=2,
            random_seed=42,
            inner_cv_seed=43,
            outer_cv_seed=44,
            data=EnzActivityScreeningDataset.DUF,
            filter_seq_list=[
                "733B7300E17E86A5",
                "C2695FC20C2378A8",
                "59678E29BA169383"
            ],
            n_steps=2000,
            eval_steps=50,
            optuna_n_startup_trials=10,
            use_cpu=True,
            eval_metrics=["avg_roc_auc", "avg_pr_auc"],
            ensemble_top_k=1
        )

        self.assertEqual(config, ExpConfig.from_dict(config.to_dict()))
