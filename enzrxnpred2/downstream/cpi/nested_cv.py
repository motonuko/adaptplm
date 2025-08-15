import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

from enzrxnpred2.data.original_enz_activity_dense_screen_datasource import EnzActivityScreeningDatasource
from enzrxnpred2.downstream.cpi.domain.exp_config import ExpConfig
from enzrxnpred2.downstream.cpi.domain.exp_master_config import ExpMasterConfig
from enzrxnpred2.downstream.cpi.ml.cls_metric import construct_metric_instance, metric_to_key, key_metric_map
from enzrxnpred2.downstream.cpi.ml.enzyme_feature import EsmFeatureConstructor
from enzrxnpred2.downstream.cpi.ml.ml_pipeline import FNNModelPipeline
from enzrxnpred2.extension.bio_ext import calculate_crc64
from enzrxnpred2.extension.optuna_ext import get_top_trials
from enzrxnpred2.extension.rdkit_ext import compute_morgan_fingerprint_as_array
from enzrxnpred2.extension.seed import set_random_seed
from enzrxnpred2.extension.torch_ext import get_device

logger = logging.getLogger(__name__)


def train_and_evaluate_fold(fold_idx, c: ExpConfig, m_x_train, m_y_train, label_train, df_test, output_dir: Path):
    fold_seed = c.random_seed * fold_idx
    set_random_seed(fold_seed)

    # np.arrays might be converted to memmap if using 'loky' backend.
    x_train = tuple(np.array(mem) for mem in m_x_train)
    y_train = np.array(m_y_train)

    x_test = (np.array(df_test["SEQ_FEAT"].tolist()), np.array(df_test["MOL_FEAT"].tolist()))
    y_test = np.array(df_test["Activity"].to_list())
    label_test = df_test["SEQ_CHECKSUM"].to_list()

    # print(f"test size {len(y_test)}, train-val size {len(y_train)}")
    if c.model == 'ridge':
        raise ValueError('ridge model disabled')
        # pipeline = RidgeModelPipeline(c)
    elif c.model == "fnn":
        pipeline = FNNModelPipeline(c)
    else:
        raise ValueError()
    seed_fold_key = f"seed_{c.random_seed}_fold_{fold_idx}"
    training_log_out_dir = output_dir / seed_fold_key
    training_log_out_dir.mkdir(parents=True, exist_ok=True)
    study, recorder, y_score, final_model_params = pipeline.run_whole_steps(x_train, y_train,
                                                                            label_train, x_test,
                                                                            seed=fold_seed,
                                                                            log_dir=training_log_out_dir,
                                                                            ensemble_top_k=c.ensemble_top_k)

    sorted_seq = list(sorted(set(label_test)))
    best_params = [trial.params for trial in get_top_trials(study, n=c.ensemble_top_k)]
    result = {"fold_idx": fold_idx, "best_params": best_params, 'test_seq_crc64': sorted_seq,
              'final_model_params': final_model_params}
    for metric in [construct_metric_instance(metric) for metric in c.eval_metrics]:
        result[metric_to_key(metric)] = metric.compute(y_test, y_score, label_test)
    optim_results = [{"fold_idx": fold_idx, **rcord} for rcord in recorder.results]
    pred = df_test.copy()
    pred['y_score'] = y_score
    pred = pred[['SEQ_CHECKSUM', 'SUBSTRATES', 'Activity', 'y_score']]
    pred.to_csv(training_log_out_dir / f"pred_{seed_fold_key}.csv", index=False)
    return result, optim_results


def one_nested_cv_trial(c: ExpConfig, df, n_jobs, output_dir: Path):
    # outer fold cv dataset
    unique_seqs = df['SEQ_CHECKSUM'].unique()
    outer_cv = KFold(n_splits=c.n_outer_folds, shuffle=True, random_state=c.outer_cv_seed)
    outer_cv_datasets = []
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(unique_seqs)):
        seq_train, seq_test = unique_seqs[train_idx], unique_seqs[test_idx]
        assert np.unique(np.concatenate((seq_train, seq_test))).size == unique_seqs.size
        df_train = df[df['SEQ_CHECKSUM'].isin(seq_train)]
        df_test = df[df['SEQ_CHECKSUM'].isin(seq_test)]
        # TODO: pass df_train directly (like df_test)
        x_train = (np.array(df_train["SEQ_FEAT"].tolist()), np.array(df_train["MOL_FEAT"].tolist()))
        y_train = np.array(df_train["Activity"].to_list())
        label_train = df_train["SEQ_CHECKSUM"].to_list()
        outer_cv_datasets.append((fold_idx, c, x_train, y_train, label_train, df_test, output_dir))

    # NOTE: loky -> process based backend
    # NOTE: Assumes training multiple small models on a single GPU.
    # NOTE: Parallel processing optuna optimization might change results due to execution order variations (?).
    # Currently, it uses single job.
    result = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(train_and_evaluate_fold)(*data) for data in outer_cv_datasets)
    fold_results, optim_results = zip(*result)
    fold_results, optim_results = list(fold_results), list(i for ii in optim_results for i in ii)

    scores = [{key: value} for fold_result in fold_results for key, value in fold_result.items() if
              key in key_metric_map.keys()]
    df_scores = pd.DataFrame(scores)
    means = df_scores.mean()
    stds = df_scores.std()
    final_score = {}
    for column in df_scores.columns:
        logger.info(f"{column}: {means[column]:.4f} ± {stds[column]:.4f}")
        final_score[column] = {"mean": means[column], "std": stds[column]}

    output = {"config": c.to_dict(),
              "averaged_fold_results": final_score,
              "fold_result": fold_results}
    return output, optim_results


def run_nested_cv_on_enz_activity_cls(exp_config: Path, output_parent_dir: Path, dense_screen_dir: Path,
                                      additional_dense_source_dir: Optional[Path] = None):
    with open(exp_config, 'r') as f:
        data: Dict[str, Any] = json.load(f)
        entire_config = ExpMasterConfig.from_dict(data)

    filter_seq_hashes = entire_config.filter_seq_list

    df = EnzActivityScreeningDatasource(dense_screen_dir, additional_dense_source_dir).load_binary_dataset(
        entire_config.data)
    df['SEQ_CHECKSUM'] = df['SEQ'].apply(calculate_crc64)
    before_size = len(df)

    # filtering
    assert all(
        col in df['SEQ_CHECKSUM'].values for col in filter_seq_hashes), "some of filter items does not exist in data"
    df = df[~df['SEQ_CHECKSUM'].isin(filter_seq_hashes)]
    if len(filter_seq_hashes) != 0:
        assert len(df) < before_size, "No items were filtered"

    # build features
    device = get_device(entire_config.use_cpu)
    feat = EsmFeatureConstructor(config=entire_config.protein_feat_config, dataset=entire_config.data, device=device)
    df["SEQ_FEAT"] = df['SEQ'].apply(feat.construct_feature)
    feat.clear()
    del feat
    n_bits = entire_config.molecule_feat_config.params['length']
    df["MOL_FEAT"] = df['SUBSTRATES'].apply(lambda x: compute_morgan_fingerprint_as_array(x, n_bits=n_bits))
    # df['FEAT'] = [np.concatenate((x, y)) for x, y in zip(df['SEQ_FEAT'], df['MOL_FEAT'])]

    output_dir = output_parent_dir / entire_config.data.value / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: Wait for all inner fold evaluation finish; allows easy restart with different seeds though adds some overhead.
    for config in entire_config.to_exp_config_for_single_trial():
        output, optim_results = one_nested_cv_trial(config, df, entire_config.n_jobs, output_dir)
        with open(output_dir / f"result_seed_{config.random_seed}.json", "w") as json_file:
            json.dump(output, json_file, indent=2)
        pd.DataFrame(optim_results).to_csv(output_dir / f"optim_results_seed_{config.random_seed}.csv", index=False)
