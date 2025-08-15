from functools import partial
from pathlib import Path

import numpy as np
import optuna
import torch
from optuna import Trial
from optuna.samplers import TPESampler
from optuna.study import StudyDirection
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold

from adaptplm.downstream.cpi.domain.exp_config import ExpConfig, IntOptimParams, FloatOptimParams, \
    CategoricalOptimParams
from adaptplm.downstream.cpi.ml.model2 import FeedForwardNN2
from adaptplm.downstream.cpi.ml.training_loop import train_and_eval, train_loop
from adaptplm.extension.optuna_ext import get_top_trials
from adaptplm.extension.torch_ext import get_device


class OptimRecorder:
    def __init__(self):
        self.results = []

    def record(self, result: dict):
        self.results.append(result)


def get_optim_metric(optim_metric: str):
    if optim_metric == "roc_auc":
        return roc_auc_score
    elif optim_metric == "pr_auc":
        return average_precision_score
    raise ValueError("unexpected arguments")


def get_study_direction(optim_metric: str):
    if optim_metric == "roc_auc" or optim_metric == "pr_auc":
        return StudyDirection.MAXIMIZE
    raise ValueError("unexpected arguments")


class ModelPipeline:
    def __init__(self, config: ExpConfig):
        self.config = config

    def _construct_model(self, **params):
        raise NotImplementedError()

    @staticmethod
    def _objective(trial: Trial, x_train, y_train, label_train, c: ExpConfig, recorder: OptimRecorder,
                   model_constructor):
        raise NotImplementedError()

    def optimize_hyper_parameter(self, x_train, y_train, label_train, seed: int):
        recorder = OptimRecorder()
        objective = self._objective
        objective = partial(objective, x_train=x_train, y_train=y_train, label_train=label_train, c=self.config,
                            recorder=recorder, model_constructor=self._construct_model)
        sampler = TPESampler(seed=seed, n_startup_trials=self.config.optuna_n_startup_trials)
        study = optuna.create_study(direction=get_study_direction(self.config.optim_metric), sampler=sampler)
        study.optimize(objective, n_trials=self.config.optuna_n_trials)
        return study, recorder

    def train_final_model(self, best_params, x_train, y_train, steps: int, log_dir: Path):
        raise NotImplementedError()

    def predict_test_set(self, final_model, x_test):
        raise NotImplementedError()

    def run_whole_steps(self, x_train, y_train, label_train, x_test, seed: int, log_dir: Path, ensemble_top_k: int):
        study, recorder = self.optimize_hyper_parameter(x_train, y_train, label_train, seed)
        top_trials = get_top_trials(study, n=ensemble_top_k)
        n_trainable_params_list = []
        best_step_list = []
        final_models = []
        for trial in top_trials:
            best_step = recorder.results[trial.number].get('best_step', None)
            final_model = self.train_final_model(trial.params, x_train, y_train, best_step, log_dir)
            n_trainable_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
            n_trainable_params_list.append(n_trainable_params)
            best_step_list.append(best_step)
            final_models.append(final_model)
        y_score = self.predict_test_set(final_models, x_test)
        final_model_params = {'topk_final_models_n_trainable_params': n_trainable_params_list,
                              'topk_final_models_n_training_steps': best_step_list}
        return study, recorder, y_score, final_model_params


# class RidgeModelPipeline(ModelPipeline):
#
#     def __init__(self, config: ExpConfig):
#         super().__init__(config)
#
#     def _construct_model(self, **params):
#         return RidgeClassifier(class_weight='balanced', **params)
#
#     @staticmethod
#     def _objective(trial: Trial, x_train, y_train, c: ExpConfig, recorder: OptimRecorder, model_constructor):
#         x_train_concatenated = np.concatenate(x_train, axis=1)
#         suggested_params = {param.name: trial.suggest_float(param.name, param.min, param.max) for param in
#                             c.optim_params}
#         clf = model_constructor(**suggested_params)
#         inner_cv = KFold(n_splits=c.n_inner_folds, shuffle=True, random_state=c.inner_cv_seed)
#         scores = cross_val_score(clf, x_train_concatenated, y_train, cv=inner_cv,
#                                  scoring=make_scorer(get_optim_metric(c.optim_metric)))
#         result = suggested_params.copy()
#         result.update({"score": np.mean(scores), "trial": trial.number})
#         recorder.record(result)
#         return np.mean(scores)
#
#     def train_final_model(self, best_params, x_train, y_train):
#         x_train_concatenated = np.concatenate(x_train, axis=1)  # use whole training set
#         final_model = self._construct_model(**best_params)
#         final_model.fit(x_train_concatenated, y_train)
#         return final_model
#
#     def predict_test_set(self, final_model, x_test):
#         x_test_concatenated = np.concatenate(x_test, axis=1)
#         y_score = final_model.decision_function(x_test_concatenated)
#         return y_score
#

class FNNModelPipeline(ModelPipeline):

    def __init__(self, config: ExpConfig):
        super().__init__(config)

    def _construct_model(self, **params):
        return FeedForwardNN2(**params)

    @staticmethod
    def _objective(trial: Trial, x_train, y_train, label_train, c: ExpConfig, recorder: OptimRecorder,
                   model_constructor):
        suggested = {}
        for param in c.optim_params:
            if isinstance(param, IntOptimParams):
                suggested[param.name] = trial.suggest_int(param.name, low=param.min, high=param.max, step=param.step)
            elif isinstance(param, FloatOptimParams):
                suggested[param.name] = trial.suggest_float(param.name, param.min, param.max)
            elif isinstance(param, CategoricalOptimParams):
                suggested[param.name] = trial.suggest_categorical(param.name, param.candidates)
            else:
                raise ValueError('not supported other hyper param types')

        x1_train_tensor = torch.FloatTensor(x_train[0])
        x2_train_tensor = torch.FloatTensor(x_train[1])
        y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)

        device = get_device(c.use_cpu)
        inner_cv = KFold(n_splits=c.n_inner_folds, shuffle=True, random_state=c.inner_cv_seed)
        scores = []
        # =============== enzyme discovery scenario splitting ==================
        # unique_seqs = np.array(list(sorted(set(label_train))))
        # label_array = np.array(label_train)
        # for fold_idx, (train_seq_idx, val_seq_idx) in enumerate(inner_cv.split(unique_seqs)):
        #     train_seqs = unique_seqs[train_seq_idx]
        #     val_seqs = unique_seqs[val_seq_idx]
        #     train_idx = np.where(np.isin(label_array, train_seqs))[0]
        #     val_idx = np.where(np.isin(label_array, val_seqs))[0]
        # ====================================
        score_history_per_fold = []
        for fold_idx, (train_idx, val_idx) in enumerate(inner_cv.split(x1_train_tensor)):
            assert len(set(train_idx) & set(val_idx)) == 0
            assert len(train_idx) + len(val_idx) == len(x1_train_tensor)
            # print(f"inner val size {len(val_idx)}, inner train size {len(train_idx)}")
            x1_inner_train, x1_val = x1_train_tensor[train_idx], x1_train_tensor[val_idx]
            x2_inner_train, x2_val = x2_train_tensor[train_idx], x2_train_tensor[val_idx]
            y_inner_train, y_val = y_train_tensor[train_idx], y_train_tensor[val_idx]

            loop_param_keys = {'learning_rate', 'weight_decay', 'batch_size'}
            model_params = {k: v for k, v in suggested.items() if k not in loop_param_keys}
            loop_params = {k: v for k, v in suggested.items() if k in loop_param_keys}
            model = model_constructor(prot_input_dim=x1_inner_train.shape[1],
                                      mol_input_dim=x2_inner_train.shape[1],
                                      **model_params)
            best_score, _, _, score_history = train_and_eval(model, x1_inner_train, x2_inner_train,
                                                             y_inner_train, x1_val, x2_val, y_val, n_steps=c.n_steps,
                                                             eval_steps=c.eval_steps, device=device,
                                                             scoring=get_optim_metric(c.optim_metric),
                                                             **loop_params)
            score_history_per_fold.append(np.array(score_history))
            scores.append(best_score)
        final_score = np.mean(scores)

        avg_score_history = np.mean(np.array(score_history_per_fold), axis=0)
        # assert len(avg_val_loss_history) == c.n_steps // 50, (len(avg_val_loss_history), c.n_steps // 50)
        # mean_best_steps = (np.argmin(avg_score_history) + 1) * c.eval_steps  # NOTE: if target is loss
        mean_best_steps = int((np.argmax(avg_score_history) + 1) * c.eval_steps)
        # print(mean_best_steps)

        result = suggested.copy()
        result.update({"score": final_score, "trial": trial.number, "best_step": mean_best_steps})
        recorder.record(result)
        return final_score

    def train_final_model(self, best_params, x_train, y_train, steps, log_dir):
        x1_train_tensor = torch.FloatTensor(x_train[0])
        x2_train_tensor = torch.FloatTensor(x_train[1])
        y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
        # print(f"final model training size: {len(y_train_resampled)}, early stopping data size: {len(y_val_resampled)}")

        loop_param_keys = {'learning_rate', 'weight_decay', 'batch_size'}
        model_params = {k: v for k, v in best_params.items() if k not in loop_param_keys}
        loop_params = {k: v for k, v in best_params.items() if k in loop_param_keys}
        device = get_device(self.config.use_cpu)
        model = self._construct_model(prot_input_dim=x1_train_tensor.shape[1], mol_input_dim=x2_train_tensor.shape[1],
                                      **model_params)
        final_model = train_loop(model, x1_train_tensor, x2_train_tensor, y_train_tensor, n_steps=steps, **loop_params,
                                 device=device, log_dir=log_dir)
        return final_model

    def predict_test_set(self, final_models, x_test):
        x1_test_tensor = torch.FloatTensor(x_test[0])
        x2_test_tensor = torch.FloatTensor(x_test[1])
        device = get_device(self.config.use_cpu)
        x1_test_tensor = x1_test_tensor.to(device)
        x2_test_tensor = x2_test_tensor.to(device)
        y_scores = []
        for final_model in final_models:
            final_model.to(device)
            final_model.eval()
            with torch.no_grad():
                output = final_model(x1_test_tensor, x2_test_tensor)
                y_score = output['logit'].squeeze().cpu().numpy()
                y_scores.append(y_score)
            final_model.to('cpu')
        mean = np.mean(y_scores, axis=0)
        sigmoid_out = torch.sigmoid(torch.Tensor(mean))
        return sigmoid_out
