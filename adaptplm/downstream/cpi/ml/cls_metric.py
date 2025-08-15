import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


class MetricAbstract:
    name = None

    def __init__(self):
        self.results = []

    def compute(self, y_test, y_score, labels):
        pass


class ROCAUCMetric(MetricAbstract):
    name = "ROC-AUC"
    description = "ROC-AUC score"

    def compute(self, y_test, y_score, labels):
        return roc_auc_score(y_test, y_score)


class PRAUCMetric(MetricAbstract):
    name = 'PR-AUC'
    description = "PR-AUC score"

    def compute(self, y_test, y_score, labels):
        return average_precision_score(y_test, y_score)


class ROCAUCAveragePerSeqMetric(MetricAbstract):
    name = 'Average ROC-AUC (single task)'

    def compute(self, y_test, y_score, labels):
        df = pd.DataFrame({"y_test": y_test, "y_score": y_score, "label": labels})
        performance_per_label = []
        for group_name, group_df in df.groupby("label"):
            grouped_roc_auc = roc_auc_score(group_df["y_test"].tolist(), group_df["y_score"].tolist())
            performance_per_label.append(grouped_roc_auc)
        return np.mean(performance_per_label)


class PRAUCAveragePerSeqMetric(MetricAbstract):
    name = 'Average PR-AUC (single task)'

    def compute(self, y_test, y_score, labels):
        df = pd.DataFrame({"y_test": y_test, "y_score": y_score, "label": labels})
        performance_per_label = []
        for group_name, group_df in df.groupby("label"):
            grouped_roc_auc = average_precision_score(group_df["y_test"].tolist(), group_df["y_score"].tolist())
            performance_per_label.append(grouped_roc_auc)
        return np.mean(performance_per_label)


key_metric_map = {"roc_auc": ROCAUCMetric,
                  "pr_auc": PRAUCMetric,
                  # "avg_roc_auc_single_task": ROCAUCAveragePerSeqMetric,
                  # "avg_pr_auc_single_task": PRAUCAveragePerSeqMetric
                  }


def metric_to_key(metric: MetricAbstract):
    hit_key = [key for key, value in key_metric_map.items() if isinstance(metric, value)]
    return hit_key[0]


def construct_metric_instance(score_based_metric: str):
    return key_metric_map[score_based_metric]()
