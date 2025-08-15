import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold

from enzrxnpred2.core.default_path import DefaultPath
from enzrxnpred2.data.original_enz_activity_dense_screen_datasource import EnzActivityScreeningDatasource, \
    EnzActivityScreeningDataset


def main():
    df = EnzActivityScreeningDatasource(DefaultPath().data_original_dense_screen_processed).load_binary_dataset(
        EnzActivityScreeningDataset.OLEA)

    unique_seqs = df['SEQ'].unique()

    outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)

    outer_scores_roc = []
    outer_scores_pr = []
    best_alpha_per_fold = []

    for i_fold_loop, (train_idx, test_idx) in enumerate(outer_cv.split(unique_seqs)):
        fold_idx = i_fold_loop + 1
        seq_train, seq_test = unique_seqs[train_idx], unique_seqs[test_idx]
        df_train, df_test = df[df['SEQ'].isin(seq_train)], df[df['SEQ'].isin(seq_test)]

        y_test = df_test["Activity"].to_list()

        n_active_entries = df_test[df_test['Activity'] == 1].shape[0]
        ratio_ones = n_active_entries / df_test.shape[0]
        print(f"Active ratio (Fold {fold_idx}): {ratio_ones:.2f} ({n_active_entries} / {df_test.shape[0]})")

        activity_sum = df_train.groupby('SUBSTRATES')['Activity'].sum().reset_index(name='Activity_Sum')

        max_activity_sum = activity_sum['Activity_Sum'].max()
        activity_sum['Scaled_Activity_Sum'] = activity_sum['Activity_Sum'] / max_activity_sum

        df_test = df_test.merge(activity_sum[['SUBSTRATES', 'Scaled_Activity_Sum']], on='SUBSTRATES', how='left')

        y_score = df_test["Scaled_Activity_Sum"].tolist()

        roc_auc = roc_auc_score(y_test, y_score)
        pr_auc = average_precision_score(y_test, y_score)

        outer_scores_roc.append(roc_auc)
        outer_scores_pr.append(pr_auc)

    mean_roc_auc = np.mean(outer_scores_roc)
    std_roc_auc = np.std(outer_scores_roc)
    mean_pr_auc = np.mean(outer_scores_pr)
    std_pr_auc = np.std(outer_scores_pr)
    # mean_recall = np.mean(outer_scores_recall)
    # std_recall = np.std(outer_scores_recall)
    # mean_precision = np.mean(outer_scores_precision)
    # std_precision = np.std(outer_scores_precision)

    print(f'Nested CV ROC-AUC: {mean_roc_auc:.4f} ± {std_roc_auc:.4f}')
    print(f'Nested CV PR-AUC: {mean_pr_auc:.4f} ± {std_pr_auc:.4f}')
    # print(f'Nested CV Recall: {mean_recall:.4f} ± {std_recall:.4f}')
    # print(f'Nested CV Precision: {mean_precision:.4f} ± {std_precision:.4f}')
    print(best_alpha_per_fold)


if __name__ == '__main__':
    main()
