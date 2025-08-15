import pandas as pd
from scipy import stats

from enzrxnpred2.core.default_path import DefaultPath
from enzrxnpred2.data.original_enz_activity_dense_screen_datasource import EnzActivityScreeningDataset
from enzrxnpred2.viz.cpi_result.boxplot_utils import split_active_site, min_max, sort_key


def xx(df):
    mapping_dict = {
        'Mean': '',
        'Active Site\n(dist. 3)': '距離3Å',
        'Active Site\n(dist. 4)': '距離4Å',
        'Active Site\n(dist. 5)': '距離5Å',
        'Active Site\n(dist. 6)': '距離6Å',
        'Active Site\n(dist. 7)': '距離7Å',
        'Active Site\n(dist. 8)': '距離8Å',
        'Active Site\n(dist. 9)': '距離9Å',
        'Active Site\n(dist. 10)': '距離10Å',
        'Active Site\n(dist. 11)': '距離11Å',
        'Active Site\n(dist. 12)': '距離12Å',
        # 'Masked LM 241216_153546': 'base*',
        'Masked LM 241229_023204': '事前学習あり',
        'Masked LM 241226_164549': '事前学習なし',
        # 'Masked LM 250101_123329': '2 layers',
        # 'Masked LM 241223_212743': '0.15?',
    }
    df['label'] = df['condition'].map(mapping_dict)
    df = df.dropna(subset=['label'])

    df_mean_and_mlm, df_active_site = split_active_site(df)
    hoge = min_max(df_active_site, 'roc_auc_mean')
    huga = min_max(df_active_site, 'pr_auc_mean')
    df_duf_roc = pd.concat([df_mean_and_mlm, hoge])
    df_duf_pr = pd.concat([df_mean_and_mlm, huga])
    df_duf_roc = df_duf_roc.sort_values(by="condition", key=lambda x: x.map(sort_key)).reset_index(drop=True)
    df_duf_pr = df_duf_pr.sort_values(by="condition", key=lambda x: x.map(sort_key)).reset_index(drop=True)
    return df_duf_roc, df_duf_pr


def pair_wise_t_test(df, key='row_pr_auc'):
    results = []
    for i in range(len(df)):
        # for j in range(i + 1, len(df)):  # full pair-wise
        for j in [4]:  # no-pretraining only
            score1 = df.iloc[i][key]
            score2 = df.iloc[j][key]
            t_stat, p_val = stats.ttest_ind(score1, score2)
            significant = 'Y' if p_val < 0.05 else 'N'
            results.append({'Condition A': df.iloc[i]['condition'],
                            'Condition B': df.iloc[j]['condition'],
                            't-Value': t_stat,
                            'p-Value': p_val,
                            'diff': significant})
    results = pd.DataFrame(results)
    print(results)
    print()


def main(is_per_trial=True):
    ver = "v5"
    ver = 'thesis'
    base_path = DefaultPath().build / 'fig' / ver
    for d in [EnzActivityScreeningDataset.HALOGENASE_NABR, EnzActivityScreeningDataset.DUF,
              EnzActivityScreeningDataset.ESTERASE]:
        print(d)
        dff = pd.read_pickle(base_path / f"cv_trials_summary_{d.value}.pkl")
        _, df_pr = xx(dff)
        if is_per_trial:
            pair_wise_t_test(df_pr, "pr_auc_mean_per_trial")
            # pair_wise_t_test(df_pr, "roc_auc_mean_per_trial")
        else:
            pair_wise_t_test(df_pr, "pr_auc_per_outer_fold")


if __name__ == '__main__':
    main()
