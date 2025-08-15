import pandas as pd
from matplotlib import pyplot as plt

from enzrxnpred2.core.default_path import DefaultPath
from enzrxnpred2.data.original_enz_activity_dense_screen_datasource import EnzActivityScreeningDataset, \
    EnzActivityScreeningDatasource


def _vis_cross_tab(df: pd.DataFrame):
    df['Total'] = df[0] + df[1]
    df['Ratio_0'] = df[0] / df['Total']
    df['Ratio_1'] = df[1] / df['Total']

    fig, ax = plt.subplots()

    df[['Ratio_0', 'Ratio_1']].plot(kind='bar', stacked=True, ax=ax, figsize=(18, 6))

    ax.set_title('Proportion of Activity 0 and 1 across SEQs')
    ax.set_xlabel('SEQ')
    ax.set_ylabel('Proportion')
    ax.legend(['Activity 0', 'Activity 1'], loc='upper right')

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def check_cross_table():
    df, _ = EnzActivityScreeningDatasource(DefaultPath().data_original_dense_screen_processed).load_binary_dataset(
        EnzActivityScreeningDataset.HALOGENASE_NABR)

    cross_tab_a = pd.crosstab(df['SEQ'], df['Activity'])
    print(cross_tab_a)
    _vis_cross_tab(cross_tab_a)

    cross_tab_b = pd.crosstab(df['SUBSTRATES'], df['Activity'])
    _vis_cross_tab(cross_tab_b)


if __name__ == '__main__':
    check_cross_table()
