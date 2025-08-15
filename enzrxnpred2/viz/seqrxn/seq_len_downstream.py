import pandas as pd
from matplotlib import pyplot as plt

from enzrxnpred2.core.default_path import DefaultPath
from enzrxnpred2.data.original_enz_activity_dense_screen_datasource import EnzActivityScreeningDataset, \
    EnzActivityScreeningDatasource


def hist(data, title: str):
    plt.figure(figsize=(12, 6))
    plt.hist(data, bins=40, color='green', edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main():
    df_train = pd.read_pickle(DefaultPath().data_original_kcat_prediction_kcat_data_splits / "train_df_kcat.pkl")
    df_test = pd.read_pickle(DefaultPath().data_original_kcat_prediction_kcat_data_splits / "test_df_kcat.pkl")
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    seq_len = [len(seq) for seq in df['Sequence'].tolist()]
    print(max(seq_len))
    hist(seq_len, 'turnup')


def main2():
    source = EnzActivityScreeningDatasource(DefaultPath().data_original_dense_screen_processed)
    for dataset in EnzActivityScreeningDataset:
        df = source.load_binary_dataset(dataset)
        seq_len = [len(seq) for seq in df['SEQ'].tolist()]
        print(max(seq_len))
        hist(seq_len, dataset.value)


if __name__ == '__main__':
    main()
    main2()
