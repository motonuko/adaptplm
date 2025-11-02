import pandas as pd

from adaptplm.core.default_path import DefaultPath
from adaptplm.data.esp_datasource import load_esp_df


def main():
    result = []
    #
    df_enzsrp = pd.read_csv(DefaultPath().data_dataset_processed.joinpath('enzsrp_full_cleaned.csv'))
    result.append({'name': 'enzsrp_full',
                   'n_total': len(df_enzsrp),
                   'n_prot': df_enzsrp['sequence'].nunique(),
                   'n_rxn_sub': df_enzsrp['rxn'].nunique()})

    df_enzsrp_train = pd.read_csv(DefaultPath().data_dataset_processed.joinpath('enzsrp_full_cleaned', 'enzsrp_full_cleaned_train.csv'))
    result.append({'name': 'enzsrp_full (train)',
                   'n_total': len(df_enzsrp_train),
                   'n_prot': df_enzsrp_train['sequence'].nunique(),
                   'n_rxn_sub': df_enzsrp_train['rxn'].nunique()})

    # ======== ESP pretrain training set ========
    # https://github.com/AlexanderKroll/ESP/blob/main/notebooks_and_code/training_ESM1b_taskspecific.py
    #
    df_esp = load_esp_df(DefaultPath().original_esp_fine_tuning_pkl)  # ESM1b_training/train_data_ESM_training.pkl
    result.append({'name': 'esp (fine-tuning)',
                   'n_total': len(df_esp),
                   'n_prot': df_esp['Sequence'].nunique(),
                   'n_rxn_sub': df_esp['molecule ID'].nunique()})

    # ========= ESP classification dataset ========
    # You can find the preprocessing code in the following notebook.
    # https://github.com/AlexanderKroll/ESP/blob/main/notebooks_and_code/2_0%20-%20Training%20gradient%20boosting%20models.ipynb
    #
    # ESP training set
    df_train = pd.read_pickle(DefaultPath().original_esp_data_dir.joinpath("splits", "df_train_with_ESM1b_ts_GNN.pkl"))
    org = len(df_train)
    df_train = df_train.loc[df_train["ESM1b"].apply(lambda x: len(x) > 0)]
    df_train = df_train.loc[df_train["type"] != "engqvist"]
    df_train = df_train.loc[df_train["GNN rep"].apply(lambda x: len(x) > 0)]
    df_train.reset_index(inplace=True, drop=True)
    print(f"original {org}, after: {len(df_train)}")
    # ESP test set
    df_test = pd.read_pickle(DefaultPath().original_esp_data_dir.joinpath("splits", "df_test_with_ESM1b_ts_GNN.pkl"))
    org = len(df_test)
    df_test = df_test.loc[df_test["ESM1b"].apply(lambda x: len(x) > 0)]
    df_test = df_test.loc[df_test["type"] != "engqvist"]
    df_test = df_test.loc[df_test["GNN rep"].apply(
        lambda x: len(x) > 0)]  # NOTE: if removing this process, train + test = 69,365 (same as the paper)
    print(f"original {org}, after: {len(df_test)}")
    df_test.reset_index(inplace=True, drop=True)

    df_cls = pd.concat([df_train, df_test], ignore_index=True)
    df_cls['ESM1b'] = df_cls['ESM1b'].apply(lambda x: tuple(x))
    df_cls['GNN rep'] = df_cls['GNN rep'].apply(lambda x: tuple(x))
    result.append({'name': 'esp_(cls)',
                   'n_total': len(df_cls),
                   'n_prot': df_cls['ESM1b'].nunique(),
                   'n_rxn_sub': df_cls['ECFP'].nunique()})

    count_table = pd.DataFrame(result)
    # count_table.to_latex('count.tex', index=False)
    print(count_table)


if __name__ == '__main__':
    main()
