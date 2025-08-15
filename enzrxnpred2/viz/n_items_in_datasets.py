import pandas as pd

from enzrxnpred2.core.default_path import DefaultPath
from enzrxnpred2.data.esp_datasource import load_esp_df


def main():
    result = []

    df_unienz_full = pd.read_csv(DefaultPath().data_dataset_processed.joinpath('enzsrp_full_cleaned.csv'))
    # df_unienz_filtered = pd.read_csv(DefaultPath().data_dataset_processed.joinpath('unienz_filtered.csv'))
    result.append({'name': 'enzsrp_full',
                   'n_total': len(df_unienz_full),
                   'n_prot': df_unienz_full['sequence'].nunique(),
                   'n_rxn_sub': df_unienz_full['rxn'].nunique()})
    # result.append({'name': 'unienz_filtered',
    #                'n_total': len(df_unienz_filtered),
    #                'n_prot': df_unienz_filtered['sequence'].nunique(),
    #                'n_rxn': df_unienz_filtered['rxn'].nunique()})
    # dense_source = EnzActivityScreeningDatasource(DefaultPath().data_original_dense_screen_processed)
    # for dataset in datasets_used_in_paper:
    #     df_dense, _ = dense_source.load_binary_dataset(dataset)
    #     result.append({'name': dataset.short_name_for_original_files,
    #                    'n_total': len(df_dense),
    #                    'n_prot': df_dense['SEQ'].nunique(),
    #                    'n_rxn': df_dense['SUBSTRATES'].nunique()})

    df_esp = load_esp_df(DefaultPath().original_esp_fine_tuning_pkl)
    result.append({'name': 'esp_(fine-tuning)',
                   'n_total': len(df_esp),
                   'n_prot': df_esp['Sequence'].nunique(),
                   'n_rxn_sub': df_esp['molecule ID'].nunique()})

    count_table = pd.DataFrame(result)
    # count_table.to_latex('count.tex', index=False)
    print(count_table)


if __name__ == '__main__':
    main()
