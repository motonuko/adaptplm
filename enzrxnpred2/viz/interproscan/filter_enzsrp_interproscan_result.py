from enzrxnpred2.core.default_path import DefaultPath
from enzrxnpred2.data.load_interproscan_tsv import load_interproscan_result_tsv
from enzrxnpred2.data.original_enz_activity_dense_screen_datasource import datasets_used_in_paper


# Make a slimmed version for sharing data
def filter_enzsrp_interproscan_result():
    base_path = DefaultPath().data_dataset_dir / 'interproscan-5.75-106.0'
    ipas_in_screening_dataset_results = []
    for dataset in datasets_used_in_paper:
        df = load_interproscan_result_tsv(base_path / f"{dataset.value}_input.fasta.tsv")
        ipas_in_screening_dataset_results.extend(df['InterPro Accession'].tolist())
    target_df = load_interproscan_result_tsv(base_path / 'enzsrp_full_train_input.fasta.tsv')
    filtered_df = target_df[target_df['InterPro Accession'].isin(ipas_in_screening_dataset_results)]

    filtered_df.to_csv(base_path / 'enzsrp_full_train_input.fasta_filtered.tsv', index=False, sep='\t', header=None)


if __name__ == '__main__':
    filter_enzsrp_interproscan_result()
