from enzrxnpred2.core.default_path import DefaultPath
from enzrxnpred2.data.original_enz_activity_dense_screen_datasource import EnzActivityScreeningDatasource, \
    EnzActivityScreeningDataset
from enzrxnpred2.extension.bio_ext import calculate_crc64
from enzrxnpred2.preprocess.cdhit2.check_overlap2 import filter_out_overlap_sequences
from enzrxnpred2.preprocess.cdhit2.create_cdhit_input import CPI_CDHIT_DATASETS, CDHitTargetDataset

mapping = {
    CDHitTargetDataset.DUF: EnzActivityScreeningDataset.DUF,
    CDHitTargetDataset.ESTERASE: EnzActivityScreeningDataset.ESTERASE,
    CDHitTargetDataset.GT_ACCEPTORS_CHIRAL: EnzActivityScreeningDataset.GT_ACCEPTORS_CHIRAL,
    CDHitTargetDataset.HALOGENASE_NABR: EnzActivityScreeningDataset.HALOGENASE_NABR,
    CDHitTargetDataset.OLEA: EnzActivityScreeningDataset.OLEA,
    CDHitTargetDataset.PHOSPHATASE_CHIRAL: EnzActivityScreeningDataset.PHOSPHATASE_CHIRAL
}


def create_filtered_cpi_dataset():
    source = EnzActivityScreeningDatasource(DefaultPath().data_original_dense_screen_processed)

    for cdhit_dataset in CPI_CDHIT_DATASETS:
        df = source.load_binary_dataset(mapping[cdhit_dataset])
        df['seq_hash'] = df['SEQ'].apply(calculate_crc64)
        seq_names = filter_out_overlap_sequences(key=f"enzsrp_full_train__esp__{cdhit_dataset.value}",
                                                 filtering_target=cdhit_dataset.value,
                                                 similarity_threshold=60)
        df = df[df['seq_hash'].isin(seq_names)]
        df = df.drop(columns=['seq_hash'])
        parent_dir = DefaultPath().data_dataset_processed / 'cpi'
        parent_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(parent_dir / f"{cdhit_dataset.value}_filtered_binary.csv", index=False)


if __name__ == '__main__':
    create_filtered_cpi_dataset()
