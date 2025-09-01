from adaptplm.core.default_path import DefaultPath
from adaptplm.data.turnup_datasource import load_kcat_df
from adaptplm.extension.bio_ext import calculate_crc64
from adaptplm.preprocess.cdhit2.check_overlap2 import filter_out_overlap_sequences


def create_filtered_kcat_test_sequence_list():
    df = load_kcat_df(DefaultPath().original_kcat_test_pkl)
    df['seq_hash'] = df['Sequence'].apply(calculate_crc64)
    seq_names = filter_out_overlap_sequences(key='enzsrp_full_train__esp__kcat', filtering_target='kcat',
                                             similarity_threshold=60)
    df = df[df['seq_hash'].isin(seq_names)]
    df = df.drop(columns=['seq_hash'])
    parent_dir = DefaultPath().build / 'kcat'
    parent_dir.mkdir(parents=True, exist_ok=True)
    df[['Sequence ID']].to_csv(parent_dir / 'test_df_kcat_filtered_sequence_ids.csv', index=False)


if __name__ == '__main__':
    create_filtered_kcat_test_sequence_list()
