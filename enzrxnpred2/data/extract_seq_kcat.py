import pandas as pd

from enzrxnpred2.core.default_path import DefaultPath

# Max token length of ESP model is 1022 (including special tokens)
# https://github.com/AlexanderKroll/ESP/blob/main/notebooks_and_code/extract.py#L99
CUTOFF = 1020


def extract_seq_kcat():
    output_dir = DefaultPath().data_dataset_processed / 'kcat'
    output_dir.mkdir(exist_ok=True, parents=True)
    df_train = pd.read_pickle(DefaultPath().data_original_kcat_prediction_kcat_data_splits / "train_df_kcat.pkl")
    df_test = pd.read_pickle(DefaultPath().data_original_kcat_prediction_kcat_data_splits / "test_df_kcat.pkl")
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    df['trimmed_seq'] = df['Sequence'].apply(lambda x: x[:CUTOFF])
    df = df.drop_duplicates(subset='trimmed_seq', keep='first')
    df[['trimmed_seq']].to_csv(output_dir / 'kcat_sequences.txt', index=False, header=False)


if __name__ == '__main__':
    extract_seq_kcat()
