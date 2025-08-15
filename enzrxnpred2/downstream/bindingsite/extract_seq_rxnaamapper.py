import pandas as pd

from enzrxnpred2.core.default_path import DefaultPath

# Max token length of ESP model is 1022 (including special tokens)
# https://github.com/AlexanderKroll/ESP/blob/main/notebooks_and_code/extract.py#L99
CUTOFF = 1020


def extract_seq_rxnaamapper():
    source_dir = DefaultPath().data_original_rxnaamapper_predictions
    output_dir = DefaultPath().build

    threshold = 1024
    df = pd.read_csv(source_dir / "pfam.csv")
    print(len(df))
    df = df[df['aa_sequence'].str.len() <= threshold]
    print(len(df))  # nothing has filtered
    df = df[['aa_sequence']].drop_duplicates(subset='aa_sequence', keep='first')
    df.to_csv(output_dir / f"rxnaamapper_sequences_{threshold}.txt", index=False, header=False)


if __name__ == '__main__':
    extract_seq_rxnaamapper()
