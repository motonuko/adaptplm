from pathlib import Path

from enzrxnpred2.data.enz_seq_rxn_datasource import load_enz_seq_rxn_datasource


def create_filtered_seq_rxn_pair_dataset(seq_rxn_pair_file: Path, output_file: Path, exclude_seq_hashes: Path):
    df = load_enz_seq_rxn_datasource(seq_rxn_pair_file, need_hash=True)
    with open(exclude_seq_hashes) as f:
        similar_seqs = [line.strip() for line in f.readlines()]
    filtered_df = df[~df['seq_crc64'].isin(similar_seqs)]
    assert len(df) - len(filtered_df) >= len(similar_seqs)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_file, index=False)


