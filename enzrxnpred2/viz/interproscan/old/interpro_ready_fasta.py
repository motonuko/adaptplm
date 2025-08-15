from enzrxnpred2.core.default_path import DefaultPath
from enzrxnpred2.data.original_enz_activity_dense_screen_datasource import EnzActivityScreeningDataset


def split_fasta(input_file, output_dir, chunk_size=100):
    with open(input_file, 'r') as f:
        sequences = []
        current_seq = []

        for line in f:
            if line.startswith('>'):
                if current_seq:
                    sequences.append(current_seq)
                current_seq = [line]
            else:
                current_seq.append(line)

        # 最後のシーケンスも追加
        if current_seq:
            sequences.append(current_seq)

    # 分割して保存
    for i in range(0, len(sequences), chunk_size):
        chunk = sequences[i:i + chunk_size]
        output_file = output_dir / f"{input_file.stem}_part_{i // chunk_size + 1}.fasta"
        with open(output_file, 'w') as out:
            for seq in chunk:
                out.writelines(seq)

    print(f"{len(sequences)} sequences split into {((len(sequences) - 1) // chunk_size + 1)} files.")


def main():
    key = EnzActivityScreeningDataset.PHOSPHATASE_CHIRAL.value
    key = EnzActivityScreeningDataset.HALOGENASE_NABR_FILTERED.value
    path = DefaultPath().build / 'fasta' / key / f"{key}_input.fasta"
    output = DefaultPath().build / 'fasta_100' / key
    output.mkdir(parents=True, exist_ok=True)
    split_fasta(path, output_dir=output)


if __name__ == '__main__':
    main()
