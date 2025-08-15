import statistics
from pathlib import Path
from typing import List

from matplotlib import pyplot as plt

from adaptplm.core.default_path import DefaultPath
from adaptplm.core.package_version import get_package_major_version
from adaptplm.data.enz_seq_rxn_datasource import load_enz_seq_rxn_datasource


def print_stat(numbers: List[int]):
    print("Mean:", statistics.mean(numbers))
    print("Median:", statistics.median(numbers))
    print("Variance:", statistics.variance(numbers))  # Sample variance
    print("Standard Deviation:", statistics.stdev(numbers))  # Sample standard deviation
    print("Minimum:", min(numbers))
    print("Maximum:", max(numbers))


def main(data_path: Path, output_dir: Path):
    df = load_enz_seq_rxn_datasource(data_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = list(df['sequence'])
    unique_sequences = list(df['sequence'].unique())

    # Calculate sequence lengths
    sequence_lengths = [len(seq) for seq in sequences]
    unique_sequence_lengths = [len(seq) for seq in unique_sequences]

    print_stat(sequence_lengths)
    print()
    print_stat(unique_sequence_lengths)
    print()

    # n_bin = range(min(sequence_lengths), max(sequence_lengths) + 2)
    # n_bin = math.ceil(math.log2(len(sequences)) + 1)
    n_bin = 30
    # Plot sequence lengths for all sequences
    plt.figure(figsize=(10, 6))
    plt.hist(sequence_lengths, bins=n_bin, edgecolor='black',
             alpha=0.7)
    # plt.title('Distribution of Sequence Lengths')
    plt.xlabel('Length of Sequence')
    plt.ylabel('Frequency')
    plt.savefig(output_dir / f"seq_len_all.pdf", format="pdf")
    plt.show()

    # Plot sequence lengths for unique sequences
    # n_bin = range(min(sequence_lengths), max(sequence_lengths) + 2)
    # n_bin = math.ceil(math.log2(len(unique_sequences)) + 1)
    n_bin = 30
    plt.figure(figsize=(10, 6))
    plt.hist(unique_sequence_lengths, bins=n_bin,
             edgecolor='black', alpha=0.7)
    # plt.title('Distribution of Sequence Lengths (Unique Sequences)')
    plt.xlabel('Length of Sequence')
    plt.ylabel('Frequency')
    plt.savefig(output_dir / f"seq_len_unique.pdf", format="pdf")
    plt.show()


if __name__ == '__main__':
    v = get_package_major_version()
    main(data_path=DefaultPath().data_dataset_processed / 'enzsrp_full_cleaned.csv',
         output_dir=DefaultPath().build / 'fig' / v / 'enzsrp_full')
