import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from adaptplm.core.default_path import DefaultPath
from adaptplm.domain.regex_tokenizer import MultipleSmilesTokenizer


def main():
    tokenizer = MultipleSmilesTokenizer()
    df = pd.read_csv(DefaultPath().data_dataset_raw / 'enzsrp_full.csv')
    counts = []
    for rxn in df['rxn']:
        tokenized = tokenizer.tokenize(rxn)
        counts.append(len(tokenized))

    min_count = min(counts)
    max_count = max(counts)
    q1 = np.percentile(counts, 25)  # First quartile
    q3 = np.percentile(counts, 75)  # Third quartile
    iqr = q3 - q1  # Interquartile range
    upper_bound = q3 + 1.5 * iqr  # Upper bound for outliers
    print(
        f"min: {min_count}, max: {max_count}, upper_bound: {upper_bound}")

    # plt.hist(counts, bins=10, edgecolor='k')
    # plt.xlabel('Token Counts')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Token Counts')
    # plt.show()

    plt.boxplot(counts, vert=False, patch_artist=True)
    plt.xlabel('Token Counts')
    plt.title('Box Plot of Token Counts')
    plt.show()



if __name__ == '__main__':
    main()
