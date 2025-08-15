from pathlib import Path

from adaptplm.core.default_path import DefaultPath
from adaptplm.mlm.tokenizer.smile_bert_tokenizer import SmilesBertTokenizer


def get_vocab(path: Path):
    with open(path) as f:
        return [line.strip() for line in f]


def main():
    vocab_base = DefaultPath().data_dataset_processed / 'vocab'
    enzsrp_full_vocab = vocab_base / 'enzsrp_full_cleand_train_vocab.txt'

    tokenizer2 = SmilesBertTokenizer(enzsrp_full_vocab.as_posix())
    print(f"2. total {len(tokenizer2.vocab)}")


if __name__ == '__main__':
    main()
