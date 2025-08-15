import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional

from enzrxnpred2.domain.regex_tokenizer import MultipleSmilesTokenizer

# see BertTokenizer
SPECIAL_TOKENS_FOR_BERT_MODEL = [
    "[PAD]",
    "[unused1]",
    "[unused2]",
    "[unused3]",
    "[unused4]",
    "[unused5]",
    "[unused6]",
    "[unused7]",
    "[unused8]",
    "[unused9]",
    "[unused10]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
]

control_char_pattern = re.compile(r'[\x00-\x1F\x7F]')


def build_vocab(data_file: Path, output_filepath: Path, ignore_first_line: bool, min_freq: int,
                base_tokens: Optional[List] = None):
    tokenizer = MultipleSmilesTokenizer()
    vocabulary_counter = Counter()
    with open(data_file) as f:
        lines = [line.strip() for line in f]
        if ignore_first_line:
            lines = lines[1:]
    for line in lines:
        tokens = tokenizer.tokenize(line)
        vocabulary_counter.update(tokens)
    with open(output_filepath, "wt") as fp:
        tokens = [token for token, freq in vocabulary_counter.most_common() if freq >= min_freq]
        tokens = [control_char_pattern.sub('', t) for t in tokens]
        if base_tokens:
            not_duplicated_base_tokens = [token for token in base_tokens if token not in tokens]
            tokens = not_duplicated_base_tokens + list(tokens)  # ordering
        fp.write(os.linesep.join(tokens))
    logging.info(f"Process completed. Token size {len(tokens)}. File has been saved to: {output_filepath}")
