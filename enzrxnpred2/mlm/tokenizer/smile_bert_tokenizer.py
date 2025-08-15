"""Tokenization utilties for exrepssions."""
from typing import List

from transformers import BertTokenizer

from adaptplm.domain.regex_tokenizer import MultipleSmilesTokenizer


class SmilesBertTokenizer(BertTokenizer):
    """
    Constructs a SmilesBertTokenizer.
    Adapted from https://github.com/huggingface/transformers
    and https://github.com/rxn4chemistry/rxnfp.

    Args:
        vocabulary_file: path to a token per line vocabulary file.
    """

    def __init__(
            self,
            vocab_file: str,
            unk_token: str = "[UNK]",
            sep_token: str = "[SEP]",
            pad_token: str = "[PAD]",
            cls_token: str = "[CLS]",
            mask_token: str = "[MASK]",
            do_lower_case=False,
            **kwargs,
    ) -> None:
        super().__init__(
            vocab_file=vocab_file,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            do_lower_case=do_lower_case,
            **kwargs,
        )
        self.tokenizer = MultipleSmilesTokenizer()

    @property
    def vocab_list(self) -> List[str]:
        return list(self.vocab.keys())

    def _tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)
