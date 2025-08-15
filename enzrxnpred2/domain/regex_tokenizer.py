import re
from abc import ABC
from typing import List


class RegexTokenizer(ABC):
    def __init__(self, regex_pattern: str, suffix: str = None) -> None:
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)
        self.suffix = suffix

    def tokenize(self, text: str) -> List[str]:
        if self.suffix is None:
            return self.regex.findall(text)
        else:
            return [f"{token}{self.suffix}" for token in self.regex.findall(text)]


class MultipleSmilesTokenizer(RegexTokenizer):
    # https://github.com/rxn4chemistry/biocatalysis-model/blob/main/rxn_biocatalysis_tools/tokenizer.py
    SMILES_TOKENIZER_PATTERN = r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"

    def __init__(self, suffix: str = ""):
        super().__init__(self.SMILES_TOKENIZER_PATTERN, suffix)


if __name__ == '__main__':
    print(MultipleSmilesTokenizer().tokenize("O=C([O-])O.[H+]>>O=C=O.[H]O[H]"))
