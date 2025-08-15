import unittest

from adaptplm.domain.regex_tokenizer import MultipleSmilesTokenizer


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MultipleSmilesTokenizer()

    def test_something(self):
        expected = ['O', '=', 'C', '(', '[O-]', ')', 'O', '.', '[H+]', '>>', 'O', '=', 'C', '=', 'O', '.', '[H]', 'O',
                    '[H]']
        tokenized = self.tokenizer.tokenize("O=C([O-])O.[H+]>>O=C=O.[H]O[H]")

        self.assertEqual(expected, tokenized)


if __name__ == '__main__':
    unittest.main()
