import unittest

from enzrxnpred2.downstream.cpi.domain.generate_seeds import generate_nested_cv_seeds


class TestGenerateNestedCvSeeds(unittest.TestCase):

    def test_generate_nested_cv_seeds_default_offsets(self):
        seed_list = [42, 43, 44]
        expected = [
            {"base_seed": 42, "outer_cv_seed": 142, "inner_cv_seed": 10042},
            {"base_seed": 43, "outer_cv_seed": 143, "inner_cv_seed": 10043},
            {"base_seed": 44, "outer_cv_seed": 144, "inner_cv_seed": 10044},
        ]
        result = generate_nested_cv_seeds(seed_list)
        self.assertEqual(result, expected)

    def test_generate_nested_cv_seeds_custom_offsets(self):
        seed_list = [10, 20]
        outer_offset = 200
        inner_offset = 5000
        expected = [
            {"base_seed": 10, "outer_cv_seed": 210, "inner_cv_seed": 5010},
            {"base_seed": 20, "outer_cv_seed": 220, "inner_cv_seed": 5020},
        ]
        result = generate_nested_cv_seeds(seed_list, outer_offset, inner_offset)
        self.assertEqual(result, expected)

    def test_generate_nested_cv_seeds_empty_list(self):
        seed_list = []
        expected = []
        result = generate_nested_cv_seeds(seed_list)
        self.assertEqual(result, expected)

    def test_generate_nested_cv_seeds_single_seed(self):
        seed_list = [1]
        expected = [
            {"base_seed": 1, "outer_cv_seed": 101, "inner_cv_seed": 10001},
        ]
        result = generate_nested_cv_seeds(seed_list)
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
