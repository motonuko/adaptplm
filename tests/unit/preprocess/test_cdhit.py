import unittest

# noinspection PyProtectedMember
from adaptplm.preprocess.clean_datasets import _rxn_smiles_preprocessing


class TestRxnSmilesPreprocessing(unittest.TestCase):

    def test_valid_reaction_smiles_with_mapping(self):
        rxn_smiles = "CCO>>CO"
        result = _rxn_smiles_preprocessing(rxn_smiles, need_to_remove_atom_mapping=True)
        self.assertEqual(result, "CCO>>CO", f"Unexpected result: {result}")

    def test_valid_reaction_smiles_without_mapping(self):
        rxn_smiles = "CCO>>CO"
        result = _rxn_smiles_preprocessing(rxn_smiles, need_to_remove_atom_mapping=False)
        self.assertEqual(result, "CCO>>CO", f"Unexpected result: {result}")

    def test_reaction_with_no_change(self):
        rxn_smiles = "CCO>>CCO"
        result = _rxn_smiles_preprocessing(rxn_smiles, need_to_remove_atom_mapping=True)
        self.assertIsNone(result, "Should return None for no-change reactions")

    def test_reaction_with_empty_reactants_or_products(self):
        rxn_smiles = ">>CO"
        result = _rxn_smiles_preprocessing(rxn_smiles, need_to_remove_atom_mapping=True)
        self.assertIsNone(result, "Should return None for empty reactants")

        rxn_smiles = "CCO>>"
        result = _rxn_smiles_preprocessing(rxn_smiles, need_to_remove_atom_mapping=True)
        self.assertIsNone(result, "Should return None for empty products")

    def test_reaction_with_insufficient_heavy_atoms(self):
        rxn_smiles = "CC>>C"  # invalid
        result = _rxn_smiles_preprocessing(rxn_smiles, need_to_remove_atom_mapping=True, min_heavy_atom_count=2)
        self.assertIsNone(result, "Should return None for insufficient heavy atoms")

    def test_reaction_with_insufficient_heavy_atoms2(self):
        rxn_smiles = "C>>CC"  # invalid
        result = _rxn_smiles_preprocessing(rxn_smiles, need_to_remove_atom_mapping=True, min_heavy_atom_count=2)
        self.assertIsNone(result, "Should return None for insufficient heavy atoms")

    def test_reaction_with_long_tokenized_length(self):
        rxn_smiles = "C" * 100 + ">>" + "C" * 100
        result = _rxn_smiles_preprocessing(rxn_smiles, need_to_remove_atom_mapping=True, max_tokenized_len=50)
        self.assertIsNone(result, "Should return None for long tokenized lengths")

    def test_invalid_reaction_smiles(self):
        rxn_smiles = "INVALID_SMILES"
        result = _rxn_smiles_preprocessing(rxn_smiles, need_to_remove_atom_mapping=True)
        self.assertIsNone(result, "Should return None for invalid SMILES")

    def test_exception_handling(self):
        rxn_smiles = ">>>"
        result = _rxn_smiles_preprocessing(rxn_smiles, need_to_remove_atom_mapping=True)
        self.assertIsNone(result, "Should handle exceptions gracefully")


if __name__ == "__main__":
    unittest.main()
