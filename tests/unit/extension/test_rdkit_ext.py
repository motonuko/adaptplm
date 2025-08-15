import unittest

from rdkit import Chem
from rdkit.Chem import AllChem

from adaptplm.extension.rdkit_ext import remove_atom_mapping, randomize_reaction_smiles, \
    compute_morgan_fingerprint_as_array


class TestRemoveAtomMapping(unittest.TestCase):
    def test_remove_atom_mapping(self):
        rxn_str = "[NH3:3].[O:2]=[C:1]O>>[N:3][C:1]=[O:2]"
        rxn = AllChem.ReactionFromSmarts(rxn_str)
        reactants = [remove_atom_mapping(mol) for mol in rxn.GetReactants()]
        products = [remove_atom_mapping(mol) for mol in rxn.GetProducts()]
        joined_reactant_smiles = ".".join([Chem.MolToSmiles(mol) for mol in reactants])
        joined_product_smiles = ".".join([Chem.MolToSmiles(mol) for mol in products])
        reaction_smarts = f"{joined_reactant_smiles}>>{joined_product_smiles}"
        expected = "N.O=CO>>NC=O"
        self.assertEqual(expected, reaction_smarts)

    def test_allchem_does_not_remove_atom_mapping(self):
        rxn_str = "O[C:1]=[O:2].[NH3:3]>>[C:1](=[O:2])[N:3]"  # canonical order
        rxn = AllChem.ReactionFromSmarts(rxn_str)
        rxn = AllChem.ReactionToSmiles(rxn)
        print(rxn)
        self.assertEqual(rxn_str, rxn)


class TestRandomizeReactionSmiles(unittest.TestCase):

    def setUp(self):
        # A simple reaction SMARTS with no agents
        self.rxn_smiles = "CCO>>CC=O"
        self.seed = 42

    @staticmethod
    def _canonicalize_reaction_smiles(rxn_smiles: str) -> str:
        # reaction_from_smarts can parse reaction smiles
        rxn = AllChem.ReactionFromSmarts(rxn_smiles)
        parts = [rxn.GetReactants(), rxn.GetAgents(), rxn.GetProducts()]
        canonical_parts = []
        for part in parts:
            # Get canonical SMILES (default: canonical=True) for each molecule and sort them
            canonical_smiles_list = sorted([Chem.MolToSmiles(mol) for mol in part])
            canonical_parts.append(".".join(canonical_smiles_list))
        return ">".join(canonical_parts)

    # check: https://github.com/rdkit/rdkit/discussions/6489#discussioncomment-6323403
    def test_deterministic_with_fixed_seed(self):
        output1 = randomize_reaction_smiles(self.rxn_smiles)
        self.assertEqual(output1, 'CCO>>CC=O')

    # check: https://github.com/rdkit/rdkit/discussions/6489#discussioncomment-6323403
    def test_different_with_different_seed(self):
        output1 = randomize_reaction_smiles(self.rxn_smiles)
        output2 = randomize_reaction_smiles(self.rxn_smiles)
        self.assertNotEqual(output1, output2)  # not 100% (depends on the seed)

    def test_output_format(self):
        output = randomize_reaction_smiles(self.rxn_smiles)
        parts = output.split(">")
        self.assertEqual(len(parts), 3)
        self.assertTrue(parts[0])
        self.assertTrue(parts[2])

    def test_canonical_consistency(self):
        canonical_input = self._canonicalize_reaction_smiles(self.rxn_smiles)
        randomized = randomize_reaction_smiles(self.rxn_smiles)
        canonical_randomized = self._canonicalize_reaction_smiles(randomized)
        self.assertEqual(canonical_input, canonical_randomized)


class TestMorganFingerprintFunctions(unittest.TestCase):
    # def test_compute_morgan_fingerprint_valid_smiles(self):
    #     smiles = "CCO"  # Ethanol
    #     fingerprint = compute_morgan_fingerprint(smiles)
    #     self.assertIsNotNone(fingerprint, "Fingerprint should not be None for valid SMILES")
    #     self.assertEqual(fingerprint.GetNumBits(), 1024)
    #
    # def test_compute_morgan_fingerprint_invalid_smiles(self):
    #     smiles = "invalid_smiles"
    #     fingerprint = compute_morgan_fingerprint(smiles)
    #     self.assertIsNone(fingerprint, "Fingerprint should be None for invalid SMILES")

    def test_compute_morgan_fingerprint_as_list_valid_smiles(self):
        smiles = "CCO"  # Ethanol
        fingerprint_list = compute_morgan_fingerprint_as_array(smiles, n_bits=1024)
        print(type(fingerprint_list))
        self.assertIsNotNone(fingerprint_list, "Fingerprint list should not be None for valid SMILES")
        self.assertEqual(len(fingerprint_list), 1024, "Fingerprint list should have 1024 elements")

    def test_compute_morgan_fingerprint_as_list_valid_smiles2(self):
        smiles = "CCO"  # Ethanol
        fingerprint_list = compute_morgan_fingerprint_as_array(smiles)
        print(type(fingerprint_list))
        self.assertIsNotNone(fingerprint_list, "Fingerprint list should not be None for valid SMILES")
        self.assertEqual(len(fingerprint_list), 1024, "Fingerprint list should have 1024 elements")
        self.assertTrue((fingerprint_list == 0).sum() + (fingerprint_list == 1).sum() == 1024,
                        "Fingerprint list should contain only 0s and 1s")

    def test_compute_morgan_fingerprint_as_list_invalid_smiles(self):
        smiles = "invalid_smiles"
        fingerprint_list = compute_morgan_fingerprint_as_array(smiles)
        self.assertIsNone(fingerprint_list, "Fingerprint list should be None for invalid SMILES")


if __name__ == "__main__":
    unittest.main()
