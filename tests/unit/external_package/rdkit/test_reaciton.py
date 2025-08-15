import unittest

from rdkit.Chem import AllChem
from rdkit.Chem.rdChemReactions import ReactionToSmiles


class TestReactionToSmiles(unittest.TestCase):
    def setUp(self):
        # self.reaction = AllChem.ReactionFromSmarts("[NH3:3].O[C:1]=[O:2]>>[C:1](=[O:2])[N:3]")  # non canonical order
        self.reaction = AllChem.ReactionFromSmarts("OC=O.N>>NC=O")  # non canonical order

    def test_canonical_smiles(self):
        canonical_smiles = ReactionToSmiles(self.reaction, canonical=True)
        # expected_canonical = "O[C:1]=[O:2].[NH3:3]>>[C:1](=[O:2])[N:3]"
        expected_canonical = 'N.OC=O>>NC=O'
        self.assertEqual(canonical_smiles, expected_canonical)  # `canonical=True` changes the order of molecules.

    def test_non_canonical_smiles(self):
        non_canonical_smiles = ReactionToSmiles(self.reaction, canonical=False)
        expected_non_canonical = "OC=O.N>>NC=O"
        self.assertEqual(non_canonical_smiles, expected_non_canonical)


if __name__ == '__main__':
    unittest.main()
