import unittest

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


class MyTestCase(unittest.TestCase):
    def test_something(self):
        radius = 2
        n_bits = 1024
        smiles = "C[C@H](O)C(=O)O"
        mol = Chem.MolFromSmiles(smiles)
        old = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, useChirality=True).ToList()
        generator = GetMorganGenerator(radius=radius, fpSize=n_bits, includeChirality=True)
        new = generator.GetFingerprint(mol).ToList()
        self.assertEqual(old, new)

    def test_something2(self):
        radius = 2
        n_bits = 1024
        smiles = "C[C@@H](O)C(=O)O"
        mol = Chem.MolFromSmiles(smiles)
        old = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, useChirality=True).ToList()
        generator = GetMorganGenerator(radius=radius, fpSize=n_bits, includeChirality=True)
        new = generator.GetFingerprint(mol).ToList()
        self.assertEqual(old, new)

    def test_something3(self):
        radius = 2
        n_bits = 1024
        smiles = "CC(O)C(=O)O"
        mol = Chem.MolFromSmiles(smiles)
        generator = GetMorganGenerator(radius=radius, fpSize=n_bits, includeChirality=True)
        new = generator.GetFingerprint(mol).ToList()
        generator2 = GetMorganGenerator(radius=radius, fpSize=n_bits)
        new2 = generator2.GetFingerprint(mol).ToList()
        self.assertEqual(new, new2)


if __name__ == '__main__':
    unittest.main()
