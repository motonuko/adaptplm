import random
from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


def remove_atom_mapping(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    smiles = Chem.MolToSmiles(mol)
    newmol = Chem.MolFromSmiles(smiles)
    # newmol = Chem.RemoveHs(mol)  # Labeled hydrogen? H cannot be removed.
    return newmol


# https://github.com/rdkit/rdkit/discussions/6489#discussioncomment-6323403
# You only need to set the seed once at the start of the program (same for the random package)
def randomize_reaction_smiles(rxn_smiles: str):
    rxn = AllChem.ReactionFromSmarts(rxn_smiles)
    parts = [rxn.GetReactants(), rxn.GetAgents(), rxn.GetProducts()]
    part_texts = []
    for part in parts:
        smiles_texts_in_part = []
        for mol in part:
            random_smiles = Chem.MolToSmiles(mol, doRandom=True)
            smiles_texts_in_part.append(random_smiles)
        random.shuffle(smiles_texts_in_part)
        part_texts.append('.'.join(smiles_texts_in_part))
    return '>'.join(part_texts)


# @lru_cache
# def compute_morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 1024):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is not None:
#         return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, useChirality=True)
#     else:
#         return None

@lru_cache
def compute_morgan_fingerprint_as_array(smiles: str, radius: int = 2, n_bits: int = 1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # NOTE: ToList() should be used before np.array()?
        generator = GetMorganGenerator(radius=radius, fpSize=n_bits, includeChirality=True)
        return np.array(generator.GetFingerprint(mol))
        # return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, useChirality=True))
    else:
        return None
