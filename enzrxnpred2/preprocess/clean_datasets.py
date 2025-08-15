import logging
from functools import lru_cache
from pathlib import Path

from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions
from tqdm import tqdm

from enzrxnpred2.core.constants import MAX_RXN_TOKENIZED_LEN, MIN_HEAVY_ATOM, MAX_SEQUENCE_LENGTH, MIN_SEQUENCE_LENGTH
from enzrxnpred2.data.enz_seq_rxn_datasource import load_enz_seq_rxn_datasource
from enzrxnpred2.domain.regex_tokenizer import MultipleSmilesTokenizer
from enzrxnpred2.extension.rdkit_ext import remove_atom_mapping

tqdm.pandas()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@lru_cache()
def tokenized_len(rxn: str):
    tokenizer = MultipleSmilesTokenizer()
    tokenized_rxn = tokenizer.tokenize(rxn)
    return len(tokenized_rxn)


def _rxn_smiles_preprocessing(rxn_smiles: str, need_to_remove_atom_mapping: bool, min_heavy_atom_count=MIN_HEAVY_ATOM,
                              max_tokenized_len=MAX_RXN_TOKENIZED_LEN):
    try:
        # Remove reagent
        reactants_smiles, _, products_smiles = rxn_smiles.split('>')
        rxn = AllChem.ReactionFromSmarts(f"{reactants_smiles}>>{products_smiles}")
        # Remove atom mapping
        if need_to_remove_atom_mapping:
            reactants = [remove_atom_mapping(mol) for mol in rxn.GetReactants()]
            products = [remove_atom_mapping(mol) for mol in rxn.GetProducts()]
        else:
            reactants = [mol for mol in rxn.GetReactants()]
            products = [mol for mol in rxn.GetProducts()]
        # Remove no reactants or no products reactions
        if len(reactants) == 0 or len(products) == 0:
            return None
        # Remove no chemical transformation reactions
        reactants_smiles = [Chem.MolToSmiles(mol) for mol in reactants]
        products_smiles = [Chem.MolToSmiles(mol) for mol in products]
        if set(reactants_smiles) == set(products_smiles):
            return None
        # Filter by heavy atom count
        if min_heavy_atom_count is not None:
            reactant_heavy_atom_count = sum(mol.GetNumHeavyAtoms() for mol in reactants)
            product_heavy_atom_count = sum(mol.GetNumHeavyAtoms() for mol in products)
            if reactant_heavy_atom_count < min_heavy_atom_count or product_heavy_atom_count < min_heavy_atom_count:
                # print(rxn_smiles)
                return None
        joined_reactant_smiles = ".".join([Chem.MolToSmiles(mol) for mol in reactants])
        joined_product_smiles = ".".join([Chem.MolToSmiles(mol) for mol in products])
        reaction_smarts = f"{joined_reactant_smiles}>>{joined_product_smiles}"
        reaction = rdChemReactions.ReactionFromSmarts(reaction_smarts)
        reaction_smiles = rdChemReactions.ReactionToSmiles(reaction)
        if tokenized_len(rxn_smiles) > max_tokenized_len:
            return None
        # print(reaction_smiles)
        return reaction_smiles
    except Exception as e:
        logging.info(f"Error processing reaction SMILES. Skipping this entry({rxn_smiles}). Details: {e}")
        # P12946 have `->[Fe]45<-` in rxn, unexpected.
        return None


def clean_enzyme_reaction_pair_dataset(data: Path, output_file: Path, max_sequence_length=MAX_SEQUENCE_LENGTH,
                                       min_sequence_length=MIN_SEQUENCE_LENGTH,
                                       n_jobs=-1):
    df = load_enz_seq_rxn_datasource(data)
    original_size = len(df)
    df['rxn'] = Parallel(n_jobs=n_jobs)(  # The order of the results keeps original order
        delayed(_rxn_smiles_preprocessing)(row, need_to_remove_atom_mapping=False) for row in
        tqdm(df["rxn"]))
    df['sequence'] = df['sequence'].apply(lambda x: x if min_sequence_length <= len(x) <= max_sequence_length else None)
    df = df.dropna(subset=['rxn', 'sequence'])
    logging.info(f"{original_size - len(df)} pairs removed ")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    logging.info(f"Preprocessing completed. File has been saved to: {output_file}")
