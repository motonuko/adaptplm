import pandas as pd
from matplotlib import pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator

from enzrxnpred2.core.default_path import DefaultPath
from enzrxnpred2.data.enz_seq_rxn_datasource import load_enz_seq_rxn_datasource
from enzrxnpred2.data.esp_datasource import load_esp_df
from enzrxnpred2.data.turnup_datasource import load_kcat_df


def mol_from_inchi_safe(inchi):
    mol = Chem.MolFromInchi(inchi)
    if mol:
        return mol
    raise RuntimeError('unexpected')


def load_kcat_all_mols_for_unique_inchis():
    df1 = load_kcat_df(DefaultPath().original_kcat_test_pkl)
    df2 = load_kcat_df(DefaultPath().original_kcat_train_pkl)
    df = pd.concat([df1, df2], axis=0, ignore_index=True)
    all_unique_inchis = set()
    for cell in [*df["substrates"], *df["products"]]:
        all_unique_inchis.update(cell)
    return [mol_from_inchi_safe(inchi) for inchi in all_unique_inchis]


def get_mols_from_reaction_smiles(rsmi: str):
    if not rsmi or not isinstance(rsmi, str):
        return []
    rxn = AllChem.ReactionFromSmarts(rsmi, useSmiles=True)
    if rxn is None:
        return []

    mols = []
    for reactant in rxn.GetReactants():
        mols.append(reactant)
    for product in rxn.GetProducts():
        mols.append(product)
    for agent in rxn.GetAgents():
        mols.append(agent)

    return mols


def has_dummy_atoms(mol: Chem.Mol) -> bool:
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if any(substring in symbol for substring in ["*", "R", "R#"]):
            return True
    return False


def has_wildcard_atoms(mol: Chem.Mol) -> bool:
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'A':
            return True
        if atom.GetSymbol() == 'a':
            return True
    return False


def load_enzsrp_train_mols():
    path = DefaultPath().data_dataset_processed.joinpath('enzsrp_full_cleaned', 'enzsrp_full_cleaned_train.csv')
    df = load_enz_seq_rxn_datasource(path)
    all_mols = []
    for rsmi in df["rxn"].unique():
        all_mols.extend(get_mols_from_reaction_smiles(rsmi))
    return all_mols


def load_esp_train_mols():
    base_dir = DefaultPath().original_esp_fine_tuning_pkl.parent.parent
    df = load_esp_df(DefaultPath().original_esp_fine_tuning_pkl)
    df_all_molecule = df[['molecule ID']].drop_duplicates(keep="first")
    n_of_all_mols = len(df_all_molecule)

    df2 = pd.read_csv(base_dir / 'substrate_data' / 'chebiID_to_inchi.tsv', sep='\t')
    df3 = pd.read_csv(DefaultPath().build / 'kegg' / 'kegg_mol_inchi.tsv', sep='\t')

    df2 = df2[['Input', 'Inchi']].drop_duplicates(keep="first")
    df2["Input"] = df2["Input"].str.replace("ChEBI:", "CHEBI:", regex=False)
    df2 = df2.rename(columns={"Inchi": "InChI_chebi"})
    df3 = df3.rename(columns={"InChI": "InChI_kegg"})

    df_all_molecule = df_all_molecule.merge(df2, left_on="molecule ID", right_on='Input', how="left")
    df_all_molecule = df_all_molecule.merge(df3, left_on="molecule ID", right_on='ID', how="left")
    df_all_molecule = df_all_molecule[["molecule ID", "InChI_chebi", "InChI_kegg"]]
    df_all_molecule["InChI"] = df_all_molecule["InChI_chebi"].fillna(df_all_molecule["InChI_kegg"])
    df_all_molecule = df_all_molecule.drop(columns=["InChI_chebi", "InChI_kegg"])
    assert n_of_all_mols == len(df_all_molecule)

    df_all_molecule = df_all_molecule.dropna()
    inches = df_all_molecule['InChI'].unique().tolist()
    return [mol_from_inchi_safe(inchi) for inchi in inches]


def deduplicate_mols_by_inchikey(mol_list):
    seen = set()
    unique_mols = []
    for mol in mol_list:
        if mol is None:
            continue
        inchikey = Chem.MolToInchiKey(mol)
        if inchikey not in seen:
            seen.add(inchikey)
            unique_mols.append(mol)
    return unique_mols


def max_similarities(query_fps, ref_fps):
    max_sims = []
    for q in query_fps:
        sims = DataStructs.BulkTanimotoSimilarity(q, ref_fps)
        max_sims.append(max(sims))
    return max_sims


# def mols_to_ecfp4(mols, radius=2, nBits=1024):
def mols_to_ecfp4(mols, radius=3, nBits=2048):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    return [gen.GetFingerprint(m) for m in mols]  # compare_coverage


def clean_mols(mols):
    cleaned = []
    for m in mols:
        if m is None:
            continue
        if has_dummy_atoms(m) or has_wildcard_atoms(m):
            continue
        try:
            Chem.SanitizeMol(m)
            cleaned.append(m)
        except Exception as e:
            print(f"[WARN] Invalid mol skipped: {e}")
            continue
    return cleaned

def main():
    mols_kcat = load_kcat_all_mols_for_unique_inchis()
    mols_enzsrp = load_enzsrp_train_mols()
    mols_esp = load_esp_train_mols()

    mols_kcat = clean_mols(mols_kcat)
    mols_enzsrp = clean_mols(mols_enzsrp)
    mols_esp = clean_mols(mols_esp)

    # drop duplicates
    mols_kcat = deduplicate_mols_by_inchikey(mols_kcat)
    mols_enzsrp = deduplicate_mols_by_inchikey(mols_enzsrp)
    mols_esp = deduplicate_mols_by_inchikey(mols_esp)

    # compute fingerprint
    fps_kcat = mols_to_ecfp4(mols_kcat)
    fps_enzsrp = mols_to_ecfp4(mols_enzsrp)
    fps_esp = mols_to_ecfp4(mols_esp)

    # compute max similarity
    sims_kcat_vs_enzsrp = max_similarities(fps_kcat, fps_enzsrp)
    sims_kcat_vs_esp = max_similarities(fps_kcat, fps_esp)

    win_count = sum(1 for x, y in zip(sims_kcat_vs_enzsrp, sims_kcat_vs_esp) if x > y)
    equal_count = sum(1 for x, y in zip(sims_kcat_vs_enzsrp, sims_kcat_vs_esp) if x == y)
    lose_count = sum(1 for x, y in zip(sims_kcat_vs_enzsrp, sims_kcat_vs_esp) if x < y)
    total = len(sims_kcat_vs_enzsrp)

    # draw scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(sims_kcat_vs_enzsrp, sims_kcat_vs_esp, alpha=0.6)
    plt.xlabel("Nearest neighbor similarity (1 vs 2)")
    plt.ylabel("Nearest neighbor similarity (1 vs 3)")
    plt.title("Coverage comparison: Dataset 2 vs 3")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.grid(True)
    plt.show()

    print(f"Win  : {win_count} / {total} ({win_count / total:.2%})")
    print(f"Equal: {equal_count} / {total} ({equal_count / total:.2%})")
    print(f"Lose : {lose_count} / {total} ({lose_count / total:.2%})")

    return sims_kcat_vs_enzsrp, sims_kcat_vs_esp


if __name__ == '__main__':
    main()
