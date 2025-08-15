from pathlib import Path

from rdkit import Chem

from adaptplm.core.default_path import DefaultPath


def molfile_to_mol(molfile_path: Path):
    try:
        with open(molfile_path, "r") as f:
            mol_block = f.read()
        mol = Chem.MolFromMolBlock(mol_block, sanitize=True)
        return mol
    except Exception as e:
        raise ValueError(f"Error reading {molfile_path}: {e}") from e


def create_id_inchi_table(id_file: Path, mol_dir: Path, output_file: Path):
    with open(id_file, "r") as f:
        ids = [line.strip() for line in f if line.strip()]

    with open(output_file, "w") as out_f:
        out_f.write("ID\tInChI\n")
        for mol_id in ids:
            mol_path = mol_dir / f"{mol_id}.mol"
            try:
                mol = molfile_to_mol(mol_path)
                if mol is None:
                    raise ValueError(f"Failed to map to Mol object: {mol_id}")
                inchi_str = Chem.inchi.MolToInchi(mol)
                out_f.write(f"{mol_id}\t{inchi_str}\n")
            except Exception as e:
                print(e)
                out_f.write(f"{mol_id}\t\n")


if __name__ == '__main__':
    base_dir = DefaultPath().build / 'kegg'
    create_id_inchi_table(id_file=base_dir / 'kegg_mol_ids_for_esp_fine_tuning.txt', mol_dir=base_dir / 'kegg_entries',
                          output_file=base_dir / 'kegg_mol_inchi.tsv')
