import numpy as np

from adaptplm.core.default_path import DefaultPath
from adaptplm.data.esp_datasource import load_esp_df


def list_up_kegg_mols():
    output_dir = DefaultPath().build / 'kegg'
    df = load_esp_df(DefaultPath().original_esp_fine_tuning_pkl)
    result = df.loc[~df['molecule ID'].str.startswith('CHEBI:'), 'molecule ID'].unique()
    result = np.sort(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / 'kegg_mol_ids_for_esp_fine_tuning.txt', result, fmt='%s')


if __name__ == '__main__':
    list_up_kegg_mols()
