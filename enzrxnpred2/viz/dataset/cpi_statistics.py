import pandas as pd

from adaptplm.core.default_path import DefaultPath
from adaptplm.data.original_enz_activity_dense_screen_datasource import EnzActivityScreeningDatasource, \
    EnzActivityScreeningDataset

title_map = {
    'duf': 'BKACE',
    'duf_filtered': 'BKACE',
    'esterase': 'Esterase',
    'esterase_filtered': 'Esterase',
    'gt_acceptors_chiral': 'Glycosyltransferase',
    'gt_acceptors_chiral_filtered': 'Glycosyltransferase',
    'halogenase_NaBr': 'Halogenase',
    'halogenase_NaBr_filtered': 'Halogenase',
    'olea': 'Thiolase',
    'olea_filtered': 'Thiolase',
    'phosphatase_chiral': 'Phosphatase',
    'phosphatase_chiral_filtered': 'Phosphatase',
}

def check_cpi_statistics():
    path = DefaultPath().data_original_dense_screen_processed
    path2 = DefaultPath().data_dataset_processed / 'cpi'
    source = EnzActivityScreeningDatasource(path, path2)
    output_dir_path = DefaultPath().build / 'viz'

    targets = [
        # EnzActivityScreeningDataset.HALOGENASE_NABR,
        EnzActivityScreeningDataset.HALOGENASE_NABR_FILTERED,
        # EnzActivityScreeningDataset.OLEA,
        EnzActivityScreeningDataset.OLEA_FILTERED,
        # EnzActivityScreeningDataset.DUF,
        EnzActivityScreeningDataset.DUF_FILTERED,
        # EnzActivityScreeningDataset.PHOSPHATASE_CHIRAL,
        EnzActivityScreeningDataset.PHOSPHATASE_CHIRAL_FILTERED,
        # EnzActivityScreeningDataset.ESTERASE,
        EnzActivityScreeningDataset.ESTERASE_FILTERED,
        # EnzActivityScreeningDataset.GT_ACCEPTORS_CHIRAL,
        EnzActivityScreeningDataset.GT_ACCEPTORS_CHIRAL_FILTERED,
    ]
    results = []
    for target in targets:
        df = source.load_binary_dataset(target)
        results.append({
            "dataset": title_map[target.value],
            "# of pairs": len(df),
            "# of unique sequences": len(df['SEQ'].unique()),
            "# of unique substrates": len(df['SUBSTRATES'].unique().tolist()),
        })
    results = pd.DataFrame(results)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir_path / 'cpi_dataset_table.csv', index=False)
    results.to_latex(output_dir_path / 'cpi_dataset_table.tex', index=False)


if __name__ == '__main__':
    check_cpi_statistics()
