from pathlib import Path
from typing import List, Optional

import pandas as pd

columns = [
    "SEQ ID", "MD5", "Sequence Length", "Analysis",
    "Signature Accession", "Signature Description", "Start", "End",
    "Score", "Status", "Date", "InterPro Accession", "InterPro Description",
    "GO Terms", "Pathways"
]


def load_interproscan_result_tsv(tsv_file_path: Path):
    return pd.read_csv(tsv_file_path, sep="\t", header=None, names=columns, comment='#')


def load_interproscan_result_tsv_as_ipa_binary_feature(tsv_file_path: Path,
                                                       ignored_accessions: Optional[List[str]] = None):
    df = load_interproscan_result_tsv(tsv_file_path)
    df = df[df["InterPro Accession"] != "-"].copy()
    if ignored_accessions:
        df = df[df["InterPro Accession"].isin(ignored_accessions)]
    else:
        assert len(df['InterPro Accession'].unique()) < 3000, \
            'This pivot_table may be slow due to the large number of columns.'  # If performance is not an issue, you may comment this out.
    presence_df = df.pivot_table(
        index="SEQ ID",
        columns="InterPro Accession",
        aggfunc="size",
        fill_value=0
    )
    presence_df = (presence_df > 0).astype(int)
    presence_df.columns.name = None
    return presence_df
