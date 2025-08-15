import json
from pathlib import Path
from typing import List

import pandas as pd


def extract_ipr_features(json_files: List[Path]) -> pd.DataFrame:
    records = []

    for file_path in json_files:
        with file_path.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue  # JSON が壊れていたらスキップ

        for i, entry in enumerate(data.get("results", [])):
            seq_id = entry['xref'][0]['name']
            matches = entry.get("matches", [])
            for match in matches:
                signature = match.get("signature", {})
                entry_data = signature.get("entry")
                if entry_data and "accession" in entry_data:
                    interpro = entry_data["accession"]
                    records.append((seq_id, interpro))

    df = pd.DataFrame(records, columns=["seq_id", "IPR_ID"]).drop_duplicates()
    df["value"] = 1
    features_df = df.pivot(index="seq_id", columns="IPR_ID", values="value").fillna(0).astype(int)
    features_df.columns.name = None
    return features_df
