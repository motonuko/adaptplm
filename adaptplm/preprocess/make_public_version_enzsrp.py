from adaptplm.data.enz_seq_rxn_datasource import load_enz_seq_rxn_datasource

from pathlib import Path
from typing import Tuple, List, Dict, Any
import pandas as pd
import re


def _swap_reaction_sides(reaction: str) -> Tuple[str, bool]:
    """
    Swap left/right sides of a reaction SMILES string separated by '>>'.
    Returns the possibly-swapped reaction and a boolean indicating whether a swap occurred.
    """
    if not isinstance(reaction, str):
        return reaction, False
    parts = reaction.split(">>")
    assert len(parts) == 2
    left, right = parts
    swapped = f"{right}>>{left}"
    return swapped, True


def _normalize_metacyc_data_id(data_id: str) -> str:
    """
    For MetaCyc rows, normalize the data_id suffix:
    - Replace trailing '_l2r' or '_r2l' with '_metacyc'
    """
    if not isinstance(data_id, str):
        return data_id
    return re.sub(r"(?:_l2r|_r2l)$", "_undefined", data_id)


def main(data_path: Path):
    # Load source dataframe
    df = load_enz_seq_rxn_datasource(data_path)

    # Basic required columns presence check for safety
    required_cols = {"direction_source", "data_id", "rxn", "sequence", "rhea_master_id"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    original_rows = len(df)
    original_cols = list(df.columns)

    # Process rows one-by-one to preserve order
    processed_rows: List[Dict[str, Any]] = []
    swap_count = 0

    for _, row in df.iterrows():
        row_dict = row.to_dict()

        if row_dict.get("direction_source") == "MetaCyc":
            # If data_id ends with _l2r, swap the reaction sides
            data_id_val = row_dict.get("data_id")
            if isinstance(data_id_val, str) and data_id_val.endswith("_r2l"):
                new_rxn, swapped = _swap_reaction_sides(row_dict.get("rxn"))
                if swapped:
                    row_dict["rxn"] = new_rxn
                    swap_count += 1
            # Normalize data_id suffix to _metacyc (for both _l2r and _r2l)
            row_dict["data_id"] = _normalize_metacyc_data_id(data_id_val)

        # Append (dedup within MetaCyc will be handled after constructing the list to keep O(N))
        processed_rows.append(row_dict)
    # Convert back to DataFrame (still preserves order of processed_rows)
    new_df = pd.DataFrame(processed_rows, columns=df.columns)

    # Deduplicate only MetaCyc rows based on (seq, rhea_master_id), keeping the first occurrence to preserve order
    is_meta = new_df["direction_source"] == "MetaCyc"
    seen_keys = set()
    keep_mask = []

    for i, r in new_df.iterrows():
        if not is_meta.iloc[i]:
            keep_mask.append(True)
            continue
        key = (r.get("data_id"), r.get("primary_accession"), r.get("isoform_id"), r.get("ptm_key"), r.get("sequence"),
               r.get("rhea_master_id"))
        if key in seen_keys:
            keep_mask.append(False)  # drop duplicates
        else:
            seen_keys.add(key)
            keep_mask.append(True)

    new_df = new_df.loc[keep_mask].copy()
    deleted_rows = original_rows - len(new_df)

    # Drop 'rhea_id' column if present
    had_rhea_id_col = "rhea_id" in new_df.columns
    if had_rhea_id_col:
        new_df = new_df.drop(columns=["rhea_id"])

    # --------------------
    # Integrity checks
    # --------------------

    # 1) Row count consistency
    assert original_rows == len(new_df) + deleted_rows, (
        f"Row count mismatch: original={original_rows}, "
        f"new={len(new_df)}, deleted={deleted_rows}"
    )

    # 2) MetaCyc data_id must NOT contain 'l2r' or 'r2l'
    if not new_df.loc[new_df["direction_source"] == "MetaCyc", "data_id"].apply(
            lambda s: (isinstance(s, str) and ("l2r" not in s and "r2l" not in s))
    ).all():
        bad_ids = new_df.loc[
            (new_df["direction_source"] == "MetaCyc")
            & new_df["data_id"].astype(str).str.contains("l2r|r2l", regex=True),
            "data_id",
        ].tolist()
        raise AssertionError(f"Found MetaCyc data_id still containing l2r/r2l: {bad_ids}")

    # 3) Column count is one less due to removed 'rhea_id'
    #    If original file never had rhea_id, this check would fail by design.
    #    Per spec, we enforce that original contained 'rhea_id'.
    assert "rhea_id" in original_cols, "Original dataframe must contain 'rhea_id' column."
    assert new_df.shape[1] == len(original_cols) - 1, (
        f"Column count mismatch after removing 'rhea_id': "
        f"original_cols={len(original_cols)}, new_cols={new_df.shape[1]}"
    )

    # 4) For every original row with direction_source == 'UniProt_Physiological_Reaction',
    #    ensure that an identical data_id exists in the modified df.
    upr_ids = set(
        df.loc[df["direction_source"] == "UniProt_Physiological_Reaction", "data_id"]
        .astype(str)
        .tolist()
    )
    new_ids = set(new_df["data_id"].astype(str).tolist())
    missing_ids = [i for i in upr_ids if i not in new_ids]
    assert not missing_ids, (
        "Some UniProt_Physiological_Reaction data_id values disappeared after processing: "
        f"{missing_ids[:10]}{' ...' if len(missing_ids) > 10 else ''}"
    )

    # --------------------
    # Additional integrity check
    # --------------------
    meta_data_ids = new_df.loc[new_df["direction_source"] == "MetaCyc", "data_id"].astype(str)

    if not meta_data_ids.str.endswith("_undefined").all():
        bad_ids = meta_data_ids[~meta_data_ids.str.endswith("_undefined")].tolist()
        raise AssertionError(
            f"MetaCyc data_id not ending with '_undefined': {bad_ids[:10]}"
            + (" ..." if len(bad_ids) > 10 else "")
        )

    # --------------------
    # Save to CSV
    # --------------------
    # Save as '<original_filename>_modified.csv' next to the original
    src = Path(data_path)
    # Determine a sensible output filename even if the source wasn't a CSV file
    stem = src.stem  # filename without suffix
    out_path = src.with_name(f"{stem}_public.csv")
    new_df.to_csv(out_path, index=False)

    # Optional: print summary to help auditing
    print(f"[Summary] rows: original_n_rows={original_rows}, output_n_rows={len(new_df)}, "
          f"swapped_rxn_count_total(before dedup)={swap_count}, deleted_n_rows={deleted_rows}, "
          f"original_file='{src.name}', output_file='{out_path.name}'")

    return new_df, out_path


if __name__ == '__main__':
    target = Path('adaptplm_data/dataset/processed/enzsrp_full_cleaned/enzsrp_full_cleaned_test.csv')
    main(target)

    target = Path('adaptplm_data/dataset/processed/enzsrp_full_cleaned/enzsrp_full_cleaned_train.csv')
    main(target)

    target = Path('adaptplm_data/dataset/processed/enzsrp_full_cleaned/enzsrp_full_cleaned_val.csv')
    main(target)

    target = Path('adaptplm_data/dataset/processed/enzsrp_full_cleaned.csv')
    main(target)

    target = Path('adaptplm_data/dataset/raw/enzsrp_full.csv')
    main(target)
