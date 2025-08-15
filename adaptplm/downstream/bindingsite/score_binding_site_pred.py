import argparse
import ast
import json
import random

import pandas as pd

from adaptplm.core.default_path import DefaultPath
from adaptplm.downstream.bindingsite.core import slices_to_indices, indices_to_slices, parse_slice_texts
from adaptplm.downstream.bindingsite.external import get_overlap_and_penalty_score
from adaptplm.extension.bio_ext import calculate_crc64


# def overlap_score(correct: List[List[int]], predicted: List[List[int]]) -> float:
#     bunshi = 0
#     bunbo = 0
#     for a_s, b_s in correct:
#         for a_p, b_p in predicted:
#             bunshi += max(0, min(b_p, b_s) - max(a_p, a_s))
#         bunbo += b_s - a_s
#     return bunshi / bunbo
#
#
# def false_positive_rate(correct: List[List[int]], predicted: List[List[int]]) -> float:
#     bunshi = 0
#     bunbo = 0
#     for a_p, b_p in predicted:
#         no_overlap = True
#         for a_s, b_s in correct:
#             is_overlapping = max(a_p, a_s) < min(b_p, b_s)
#             if is_overlapping:
#                 no_overlap = False
#         factor = 1 if no_overlap else 0
#         bunshi += factor * (b_p - a_p)
#         bunbo += b_p - a_p
#     return bunshi / bunbo


# Reproducing the numbers from the paper + our own implementation (score calculation is done for each seq–rxn pair)
def score(parent_path):
    df_pfam = pd.read_csv(DefaultPath().data_original_rxnaamapper_predictions / 'pfam.csv')
    df_pfam = df_pfam.rename(columns={'predicted_active_site': 'predicted_sites_pfam'})
    df_rxnaamapper = pd.read_csv(DefaultPath().data_original_rxnaamapper_predictions / 'rxnaamapper.csv')
    df_rxnaamapper = df_rxnaamapper.rename(columns={'predicted_active_site': 'predicted_sites_rxnaamapper'})
    df = pd.merge(df_pfam, df_rxnaamapper, on=['pdb-id', 'EC number', 'rxn', 'aa_sequence', 'active_site'], how='inner')
    df['seq_crc64'] = df['aa_sequence'].apply(calculate_crc64)
    df['len_seq'] = df['aa_sequence'].apply(len)

    overlap_result = []
    fpr_result = []
    for i, row in df.iterrows():
        sequence = row['aa_sequence']
        seq_crc64 = row['seq_crc64']

        active_sites = ast.literal_eval(row['active_site'])
        predicted_sites_pfam = ast.literal_eval(row['predicted_sites_pfam'])
        predicted_sites_rxnaamapper = ast.literal_eval(row['predicted_sites_rxnaamapper'])

        pfam_result = get_overlap_and_penalty_score(
            predicted_sites_pfam,
            active_sites,
            sequence
        )
        rxnaamapper_result = get_overlap_and_penalty_score(
            predicted_sites_rxnaamapper,
            active_sites,
            sequence
        )

        current_overlap_result = {
            "pfam_overlap_score": pfam_result['overlap_score'],
            "rxnaamapper_overlap_score": rxnaamapper_result['overlap_score'],
        }

        current_fpr_result = {
            "pfam_false_positive_rate": pfam_result['false_positive_rate'],
            "rxnaamapper_false_positive_rate": rxnaamapper_result['false_positive_rate'],
        }

        n_residues = len(slices_to_indices(predicted_sites_rxnaamapper))

        # ============ our model ====================

        with open(parent_path / f"attn_indices_descending_{seq_crc64}.json", "r", encoding="utf-8") as file:
            attention_ordered = json.load(file)

        for k, v in attention_ordered.items():
            top_n_residues = v[:n_residues]
            slices = indices_to_slices(top_n_residues)
            head_result = get_overlap_and_penalty_score(slices, active_sites, sequence)
            current_overlap_result[f"{k}_overlap_score"] = head_result['overlap_score']
            current_fpr_result[f"{k}_false_positive_rate"] = head_result['false_positive_rate']

        overlap_result.append(current_overlap_result)
        fpr_result.append(current_fpr_result)

    overlap_result = pd.DataFrame(overlap_result)
    fpr_result = pd.DataFrame(fpr_result)
    print(overlap_result.mean().sort_values(ascending=False))
    print()
    print(fpr_result.mean())


# random prediction. overlap score will be around 0.48.
def score_random():
    df_pfam = pd.read_csv(DefaultPath().data_original_rxnaamapper_predictions / 'pfam.csv')
    df_pfam = df_pfam.rename(columns={'predicted_active_site': 'predicted_sites_pfam'})
    df_rxnaamapper = pd.read_csv(DefaultPath().data_original_rxnaamapper_predictions / 'rxnaamapper.csv')
    df_rxnaamapper = df_rxnaamapper.rename(columns={'predicted_active_site': 'predicted_sites_rxnaamapper'})
    df = pd.merge(df_pfam, df_rxnaamapper, on=['pdb-id', 'EC number', 'rxn', 'aa_sequence', 'active_site'], how='inner')
    df['seq_crc64'] = df['aa_sequence'].apply(calculate_crc64)
    df['len_seq'] = df['aa_sequence'].apply(len)

    overlap_result = []
    fpr_result = []
    for i, row in df.iterrows():
        sequence = row['aa_sequence']
        seq_crc64 = row['seq_crc64']

        active_sites = ast.literal_eval(row['active_site'])
        predicted_sites_pfam = ast.literal_eval(row['predicted_sites_pfam'])
        predicted_sites_rxnaamapper = ast.literal_eval(row['predicted_sites_rxnaamapper'])

        pfam_result = get_overlap_and_penalty_score(
            predicted_sites_pfam,
            active_sites,
            sequence
        )
        rxnaamapper_result = get_overlap_and_penalty_score(
            predicted_sites_rxnaamapper,
            active_sites,
            sequence
        )

        current_overlap_result = {
            "pfam_overlap_score": pfam_result['overlap_score'],
            "rxnaamapper_overlap_score": rxnaamapper_result['overlap_score'],
        }

        current_fpr_result = {
            "pfam_false_positive_rate": pfam_result['false_positive_rate'],
            "rxnaamapper_false_positive_rate": rxnaamapper_result['false_positive_rate'],
        }

        n_residues = len(slices_to_indices(predicted_sites_rxnaamapper))

        # ============ random baseline ====================

        attention_ordered = {f"random_{i}": list(range(len(sequence))) for i in range(20)}
        attention_ordered = {k: random.sample(v, len(v)) for k, v in attention_ordered.items()}

        for k, v in attention_ordered.items():
            top_n_residues = v[:n_residues]
            slices = indices_to_slices(top_n_residues)
            head_result = get_overlap_and_penalty_score(slices, active_sites, sequence)
            current_overlap_result[f"{k}_overlap_score"] = head_result['overlap_score']
            current_fpr_result[f"{k}_false_positive_rate"] = head_result['false_positive_rate']

        overlap_result.append(current_overlap_result)
        fpr_result.append(current_fpr_result)

    overlap_result = pd.DataFrame(overlap_result)
    fpr_result = pd.DataFrame(fpr_result)
    print(overlap_result.mean().sort_values(ascending=False))
    print()
    print(fpr_result.mean())

# not being used
def score_per_sequence():
    df_pfam = pd.read_csv(DefaultPath().data_original_rxnaamapper_predictions / 'pfam.csv')
    df_pfam = df_pfam.rename(columns={'predicted_active_site': 'predicted_sites_pfam'})
    df_rxnaamapper = pd.read_csv(DefaultPath().data_original_rxnaamapper_predictions / 'rxnaamapper.csv')
    df_rxnaamapper = df_rxnaamapper.rename(columns={'predicted_active_site': 'predicted_sites_rxnaamapper'})
    df = pd.merge(df_pfam, df_rxnaamapper, on=['pdb-id', 'EC number', 'rxn', 'aa_sequence', 'active_site'], how='inner')
    df['seq_crc64'] = df['aa_sequence'].apply(calculate_crc64)
    df['len_seq'] = df['aa_sequence'].apply(len)

    result = []
    for sequence, df_grouped in df.groupby('aa_sequence'):
        active_sites = df_grouped['active_site'].tolist()
        active_sites = parse_slice_texts(active_sites)
        predicted_sites_pfam = df_grouped['predicted_sites_pfam'].tolist()
        predicted_sites_pfam = parse_slice_texts(predicted_sites_pfam)
        predicted_sites_rxnaamapper = df_grouped['predicted_sites_rxnaamapper'].tolist()
        predicted_sites_rxnaamapper = parse_slice_texts(predicted_sites_rxnaamapper)

        pfam_result = get_overlap_and_penalty_score(
            predicted_sites_pfam,
            active_sites,
            sequence
        )
        rxnaamapper_result = get_overlap_and_penalty_score(
            predicted_sites_rxnaamapper,
            active_sites,
            sequence
        )

        result.append({
            "pfam_overlap_score": pfam_result['overlap_score'],
            "pfam_false_positive_rate": pfam_result['false_positive_rate'],
            "rxnaamapper_overlap_score": rxnaamapper_result['overlap_score'],
            "rxnaamapper_false_positive_rate": rxnaamapper_result['false_positive_rate'],
        })
    result = pd.DataFrame(result)
    print(result.mean())


# def main():
#     key = '241226_164549'
#     score(DefaultPath().build / 'esm_attention' / f'esm_{key}')

def main():
    parser = argparse.ArgumentParser(description='Run score with a specific key')
    parser.add_argument('model', type=str, help='The key to identify the file', default='250420_121652')
    args = parser.parse_args()
    model = args.model
    score(DefaultPath().build / 'esm_attention' / f'esm_{model}')


if __name__ == '__main__':
    main()
    # score(DefaultPath().build / 'esm_attention' / f'esm_250420_121652')
    # print()
    # score_random()
    # score_per_sequence()
