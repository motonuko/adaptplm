from typing import List, Dict, Any, Tuple


# copied from: https://github.com/rxn4chemistry/rxnaamapper/blob/main/src/rxn_aa_mapper/aa_mapper.py
# removed 'self' argument
# set is used instead of list internal.
# NOTE: overlap score は （重複してる binding site / 真の binding site） なので，予測が多くなるほどパフォーマンスが上がる，多分．
# "Among the binding sites predicted by RXNAAMapper, up to 52.13% overlap with the ground truth" この記述がちょっとミスリーディングかも
# Among the binding sites が分母って意味じゃなく，head の意味だな．head ごとの予測のなかで一番良かったのはって話をしている．
# https://doi.org/10.1016/j.csbj.2024.04.012
def get_overlap_and_penalty_score(
        predicted_tokens: List[Tuple[int, int]],
        ground_truth_token_indices: List[Tuple[int, int]],
        aa_sequence: str,
) -> Dict[str, Any]:
    """Computer the overlapping score between two lists of interval.
    Args:
        predicted_tokens: boundaries of the tokens predicted as the active sites of the enzyme.
        ground_truth_tokens: boundaries of the tokens experimentally determined as the active sites of the enzyme.
        aa_sequence: amino acid sequence.

    Returns:
        a dictionary the overlap score, the fpr and other metrics.
    """
    lst_pred = []
    for i, j in predicted_tokens:
        lst_pred.extend([k for k in range(i, j)])

    lst_pred = set(lst_pred)

    lst_gt = []
    for i, j in ground_truth_token_indices:
        lst_gt.extend([k for k in range(i, j)])

    lst_gt = set(lst_gt)

    amino = [i for i in range(len(aa_sequence))]

    TP = len(lst_pred.intersection(lst_gt))
    FP = len(lst_pred.difference(lst_gt))
    TN = len(set(amino).difference((lst_gt).union(lst_pred)))

    if (FP + TP) != len(lst_pred):
        print("there is a mistake in the calculations")

    output = {
        "overlap_score": TP / len(lst_gt),
        "false_positive_rate": FP / (FP + TN),
    }

    return output
