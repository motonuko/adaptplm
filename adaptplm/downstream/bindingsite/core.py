import ast
from typing import List, Tuple, Set


def parse_slice_texts(slice_texts: List[str]) -> List[Tuple[int, int]]:
    sites = [region for e in slice_texts for region in ast.literal_eval(e)]
    assert all([len(site) == 2 for site in sites])
    sites = [(site[0], site[1]) for site in sites]
    return sites


def slices_to_indices(slices: List[List[int]] | List[Tuple[int, int]]) -> Set[int]:
    assert all([len(sl) == 2 for sl in slices])
    sites = [list(range(sl[0], sl[1])) for sl in slices]
    sites = {x for sublist in sites for x in sublist}
    return sites


def indices_to_slices(indices: List[int]) -> List[Tuple[int, int]]:
    if not indices:
        return []
    ordered = list(sorted(indices))

    slices = []
    start = ordered[0]
    prev = ordered[0]

    for i in ordered[1:]:
        if i == prev + 1:
            prev = i
        else:
            slices.append([start, prev + 1])
            start = i
            prev = i
    slices.append((start, prev + 1))
    return slices
