import argparse
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml

from adaptplm.core.default_path import DefaultPath
from adaptplm.core.package_version import get_package_major_version
from adaptplm.data.original_enz_activity_dense_screen_datasource import datasets_used_in_our_paper
from adaptplm.downstream.cpi.domain.protein_lm_embedding import EmbeddingType
from adaptplm.viz.cpi_result.summarize_for_box_plot import parse


# Summarize results for drawing them in one figure.
def summarize_results(dataset, exp_result_dirs: List[Path], save_dir: Path, models: Optional[List[str]]):
    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    # seeds = [42]
    results = []
    seed_first_results = {}
    for seed in seeds:
        for result_dir in exp_result_dirs:
            try:
                parsed = parse(result_dir / f"result_seed_{seed}.json")
                if (parsed.protein_feat_config.embedding_config.embedding_type == EmbeddingType.RXN_MLM and all(
                        model not in parsed.protein_feat_config.model_name for model in models)):
                    continue
                if hoge := seed_first_results.get(seed, None):
                    assert hoge.hash_key == parsed.hash_key, f"\n {parsed} \n {hoge}"
                else:
                    seed_first_results[seed] = parsed
                results.append(parsed)
            except FileNotFoundError as e:
                print(result_dir, seed)
                pass

    grouped = defaultdict(list)
    for result in results:
        grouped[result.group_key].append(result)

    # for one_exp in grouped.values():
    #     assert len(one_exp) == len(
    #         seeds), f"{len(one_exp)}, {len(seeds)}, {one_exp[0].protein_feat_config}, {one_exp[0].data}"

    table = []
    for key, group in grouped.items():
        roc_auc_results = [exp.averaged_fold_results_avg_roc_auc_mean for exp in group]
        pr_auc_results = [exp.averaged_fold_results_avg_pr_auc_mean for exp in group]
        raw_roc_auc_results = [f.roc_auc for exp in group for f in exp.fold_results]
        raw_pr_auc_results = [f.pr_auc for exp in group for f in exp.fold_results]
        table.append({
            'dataset': dataset.value,
            'condition': key,
            'roc_auc_mean': np.mean(roc_auc_results),
            'roc_auc_std': np.std(roc_auc_results),
            'pr_auc_mean': np.mean(pr_auc_results),
            'pr_auc_std': np.std(pr_auc_results),
            'roc_auc_mean_per_trial': roc_auc_results,
            'pr_auc_mean_per_trial': pr_auc_results,
            'roc_auc_per_outer_fold': raw_roc_auc_results,
            'pr_auc_per_outer_fold': raw_pr_auc_results,
        })
    df = pd.DataFrame(table)
    df = df.sort_values(by='condition')
    # df.to_csv(save_dir / f"cv_trials_summary_{dataset.value}.csv", index=False)
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_pickle(save_dir / f"cv_trials_summary_{dataset.value}.pkl")


# All result files expected to be saved in the same dir.
def load_exp_result_paths(exp_path_map_file, exp_base_path):
    with open(exp_path_map_file, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return [exp_base_path / res for res in data['results']]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['250420_121652'])
    args = parser.parse_args()

    v = get_package_major_version()
    v_int = int(get_package_major_version(prefix_v=False))
    for d in datasets_used_in_our_paper:
        if v_int >= 11:
            result_path = DefaultPath().data_dir / 'server_exp' / 'cpi' / v / d.value
            if not result_path.exists():
                print(f'Skipped {result_path.as_posix()}')
                continue
            results = [p for p in result_path.iterdir() if p.is_dir()]
        else:
            map_file = DefaultPath().data_dataset_dir / 'viz' / v / f"cv_trial_ids_{d.value}.yaml"
            if not map_file.exists():
                print(f'Skipped {map_file.as_posix()}')
                continue
            results = load_exp_result_paths(
                exp_path_map_file=map_file,
                exp_base_path=DefaultPath().data_dir.joinpath('server_exp', 'cpi', 'cpi', d.value))
        summarize_results(d, results, save_dir=DefaultPath().build / 'fig' / v, models=args.models)
