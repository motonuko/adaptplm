import logging
from pathlib import Path
from typing import List

import click

from enzrxnpred2.core.default_path import DefaultPath
from enzrxnpred2.core.package_version import get_package_major_version
from enzrxnpred2.downstream.bindingsite.extract_attention import extract_attention_order_per_head
from enzrxnpred2.downstream.common.compute_sentence_embedding import compute_sentence_embedding
from enzrxnpred2.downstream.common.compute_sentence_embedding2 import compute_sentence_embedding2
from enzrxnpred2.downstream.cpi.nested_cv import run_nested_cv_on_enz_activity_cls
from enzrxnpred2.downstream.embedding.compute_silhauette_score import compute_clustering_scores


@click.command()
@click.option('--exp-config', type=click.Path(exists=True))
@click.option('--output-parent-dir', type=click.Path(),
              default=DefaultPath().local_exp_nested_cv_train / get_package_major_version())
@click.option('--dense-screen-dir', type=click.Path(),
              default=DefaultPath().data_original_dense_screen_processed)
@click.option('--additional-dense-screen-dir', type=click.Path(),
              default=DefaultPath().data_dataset_processed / 'cpi')
def run_nested_cv_on_enz_activity_cls_cli(exp_config: str, output_parent_dir: str, dense_screen_dir: str,
                                          additional_dense_screen_dir: str):
    # process_start_time = time.time()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_nested_cv_on_enz_activity_cls(
        Path(exp_config),
        Path(output_parent_dir),
        Path(dense_screen_dir),
        Path(additional_dense_screen_dir),
    )
    # logger.info(f"execution time: {time.time() - process_start_time :.2f} seconds")


@click.command()
@click.option('--data-path', type=click.Path(exists=True),
              help='The path to a file where sequences are saved, separated by "\n".')
@click.option('--model-path', type=click.Path(), help='The path to ESM model dir')
@click.option('--output-csv', type=click.Path(), help='')
@click.option('--seed', type=int, default=42, help='')
@click.option('--pooling-type', type=str, default='pooler', help='pooler or mean')
# Max token length of ESP model is 1022 (including special tokens)
# https://github.com/AlexanderKroll/kcat_prediction/blob/main/code/preprocessing/02%20-%20Calculate%20reaction%20fingerprints%20and%20enzyme%20representations.ipynb
# https://github.com/AlexanderKroll/ESP/blob/main/notebooks_and_code/extract.py#L99
@click.option('--max-seq-len', type=int, default=None)
@click.option('--mixed-precision', type=str, default='fp16')
@click.option('--batch-size', type=int, default=8)
def compute_sentence_embedding_cli(data_path: str, model_path: str, output_csv: str, seed: int, max_seq_len: int,
                                   pooling_type: str, mixed_precision: str, batch_size: int):
    compute_sentence_embedding(data_path=Path(data_path), model_path=Path(model_path), output_csv=Path(output_csv),
                               seed=seed, max_seq_len=max_seq_len, pooling_type=pooling_type,
                               mixed_precision=mixed_precision, batch_size=batch_size)


@click.command()
@click.option('--data-path', type=click.Path(exists=True),
              help='The path to a file where sequences are saved, separated by "\n".')
@click.option('--model-path', type=click.Path(), help='The path to ESM model dir')
@click.option('--output-npy', type=click.Path(), help='')  # TODO: rename
@click.option('--seed', type=int, default=42, help='')
@click.option('--pooling-type', type=str, default='pooler', help='pooler or mean')
# Max token length of ESP model is 1022 (including special tokens)
# https://github.com/AlexanderKroll/kcat_prediction/blob/main/code/preprocessing/02%20-%20Calculate%20reaction%20fingerprints%20and%20enzyme%20representations.ipynb
# https://github.com/AlexanderKroll/ESP/blob/main/notebooks_and_code/extract.py#L99
@click.option('--max-seq-len', type=int, default=None)
@click.option('--mixed-precision', type=str, default='fp16')
@click.option('--batch-size', type=int, default=8)
def compute_sentence_embedding_cli2(data_path: str, model_path: str, output_npy: str, seed: int, max_seq_len: int,
                                    pooling_type: str, mixed_precision: str, batch_size):
    compute_sentence_embedding2(data_path=Path(data_path), model_path=Path(model_path), output_npy=Path(output_npy),
                                seed=seed, max_seq_len=max_seq_len, pooling_type=pooling_type,
                                mixed_precision=mixed_precision, batch_size=batch_size)

@click.command()
@click.option('--data-path', type=click.Path(exists=True), help='')
@click.option('--model-path', type=click.Path(), help='')
@click.option('--output-dir', type=click.Path(), help='')
@click.option('--seed', type=int, default=42, help='')
def extract_attention_order_per_head_cli(data_path: str, model_path: str, output_dir: str, seed: int):
    extract_attention_order_per_head(data_path=Path(data_path), model_path=Path(model_path),
                                     output_dir=Path(output_dir), seed=seed)


@click.command()
@click.option("--seq-ec-file-path", type=click.Path(exists=True))
@click.option("--embedding-files", multiple=True)
def compute_clustering_scores_cli(seq_ec_file_path: str, embedding_files: List[str]):
    compute_clustering_scores(seq_ec_file_path, embedding_files)
