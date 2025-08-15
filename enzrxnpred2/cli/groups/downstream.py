import logging
import os

import click
from dotenv import load_dotenv

# NOTE: Somehow direct .env path is needed
env_path = os.path.join(os.getcwd(), '.env')
load_dotenv(env_path)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from adaptplm.cli.commands.downstream import run_nested_cv_on_enz_activity_cls_cli, \
    extract_attention_order_per_head_cli, compute_sentence_embedding_cli, compute_sentence_embedding_cli2, \
    compute_clustering_scores_cli


@click.group()
def downstream():
    pass


downstream.add_command(run_nested_cv_on_enz_activity_cls_cli, 'run-nested-cv-on-enz-activity-cls')  # type: ignore
downstream.add_command(compute_sentence_embedding_cli, 'compute-sentence-embedding')  # type: ignore
downstream.add_command(compute_sentence_embedding_cli2, 'compute-sentence-embedding2')  # type: ignore
downstream.add_command(extract_attention_order_per_head_cli, 'extract-attention-order-per-head')  # type: ignore
downstream.add_command(compute_clustering_scores_cli, 'compute-clustering-scores')  # type: ignore

if __name__ == "__main__":
    downstream()
