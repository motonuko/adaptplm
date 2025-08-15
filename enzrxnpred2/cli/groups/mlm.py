import logging
import os

import click
from dotenv import load_dotenv

# NOTE: Somehow direct .env path is needed
env_path = os.path.join(os.getcwd(), '.env')
load_dotenv(env_path)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from adaptplm.cli.commands.mlm import train_seq_rxn_encoder_with_mlm_cli, build_vocab_enzsrp_full_cli


@click.group()
def mlm():
    pass


mlm.add_command(build_vocab_enzsrp_full_cli, 'build-vocab-enzsrp-full')  # type: ignore
mlm.add_command(train_seq_rxn_encoder_with_mlm_cli, 'train-seq-rxn-encoder-with-mlm')  # type: ignore

if __name__ == "__main__":
    mlm()
