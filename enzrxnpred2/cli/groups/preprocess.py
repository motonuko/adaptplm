import logging
import os

import click
from dotenv import load_dotenv

# NOTE: Somehow direct .env path is needed
env_path = os.path.join(os.getcwd(), '.env')
load_dotenv(env_path)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from adaptplm.cli.commands.preprocess import clean_enzyme_reaction_pair_full_dataset_cli, \
    split_enzsrp_full_dataset_cli, create_sequence_inputs_for_splitting_cli, create_sequence_inputs_for_analysis_cli


@click.group()
def preprocess():
    pass


preprocess.add_command(clean_enzyme_reaction_pair_full_dataset_cli,  # type: ignore
                       'clean-enzyme-reaction-pair-full-dataset')
preprocess.add_command(split_enzsrp_full_dataset_cli, 'split-enzsrp-full-dataset')  # type: ignore
preprocess.add_command(create_sequence_inputs_for_splitting_cli, 'create-sequence-inputs-for-splitting')  # type: ignore
preprocess.add_command(create_sequence_inputs_for_analysis_cli, 'create-sequence-inputs-for-analysis')  # type: ignore

if __name__ == "__main__":
    preprocess()
