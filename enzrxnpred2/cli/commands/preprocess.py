from pathlib import Path

import click

from enzrxnpred2.core.default_path import DefaultPath
from enzrxnpred2.preprocess.cdhit2.create_cdhit_input import create_sequence_inputs_for_splitting, \
    create_sequence_inputs_for_analysis
from enzrxnpred2.preprocess.clean_datasets import clean_enzyme_reaction_pair_dataset
from enzrxnpred2.preprocess.splitv2.split_enz_rxn_dataset import split_enz_rxn_dataset


@click.command()
@click.option('--seq-rxn-csv', type=click.Path(exists=True),
              default=DefaultPath().data_dataset_raw and DefaultPath().data_dataset_raw / 'enzsrp_full.csv',
              help='')
@click.option('--output-file', type=click.Path(),
              default=DefaultPath().data_dataset_processed and DefaultPath().data_dataset_processed / 'enzsrp_full_cleaned.csv',
              help='')
def clean_enzyme_reaction_pair_full_dataset_cli(seq_rxn_csv: str, output_file: str):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    clean_enzyme_reaction_pair_dataset(data=Path(seq_rxn_csv), output_file=output_file)


@click.command()
@click.option('--enzsrp-full-path', type=click.Path(),
              default=DefaultPath().data_dataset_processed.joinpath('enzsrp_full_cleaned.csv'),
              help='')
@click.option('--enzsrp-full-train-path', type=click.Path(),
              default=DefaultPath().data_dataset_processed.joinpath('enzsrp_full_cleaned',
                                                                    'enzsrp_full_cleaned_train.csv'),
              help='')
@click.option('--output-dir', type=click.Path(),
              default=DefaultPath().build / 'fasta',
              help='')
def create_sequence_inputs_for_splitting_cli(enzsrp_full_path: str, enzsrp_full_train_path: str, output_dir: str):
    # NOTE: Assuming .env is properly configured
    create_sequence_inputs_for_splitting(
        output_dir=Path(output_dir),
        enzsrp_full_path=Path(enzsrp_full_path),
        enzsrp_full_train_path=Path(enzsrp_full_train_path))


@click.command()
@click.option('--enzsrp-full-train-path', type=click.Path(),
              default=DefaultPath().data_dataset_processed.joinpath('enzsrp_full_cleaned',
                                                                    'enzsrp_full_cleaned_train.csv'),
              help='')
@click.option('--esp-path', type=click.Path(),
              default=DefaultPath().original_esp_fine_tuning_pkl,
              help='')
@click.option('--activity-screen-dir', type=click.Path(),
              default=DefaultPath().data_original_dense_screen_processed,
              help='')
@click.option('--kcat-path', type=click.Path(),
              default=DefaultPath().original_kcat_test_pkl,
              help='')
@click.option('--output-dir', type=click.Path(),
              default=DefaultPath().build / 'fasta',
              help='')
def create_sequence_inputs_for_analysis_cli(enzsrp_full_train_path: str, esp_path: str, activity_screen_dir: str,
                                            kcat_path: str, output_dir: str):
    create_sequence_inputs_for_analysis(Path(enzsrp_full_train_path), Path(esp_path), Path(activity_screen_dir),
                                        Path(kcat_path), Path(output_dir))


@click.command()
@click.option('--enzsrp', type=click.Path(exists=True),
              default=DefaultPath().data_dataset_processed / 'enzsrp_full_cleaned.csv',
              help='')
@click.option('--cd-hit-result', type=click.Path(exists=True),
              default=DefaultPath().build_cdhit / 'enzsrp_full' / "enzsrp_full_80.clstr",
              help='')
@click.option('--output-dir', type=click.Path(),
              default=DefaultPath().data_dataset_processed / 'enzsrp_full_cleaned',
              help='')
@click.option('--val-ratio', type=float,
              default=0.05,
              help='')
@click.option('--test-ratio', type=float,
              default=0.05,
              help='')
def split_enzsrp_full_dataset_cli(
        enzsrp: str,
        cd_hit_result: str,
        output_dir: str,
        val_ratio: float,
        test_ratio: float

):
    split_enz_rxn_dataset(
        enzsrp=Path(enzsrp),
        cd_hit_result=Path(cd_hit_result),
        output_dir=Path(output_dir),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        out_file_prefix='enzsrp_full_cleaned',
    )
