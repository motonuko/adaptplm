import pandas as pd

from adaptplm.core.default_path import DefaultPath
from adaptplm.data.enz_seq_rxn_datasource import load_enz_seq_rxn_datasource
from adaptplm.domain.regex_tokenizer import MultipleSmilesTokenizer
from adaptplm.viz.analysis.add_crc_to_enzsrp2 import check_ec


def mainx():
    df = load_enz_seq_rxn_datasource(
        DefaultPath().data_dataset_raw.joinpath('enzsrp.csv'), need_hash=True)
    # NOTE: single Rhea ID can be related to multiple EC numbers
    df_ec = pd.read_csv(DefaultPath().data_dataset_raw.joinpath('rhea2ec.tsv'), sep="\t")
    df_new = pd.merge(df, df_ec, left_on='rhea_master_id', right_on='MASTER_ID', how='left')
    df_new["integrated_ec"] = df_new.apply(check_ec, axis=1)

    # https://enzyme.expasy.org/cgi-bin/enzyme/enzyme-search-cc
    # expacy keyword: flavin dependent halogenase
    halogenase_ecs = ["1.14.19.9", "1.14.19.58", "1.14.19.59"]
    # expacy keyword: eta-keto acid cleavage enzymes
    bkace_ecs = ["2.3.1.247", "2.3.1.317", "2.3.1.318", "2.3.1.319"]

    filtered_halogenase = df_new[df_new['integrated_ec'].isin(halogenase_ecs)]
    filtered_duf = df_new[df_new['integrated_ec'].isin(bkace_ecs)]
    # filtered_halogenase = df_new[df_new['ec_number'].isin(halogenase_ecs)]
    # filtered_duf = df_new[df_new['ec_number'].isin(bkace_ecs)]
    print(f"{len(filtered_halogenase)}, {len(filtered_duf)}")

    tokenizer = MultipleSmilesTokenizer()

    for rxn in filtered_halogenase['rxn']:
        tokenized = tokenizer.tokenize(rxn)
        print(len(tokenized))

    print("###")

    for rxn in filtered_duf['rxn']:
        tokenized = tokenizer.tokenize(rxn)
        print(len(tokenized))


if __name__ == '__main__':
    mainx()
