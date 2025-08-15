from adaptplm.core.default_path import DefaultPath
from adaptplm.data.enz_seq_rxn_datasource import load_enz_seq_rxn_datasource
from adaptplm.domain.regex_tokenizer import MultipleSmilesTokenizer


def main():
    df_enz = load_enz_seq_rxn_datasource(DefaultPath().data_dataset_raw.joinpath('enzsrp_full.csv'))
    rxns = set(df_enz['rxn'].values)
    rxns = list(rxns)[:100]
    tokenizer = MultipleSmilesTokenizer()
    for rxn in rxns:
        tokenized = tokenizer.tokenize(rxn)
        print(' '.join(tokenized))


if __name__ == '__main__':
    main()
