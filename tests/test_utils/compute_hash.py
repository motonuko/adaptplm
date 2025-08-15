from adaptplm.core.default_path import DefaultPath
from tests.test_utils.hash import calculate_file_hash

if __name__ == '__main__':
    print(calculate_file_hash(DefaultPath().data_dataset_processed.joinpath('enzsrp_cleaned_filtered.csv')))
