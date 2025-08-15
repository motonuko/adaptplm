# class Test(TestCase):
#     def test_calculate_sampling_weight(self):
#         train_path = DefaultPath().data_dataset_processed / 'enzsrp_cleaned' / 'enzsrp_cleaned_train.csv'
#         df = load_enz_seq_rxn_datasource(train_path)
#
#         data_path = DefaultPath().build.joinpath("enzsrp_cdhit_output.clstr")
#         result = calculate_sampling_weight(df['sequence'], CdHitResultDatasource(data_path))
#         # TODO: test
