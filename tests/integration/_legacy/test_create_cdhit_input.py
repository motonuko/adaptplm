# import unittest
#
# from enzrxnpred2.app.preprocess.enzseqrxn.cd_hit_utils import create_cd_hit_input, \
#     parse_cd_hit_result_and_find_dense_screen, find_sequences_similar_to_holdout_seqs
# from enzrxnpred2.utils.data_path import DataPath
# from test.test_utils.hash import calculate_file_hash, json_to_hash
#
#
# class MyTestCase(unittest.TestCase):
#     def test_something(self):
#         output_path = DataPath.build.joinpath('cdhit_input.fasta')
#         create_cd_hit_input(DataPath.data_original_dense_screen_processed,
#                             DataPath.data_dataset_dir.joinpath('enzyme_reaction.csv'),
#                             output_path)
#         output_hash = calculate_file_hash(output_path)
#         # self.assertEqual(output_hash, 'eb8479a41864a4ff901240c9470ec0ecdc53d69a2d301c2d925d9d2ab6a5d3ae')
#
#     def test_something2(self):
#         input_f = DataPath.build.joinpath('cdhit_output.clstr')
#         out = DataPath.build.joinpath('holdout_cluster_ids.txt')
#         parse_cd_hit_result_and_find_dense_screen(input_file=input_f, output_file=out)
#         output_hash = calculate_file_hash(out)
#         self.assertEqual(output_hash, 'abeb6344408977ac1172a14e124a2be451e1a80798aa39626daa16779086f38f')
#
#     def test_something3(self):
#         input_f = DataPath.build.joinpath('cdhit_output.clstr')
#         result = find_sequences_similar_to_holdout_seqs(input_file=input_f)
#
#         output_hash = json_to_hash({'result': result})
#         self.assertEqual(output_hash, 'e1c86155eefff9a03dfaf321fbaa59182b1787e2ce4fe6e2016cfc1bded26717')
#
#
# if __name__ == '__main__':
#     unittest.main()
