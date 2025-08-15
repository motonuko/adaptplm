import os
import tempfile
import unittest
from pathlib import Path

from adaptplm.core.default_path import DefaultPath
from adaptplm.downstream.cpi.nested_cv import run_nested_cv_on_enz_activity_cls
from adaptplm.extension.seed import set_random_seed
from tests.test_utils.hash import calculate_file_hash


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    # def test_something(self):
    #     config_path = DataPath().test_data_dir.joinpath('cpi', 'exp_config_ridge2.json')
    #     self.assertEqual('6ea179df57ed54b98db54730be5651f8f8b7f0cb7aa85eacd77b74625eb197dc',
    #                      calculate_file_hash(config_path))
    #     _run_nested_cv_on_enz_activity_cls(config_path, self.temp_dir_path)
    #
    #     for root, dirs, files in os.walk(self.temp_dir_path):
    #         for file in files:
    #             if file == 'optim_results_seed_42.csv':
    #                 self.assertEqual('f4c78728de83f6925c0625e1c45a19a526feadae6616f496bae03683c779ec9c',
    #                                  calculate_file_hash(root + '/' + file))
    #             if file == 'result_seed_42.json':
    #                 self.assertEqual('858f2d68593e374f4af766354eb270ead61713e41805c6bca15fb8fc4bb670bc',
    #                                  calculate_file_hash(root + '/' + file))

    def test_something2(self):
        config_path = DefaultPath().test_data_dir.joinpath('cpi', 'v8', 'exp_config_fnn_as.json')
        dense_screen_dir = DefaultPath().data_original_dense_screen_processed

        self.assertEqual('665bc93c5dc6b81f5de9c15b1961e7f2085fb9a44e39e2d75c31c7593073e7c4',
                         calculate_file_hash(config_path))
        run_nested_cv_on_enz_activity_cls(config_path, self.temp_dir_path, dense_screen_dir)

        found = 0
        for root, dirs, files in os.walk(self.temp_dir_path):
            for file in files:
                if file == 'optim_results_seed_42.csv':
                    self.assertEqual('e124087397de1fd4d3bcd159b004cea1e28d0364301247bc1452d071134e96a9',
                                     calculate_file_hash(root + '/' + file))
                    found += 1
                if file == 'result_seed_42.json':
                    self.assertEqual('77ea7398339aa88100dc3097eba64140e357bbfc282470cb5dba823cbf256c27',
                                     calculate_file_hash(root + '/' + file))
                    found += 1
        self.assertEqual(found, 2)

    def test_something3(self):
        config_path = DefaultPath().test_data_dir.joinpath('cpi', 'v8', 'exp_config_fnn_mlm.json')
        dense_screen_dir = DefaultPath().data_original_dense_screen_processed
        # Fix the seed to prevent the pooling head from becoming random.
        set_random_seed(42)

        self.assertEqual('05387d769b1fdfbc8b1baf76e358f69fdeb233565b52f718716913953f64f7c3',
                         calculate_file_hash(config_path))
        run_nested_cv_on_enz_activity_cls(config_path, self.temp_dir_path, dense_screen_dir)

        found = 0
        for root, dirs, files in os.walk(self.temp_dir_path):
            for file in files:
                if file == 'optim_results_seed_42.csv':
                    self.assertEqual('3f1f806c9b3bf273f650ea05bdd1a8ab0522742c7d40bf539086023aaa568c40',
                                     calculate_file_hash(root + '/' + file))
                    found += 1
                if file == 'result_seed_42.json':
                    self.assertEqual('b4298b8d1cda589d4af0be95e5a7429f671b2a0077cc37bea5444971f3da1c5b',
                                     calculate_file_hash(root + '/' + file))
                    found += 1
        self.assertEqual(found, 2)

    # def test_something4(self):
    #     config_path = DefaultPath().test_data_dir.joinpath('cpi', 'exp_config_fnn4_precomputed.json')
    #     dense_screen_dir = DefaultPath().data_original_dense_screen_processed
    #     Fix the seed to prevent the pooling head from becoming random.
    #     set_random_seed(42)
    #
    #     self.assertEqual('346ffc089c8c63be58a2d0ed8ce0993c98ef30c4e0e6ae9523f5d5ed23e7fb53',
    #                      calculate_file_hash(config_path))
    #     run_nested_cv_on_enz_activity_cls(config_path, self.temp_dir_path, dense_screen_dir)
    #
    #     found = 0
    #     for root, dirs, files in os.walk(self.temp_dir_path):
    #         for file in files:
    #             if file == 'optim_results_seed_42.csv':
    #                 self.assertEqual('87c8be2f58e7dc5019b60781c67713cc6d376b50567af7eb56cb9437fc24ac93',
    #                                  calculate_file_hash(root + '/' + file))
    #                 found += 1
    #             if file == 'result_seed_42.json':
    #                 self.assertEqual('64b82ae639b1f5d79e6fa53d8349ffbbddfcb790ea0ab56cd2097fbff8ebb83b',
    #                                  calculate_file_hash(root + '/' + file))
    #                 found += 1
    #     self.assertEqual(found, 2)

    # def test_something5(self):
    #     config_path = DefaultPath().test_data_dir.joinpath('cpi', 'v8', 'exp_config_fnn_new_duf.json')
    #     dense_screen_dir = DefaultPath().data_original_dense_screen_processed
    #     additional_dense_screen_dir = DefaultPath().data_dataset_processed / 'cpi'
    # Fix the seed to prevent the pooling head from becoming random.
    #     set_random_seed(42)
    #
    #     self.assertEqual('8f42d041557e830e0def547af562eff3536ab7a61c30a14add8daae4334787b5',
    #                      calculate_file_hash(config_path))
    #     run_nested_cv_on_enz_activity_cls(config_path, self.temp_dir_path, dense_screen_dir,
    #                                       additional_dense_screen_dir)
    #
    #     found = 0
    #     for root, dirs, files in os.walk(self.temp_dir_path):
    #         for file in files:
    #             if file == 'optim_results_seed_42.csv':
    #                 self.assertEqual('ee7ab7c06c01ee2450694f6f2d55c47d90612d3a1a698c84d9f091abc253aea7',
    #                                  calculate_file_hash(root + '/' + file))
    #                 found += 1
    #             if file == 'result_seed_42.json':
    #                 self.assertEqual('76dd3223443ac8c13da5c465cbb03338ed586010a915f7e08039163363ea14e3',
    #                                  calculate_file_hash(root + '/' + file))
    #                 found += 1
    #     self.assertEqual(found, 2)
    #
    # def test_something6(self):
    #     config_path = DefaultPath().test_data_dir.joinpath('cpi', 'v8', 'exp_config_fnn_olea.json')
    #     dense_screen_dir = DefaultPath().data_original_dense_screen_processed
    #     additional_dense_screen_dir = DefaultPath().data_dataset_processed / 'cpi'
    #     Fix the seed to prevent the pooling head from becoming random.
    #     set_random_seed(42)
    #
    #     self.assertEqual('ee2cb6080f36f1afad5f2526612b440533159ad1d1496fcaedf794fdb8b6cbe6',
    #                      calculate_file_hash(config_path))
    #     run_nested_cv_on_enz_activity_cls(config_path, self.temp_dir_path, dense_screen_dir,
    #                                       additional_dense_screen_dir)
    #
    # def test_something7(self):
    #     config_path = DefaultPath().test_data_dir.joinpath('cpi', 'v8', 'exp_config_fnn_phos.json')
    #     dense_screen_dir = DefaultPath().data_original_dense_screen_processed
    #     additional_dense_screen_dir = DefaultPath().data_dataset_processed / 'cpi'
    # Fix the seed to prevent the pooling head from becoming random.
    #     set_random_seed(42)
    #
    #     self.assertEqual('329de2d453e4737a073f377fd2bfee177fb95ee9186ce85cb285df188084ba46',
    #                      calculate_file_hash(config_path))
    #     run_nested_cv_on_enz_activity_cls(config_path, self.temp_dir_path, dense_screen_dir,
    #                                       additional_dense_screen_dir)
    #
    def test_something8(self):
        config_path = DefaultPath().test_data_dir.joinpath('cpi', 'v8', 'exp_config_fnn_gt.json')
        dense_screen_dir = DefaultPath().data_original_dense_screen_processed
        additional_dense_screen_dir = DefaultPath().data_dataset_processed / 'cpi'
        # Fix the seed to prevent the pooling head from becoming random.
        set_random_seed(43)

        self.assertEqual('598670a2d762555d210a059f404ac45dd9b16227ca40d8249c0ea36aee1ce5dd',
                         calculate_file_hash(config_path))
        run_nested_cv_on_enz_activity_cls(config_path, self.temp_dir_path, dense_screen_dir,
                                          additional_dense_screen_dir)


if __name__ == '__main__':
    unittest.main()
