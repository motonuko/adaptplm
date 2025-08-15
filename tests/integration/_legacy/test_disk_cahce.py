import os
import shutil
import unittest

from adaptplm._legacy.disk_cache_util import disk_cache
from adaptplm.utils.data_path import DataPath

cache_dir = DataPath().test_data_dir.joinpath("disk_cache")

if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)  # setUpClass だとタイミングが遅すぎる
os.makedirs(cache_dir)


@disk_cache(base_cache_dir=cache_dir)
def add(a, b):
    return a + b


class TestDiskCache(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cache_dir)

    def test_cache_creation_and_reuse(self):
        result1 = add(1, 2)
        result1_cache = add(1, 2)
        self.assertEqual(result1, result1_cache)

        result2 = add(2, 3)
        result2_cache = add(2, 3)
        self.assertEqual(result2, result2_cache)


if __name__ == '__main__':
    unittest.main()
