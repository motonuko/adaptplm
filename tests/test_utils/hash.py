import hashlib
import json
from unittest import TestCase

from adaptplm.core.default_path import DefaultPath


def calculate_file_hash(file_path):
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def dataframe_to_hash(df):
    df_sorted = df.sort_index().sort_index(axis=1)
    df_str = df_sorted.to_csv(index=False)
    return hashlib.sha256(df_str.encode('utf-8')).hexdigest()


def json_to_hash(json_data):
    json_str = json.dumps(json_data, sort_keys=True)
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


class TestHash(TestCase):
    def test_hash(self):
        path = DefaultPath().test_data_dir.joinpath("hash_test.txt")
        content = "hogehoge hugahuga"
        with open(path, 'w') as f:
            f.write(content)
        first_hash = calculate_file_hash(path)
        with open(path, 'w') as f:
            f.write(content)
        second_hash = calculate_file_hash(path)
        self.assertEqual(first_hash, second_hash)
