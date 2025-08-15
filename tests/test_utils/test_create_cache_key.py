import unittest
from dataclasses import dataclass
from enum import Enum

from enzrxnpred2._legacy.disk_cache_util import create_cache_key


@dataclass(frozen=True)
class Point:
    x: int
    y: int


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


class TestCreateCacheKey(unittest.TestCase):
    def test_empty_args_and_kwargs(self):
        self.assertEqual(create_cache_key(), create_cache_key())

    def test_positional_args(self):
        self.assertEqual(create_cache_key(1, 2, 3), create_cache_key(1, 2, 3))
        self.assertNotEqual(create_cache_key(1, 2, 3), create_cache_key(3, 2, 1))

    def test_keyword_args(self):
        self.assertEqual(create_cache_key(a=1, b=2), create_cache_key(b=2, a=1))
        self.assertNotEqual(create_cache_key(a=1, b=2), create_cache_key(a=2, b=1))

    def test_mixed_args(self):
        self.assertEqual(create_cache_key(1, 2, a=3, b=4), create_cache_key(1, 2, b=4, a=3))
        self.assertNotEqual(create_cache_key(1, 2, b=4, a=3), create_cache_key(2, 1, a=3, b=4))

    # NOTE: does not support list args.
    # def test_with_different_types(self):
    #     self.assertEqual(create_cache_key(1, "two", a=[1, 2, 3]), create_cache_key(1, "two", a=[1, 2, 3]))
    #     self.assertNotEqual(create_cache_key(1, "two", a=[1, 2, 3]), create_cache_key(1, "two", a=[3, 2, 1]))

    def test_with_dataclass(self):
        # frozen dataclass test
        point = Point(1, 2)
        self.assertEqual(create_cache_key(point), create_cache_key(Point(1, 2)))
        self.assertNotEqual(create_cache_key(point), create_cache_key(Point(2, 1)))

    def test_with_enum(self):
        # enum test
        self.assertEqual(create_cache_key(Color.RED), create_cache_key(Color.RED))
        self.assertNotEqual(create_cache_key(Color.RED), create_cache_key(Color.GREEN))


if __name__ == '__main__':
    unittest.main()
