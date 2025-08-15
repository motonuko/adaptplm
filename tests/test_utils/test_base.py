# import tempfile
# import unittest
# from pathlib import Path
#
#
# class MyTestCase(unittest.TestCase):
#     def setUp(self):
#         self.temp_dir = tempfile.TemporaryDirectory()
#         self.temp_dir_path = Path(self.temp_dir.name)
#
#     def tearDown(self):
#         self.temp_dir.cleanup()
#
#     def test_something(self):
#         pass
