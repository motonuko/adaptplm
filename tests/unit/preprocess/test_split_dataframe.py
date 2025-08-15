# import tempfile
# import unittest
# from pathlib import Path
#
# import pandas as pd
#
# from adaptplm.preprocess.split_dataframe import split_dataframe
#
#
# class TestSplitDataFrame(unittest.TestCase):
#
#     def setUp(self):
#         """Set up a temporary directory and sample DataFrame for testing."""
#         self.temp_dir = tempfile.TemporaryDirectory()
#         self.out_dir = Path(self.temp_dir.name)
#         self.data = {
#             "column1": range(100),
#             "column2": [f"value_{i}" for i in range(100)],
#         }
#         self.df = pd.DataFrame(self.data)
#         self.input_file = self.out_dir / "test_dataframe.csv"
#         self.df.to_csv(self.input_file, index=False)
#
#     def tearDown(self):
#         """Clean up temporary directory."""
#         self.temp_dir.cleanup()
#
#     def test_split_ratios(self):
#         """Test if the function correctly splits the DataFrame into train, val, and test."""
#         split_dataframe(
#             dataframe_file=self.input_file,
#             out_dir=self.out_dir,
#             out_file_prefix="split",
#             train_ratio=0.7,
#             val_ratio=0.2,
#             test_ratio=0.1,
#             seed=123,
#         )
#
#         train_file = self.out_dir / "split_train.csv"
#         val_file = self.out_dir / "split_val.csv"
#         test_file = self.out_dir / "split_test.csv"
#
#         # Check if files are created
#         self.assertTrue(train_file.exists())
#         self.assertTrue(val_file.exists())
#         self.assertTrue(test_file.exists())
#
#         # Load the output files
#         df_train = pd.read_csv(train_file)
#         df_val = pd.read_csv(val_file)
#         df_test = pd.read_csv(test_file)
#
#         # Check split sizes
#         total_rows = len(self.df)
#         self.assertEqual(len(df_train), round(0.7 * total_rows))
#         self.assertEqual(len(df_val), round(0.2 * total_rows))
#         self.assertEqual(len(df_test), total_rows - len(df_train) - len(df_val))
#
#         # Ensure no duplicates across splits
#         combined_df = pd.concat([df_train, df_val, df_test])
#         self.assertEqual(len(combined_df), total_rows)
#         self.assertFalse(combined_df.duplicated().any())
#
#     def test_invalid_ratios(self):
#         """Test if the function raises an assertion error for invalid split ratios."""
#         with self.assertRaises(AssertionError):
#             split_dataframe(
#                 dataframe_file=self.input_file,
#                 out_dir=self.out_dir,
#                 out_file_prefix="invalid_split",
#                 train_ratio=0.6,
#                 val_ratio=0.3,
#                 test_ratio=0.2,
#                 seed=123,
#             )
#
#     def test_output_directory_creation(self):
#         """Test if the function creates the output directory if it doesn't exist."""
#         non_existing_dir = self.out_dir / "new_subdir"
#         split_dataframe(
#             dataframe_file=self.input_file,
#             out_dir=non_existing_dir,
#             out_file_prefix="split",
#             train_ratio=0.8,
#             val_ratio=0.1,
#             test_ratio=0.1,
#             seed=123,
#         )
#         self.assertTrue(non_existing_dir.exists())
#
#     def test_seed_consistency(self):
#         """Test if the function produces consistent results with the same seed."""
#         split_dataframe(
#             dataframe_file=self.input_file,
#             out_dir=self.out_dir,
#             out_file_prefix="split1",
#             train_ratio=0.8,
#             val_ratio=0.1,
#             test_ratio=0.1,
#             seed=42,
#         )
#         split_dataframe(
#             dataframe_file=self.input_file,
#             out_dir=self.out_dir,
#             out_file_prefix="split2",
#             train_ratio=0.8,
#             val_ratio=0.1,
#             test_ratio=0.1,
#             seed=42,
#         )
#
#         df_train1 = pd.read_csv(self.out_dir / "split1_train.csv")
#         df_train2 = pd.read_csv(self.out_dir / "split2_train.csv")
#         pd.testing.assert_frame_equal(df_train1, df_train2)
#
#         df_val1 = pd.read_csv(self.out_dir / "split1_val.csv")
#         df_val2 = pd.read_csv(self.out_dir / "split2_val.csv")
#         pd.testing.assert_frame_equal(df_val1, df_val2)
#
#         df_test1 = pd.read_csv(self.out_dir / "split1_test.csv")
#         df_test2 = pd.read_csv(self.out_dir / "split2_test.csv")
#         pd.testing.assert_frame_equal(df_test1, df_test2)
#
#
# if __name__ == "__main__":
#     unittest.main()
