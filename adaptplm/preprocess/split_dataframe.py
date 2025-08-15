# def split_dataframe(dataframe_file: Path, out_dir: Path, out_file_prefix: str, train_ratio=0.8,
#                     val_ratio=0.1, test_ratio=0.1, seed=42, header=True):
#     assert math.isclose(train_ratio + val_ratio + test_ratio, 1), \
#         f"The sum of train_ratio, val_ratio, and test_ratio must be 1. Current: {train_ratio + val_ratio + test_ratio}"
#
#     df = pd.read_csv(dataframe_file)
#
#     df_train_val, df_test = train_test_split(df, test_size=test_ratio, random_state=seed)
#     # NOTE: Using a different calculation method may result in uneven splits when 100 is the input.
#     # Specifying test_size is the standard approach, so this process seems appropriate.
#     val_size_adjusted = val_ratio / (1 - test_ratio)
#     df_train, df_val = train_test_split(df_train_val, test_size=val_size_adjusted, random_state=seed)
#
#     combined_df = pd.concat([df_train, df_val, df_test])
#     duplicates = combined_df[combined_df.duplicated(keep=False)]
#     assert duplicates.empty, "Duplicates found"
#
#     out_dir.mkdir(parents=True, exist_ok=True)
#     df_train.to_csv(out_dir.joinpath(f"{out_file_prefix}_train.csv"), index=False, header=header)
#     df_val.to_csv(out_dir.joinpath(f"{out_file_prefix}_val.csv"), index=False, header=header)
#     df_test.to_csv(out_dir.joinpath(f"{out_file_prefix}_test.csv"), index=False, header=header)
#     logging.info(f"Data split completed. File has been saved to: {out_dir}")
