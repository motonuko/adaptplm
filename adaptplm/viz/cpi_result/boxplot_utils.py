def min_max(df, target_column):
    worst_idx = df[target_column].idxmin()
    best_idx = df[target_column].idxmax()
    filtered_df = df.loc[[worst_idx, best_idx]]
    return filtered_df


def sort_key(name):
    if name == "Mean":
        return 0
    elif name.startswith("Active Site"):
        return 1
    elif name.startswith("Precomputed"):
        return 2
    elif name.startswith("Masked LM"):
        return 3
    return 4


def split_active_site(df):
    # NOTE: 'Active Site' strategy is not used in the paper
    starts_with_a = df['condition'].str.startswith('Active Site')
    df_not_a = df[~starts_with_a].reset_index(drop=True)  # NOT starting from "a"
    df_a = df[starts_with_a].reset_index(drop=True)  # starting from "a"
    return df_not_a, df_a
