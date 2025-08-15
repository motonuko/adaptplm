def generate_nested_cv_seeds(seed_list, outer_offset=100, inner_offset=10000):
    seeds = []
    for main_seed in seed_list:
        seeds.append({
            "base_seed": main_seed,
            "outer_cv_seed": main_seed + outer_offset,
            "inner_cv_seed": main_seed + inner_offset,
        })
    return seeds
