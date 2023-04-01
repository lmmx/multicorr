from __future__ import annotations

from functools import partial
from itertools import chain, combinations, product

import numpy as np
import pandas as pd

cat_sep = ":"
lvl_sep = "_"

# Load your dataset
data = pd.read_csv("simple_dataset.csv")

cat_cols = ["c1", "c2", "c3"]
target_column = "v"
target = "null_target_column"

# Create a binary column to indicate the presence of null values in the target column
null_indicator = data[target_column].isnull()
data[target] = null_indicator.astype(int)
original_cols = data.columns


def name_multicat(s: pd.Series) -> str:
    return cat_sep.join(map(lvl_sep.join, s.items()))


def unpack_multicat(name: str) -> pd.Series:
    return pd.Series(
        {
            k: v
            for cat in name.split(cat_sep)
            for k, v in dict([cat.split(lvl_sep)]).items()
        }
    )


def dummy_2d(s: pd.Series) -> pd.Series:
    return pd.Series({name_multicat(s): 1})


def check_dummies_2d(
    row: pd.Series, dummy_cols: pd.Index, null_levelsets: list[tuple[str, ...]]
) -> pd.Series:
    """Stringifies input series (row) and gives a bool of whether it's a dummy var."""
    checks = [
        dummy_cols.str.fullmatch(name_multicat(row[null_levels].astype(str)))
        for null_levels in null_levelsets
    ]
    return pd.DataFrame(checks, columns=dummy_cols).any()


def null_levelsets(cat_cols):
    n_cats = len(cat_cols)
    idx_combos = list(
        chain.from_iterable(
            combinations(range(n_cats), r=r) for r in range(1, n_cats + 1)
        )
    )
    return [list(map(cat_cols.__getitem__, idxs)) for idxs in idx_combos]


def powerset_dummies_2d(nullspace, null_levelsets):
    ns_combos = [
        nullspace[null_levels].astype(str).apply(dummy_2d, axis=1).fillna(0).astype(int)
        for null_levels in null_levelsets
    ]
    dummies = pd.concat(ns_combos, axis=0).fillna(0).astype(int).drop_duplicates()
    return dummies


null_combos = data[null_indicator][cat_cols].drop_duplicates()
# dummy_vars = null_combos.astype(str).apply(dummy_2d, axis=1).fillna(0).astype(int)
null_ls = null_levelsets(cat_cols)
dummy_vars = powerset_dummies_2d(nullspace=null_combos, null_levelsets=null_ls)
dummy_cols = dummy_vars.columns

# List all possible combinations of the categorical columns
check_2d_dummy_cols = partial(
    check_dummies_2d, dummy_cols=dummy_cols, null_levelsets=null_ls
)
full_dummies = data[cat_cols].apply(check_2d_dummy_cols, axis=1).astype(int)
data_with_dummies = pd.concat([data, full_dummies], axis=1)

# Calculate the correlation between the binary column and the new columns
correlation_matrix = data_with_dummies[[target, *dummy_cols]].corr(numeric_only=True)

# Display the correlations between the binary column and the new columns
correlations_with_target = correlation_matrix[target].loc[dummy_cols]
print(correlations_with_target)

assert correlations_with_target.max() == 1.0
