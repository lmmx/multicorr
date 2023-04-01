from functools import partial
from itertools import product

import numpy as np
import pandas as pd

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
    return ":".join("_".join(t) for t in s.items())


def unpack_multicat(name: str) -> pd.Series:
    return pd.Series(
        {k: v for cat in name.split(":") for k, v in dict([cat.split("_")]).items()}
    )


def dummy_2d(s: pd.Series) -> pd.Series:
    return pd.Series({name_multicat(s): 1})


def check_dummies_2d(row: pd.Series, dummy_cols: pd.Index) -> pd.Series:
    """Stringifies input series (row) and gives a bool of whether it's a dummy var."""
    return pd.Series(
        dummy_cols.str.fullmatch(name_multicat(row[cat_cols].astype(str))),
        index=dummy_cols,
    )


null_combos = data[null_indicator][cat_cols].drop_duplicates()
dummy_vars = null_combos.astype(str).apply(dummy_2d, axis=1).fillna(0).astype(int)
dummy_cols = dummy_vars.columns
# List all possible combinations of the categorical columns
# combinations = list(product(*[data[null_indicator][cat].unique() for cat in cat_cols]))
check_2d_dummy_cols = partial(check_dummies_2d, dummy_cols=dummy_cols)
full_dummies = data[cat_cols].apply(check_2d_dummy_cols, axis=1).astype(int)
data_with_dummies = pd.concat([data, full_dummies], axis=1)

# Calculate the correlation between the binary column and the new columns
correlation_matrix = data_with_dummies[[target, *dummy_cols]].corr(numeric_only=True)

# Display the correlations between the binary column and the new columns
correlations_with_target = correlation_matrix[target].loc[dummy_cols]
print(correlations_with_target)

assert correlations_with_target.max() == 1.0
