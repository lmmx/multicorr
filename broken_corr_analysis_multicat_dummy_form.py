from itertools import product

import numpy as np
import pandas as pd

# Load your dataset
data = pd.read_csv("simple_dataset.csv")


cat_cols = ["c1", "c2", "c3"]
target_column = "v"
null_target = "null_target_column"

# Create a binary column to indicate the presence of null values in the target column
null_indicator = data[target_column].isnull()
data[null_target] = null_indicator.astype(int)

# List all possible combinations of the categorical columns
combinations = []
for cat in cat_cols:
    combos = product(list(data[null_indicator][cat].unique()))
    combinations.append(combos)

# Create a new column for each combo and fill it with zeros
for combo in combinations:
    column_name = "_".join([str(value) for value in combo])
    data[column_name] = 0

# Fill the new columns with ones where the corresponding combo is present
for index, row in data.iterrows():
    for combo in combinations:
        column_name = "_".join([str(value) for value in combo])
        if all(row[column] == value for column, value in zip(cat_cols, combo)):
            data.at[index, column_name] = 1

# Calculate the correlation between the binary column and the new columns
correlation_matrix = data.corr(numeric_only=True)

# Display the correlations between the binary column and the new columns
correlations_with_null_target = correlation_matrix[null_target].loc[
    [
        column
        for column in data.columns
        if column not in [*cat_cols, null_target, target_column]
    ]
]
print(correlations_with_null_target)

assert correlations_with_null_target.max() == 1.0
