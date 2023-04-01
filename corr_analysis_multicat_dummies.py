import itertools

import numpy as np
import pandas as pd

# Load your dataset
# Replace the path with the location of your dataset file
data = pd.read_csv("your_dataset.csv")


def analyze_categorical_combinations(data, categorical_columns, target_column):
    # Create a binary column to indicate the presence of null values in the target column
    data["null_target_column"] = data[target_column].isnull().astype(int)
    # Create dummy variables for all categorical columns and concatenate them
    dummy_variables = pd.get_dummies(data[categorical_columns])
    # Merge the dummy variables with the original dataframe and drop the original categorical columns
    data = pd.concat([data, dummy_variables], axis=1).drop(columns=categorical_columns)
    # Calculate the correlation between the binary column and the new columns
    correlation_matrix = data.corr(numeric_only=True)
    # Display the correlations between the binary column and the new columns
    correlations_with_null_target = correlation_matrix["null_target_column"].loc[
        dummy_variables.columns
    ]
    print(correlations_with_null_target)
    breakpoint()
    return correlations_with_null_target


target_col = "target_column"
cat_cols = ["categorical_column_1", "categorical_column_2", "categorical_column_3"]
corr_result = analyze_categorical_combinations(data, cat_cols, target_col)
assert corr_result.max() == 1.0
