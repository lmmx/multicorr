import itertools

import numpy as np
import pandas as pd

data = pd.read_csv("your_dataset.csv")


def analyze_categorical_combinations(
    data, categorical_columns, target_column, excluded_columns=[]
):
    # Create a binary column to indicate the presence of null values in the target column
    data["null_target_column"] = data[target_column].isnull().astype(int)

    # Create a list of all possible combinations of the categorical columns
    combinations = list(
        itertools.product(*[data[column].unique() for column in categorical_columns])
    )

    # Create a new column for each combination and fill it with zeros
    for combination in combinations:
        column_name = "_".join([str(value) for value in combination])
        data[column_name] = 0

    # Fill the new columns with ones where the corresponding combination is present
    for index, row in data.iterrows():
        for combination in combinations:
            column_name = "_".join([str(value) for value in combination])
            if all(
                row[column] == value
                for column, value in zip(categorical_columns, combination)
            ):
                data.at[index, column_name] = 1

    # Remove the excluded columns from the new columns
    new_columns = [
        column
        for column in data.columns
        if column
        not in [
            *categorical_columns,
            "null_target_column",
            target_column,
            *excluded_columns,
        ]
    ]

    # Calculate the correlation between the binary column and the new columns
    correlation_matrix = data.corr(numeric_only=True)

    # Display the correlations between the binary column and the new columns
    correlations_with_null_target = correlation_matrix["null_target_column"].loc[
        new_columns
    ]
    print(correlations_with_null_target)
    return correlations_with_null_target


corr_result = analyze_categorical_combinations(
    data,
    ["categorical_column_1", "categorical_column_2", "categorical_column_3"],
    "target_column",
)
