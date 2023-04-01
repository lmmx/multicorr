from itertools import chain, combinations, product

import numpy as np
import pandas as pd

data = pd.read_csv("your_dataset.csv")


def analyze_categorical_combinations(data, categorical_columns, target_column, combo):
    # Create a binary column to indicate the presence of null values in the target column
    data["null_target_column"] = data[target_column].isnull().astype(int)
    if combo == ["categorical_column_1", "categorical_column_2"]:
        breakpoint()
    # Create a new column for each combination and fill it with zeros
    dummy_variables = pd.get_dummies(data[combo])
    data_with_dummies = pd.concat([data, dummy_variables], axis=1).drop(
        columns=categorical_columns
    )
    # Calculate the correlation between the binary column and the new columns
    correlation_matrix = data_with_dummies.corr(numeric_only=True)
    correlations_with_null_target = correlation_matrix["null_target_column"].loc[
        dummy_variables.columns
    ]
    print(correlations_with_null_target)
    return correlations_with_null_target


def get_most_parsimonious_categorical_combo(data, categorical_columns, target_column):
    # Create a binary column to indicate the presence of null values in the target column
    data["null_target_column"] = data[target_column].isnull().astype(int)
    # List all possible combinations of the categorical columns
    column_combinations = list(
        chain.from_iterable(
            combinations(categorical_columns, r)
            for r in dict(enumerate(categorical_columns))
        )
    )
    best_correlation = -1
    best_combination = None
    best_columns = None
    for combo in column_combinations:
        print(combo)
        if not combo:
            continue
        df_copy = data.copy()
        correlations_with_null_target = analyze_categorical_combinations(
            data=df_copy,
            categorical_columns=categorical_columns,
            target_column=target_column,
            combo=list(combo),
        )
        dummy_columns = set(df_copy.columns) - set(data.columns)
        print(correlations_with_null_target)
        print()
        max_correlation = correlations_with_null_target.abs().max()
        new_hi_score = max_correlation > best_correlation
        score_match = max_correlation == best_correlation
        new_lo_combo = score_match and (len(combo) < len(best_combination))
        if new_hi_score or new_lo_combo:
            best_correlation = max_correlation
            best_combination = combo
            best_columns = dummy_columns
    # Print the best result
    print(f"Best categorical column combination: {best_combination}")
    print(f"Best correlation: {best_correlation}")
    return best_columns


corr_result = get_most_parsimonious_categorical_combo(
    data,
    ["categorical_column_1", "categorical_column_2", "categorical_column_3"],
    "target_column",
)
