import numpy as np
import pandas as pd

# Load your dataset
# Replace the path with the location of your dataset file
data = pd.read_csv("your_dataset.csv")
target = "target_column"
cat_cols = ["categorical_column_1", "categorical_column_2"]

# Create a binary column to indicate the presence of null values in the column you're interested in
data["null_target_col"] = data[target].isnull().astype(int)

# Create dummy variables for the categorical columns you want to analyze
dummy_variables = pd.get_dummies(data[cat_cols])

# Combine the original dataset with the dummy columns
data_with_dummies = pd.concat([data, dummy_variables], axis=1)

# Calculate the correlation between the binary column and the dummy variables
correlation_matrix = data_with_dummies.corr(numeric_only=True)

# Display the correlations between the binary column and the dummy variables
correlations_with_null_target = correlation_matrix["null_target_col"].loc[
    dummy_variables.columns
]
print(correlations_with_null_target)
