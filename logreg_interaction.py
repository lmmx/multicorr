import pandas as pd
import statsmodels.api as sm

# Load the data
df = pd.read_csv('simple_dataset.csv')

# Create a binary variable indicating the presence or absence of a null value in v
df['null_v'] = df['v'].isnull().astype(int)

# Create interaction terms between c1 and c2
df['c1:c2'] = df['c1'] + ':' + df['c2']

# Create dummy variables for the categorical variables c1, c2, c3, and their interactions
dummies_c1 = pd.get_dummies(df['c1'], prefix='c1', drop_first=True)
dummies_c2 = pd.get_dummies(df['c2'], prefix='c2', drop_first=True)
dummies_c3 = pd.get_dummies(df['c3'], prefix='c3', drop_first=True)
dummies_c1c2 = pd.get_dummies(df['c1:c2'], prefix='c1c2', drop_first=True)

# Concatenate the dummy variables and the null_v variable into a single DataFrame
X = pd.concat([dummies_c1, dummies_c2, dummies_c3, dummies_c1c2], axis=1)
y = df['null_v']

# Fit a logistic regression model to the data
logit_model = sm.Logit(y, X)
result = logit_model.fit()

# Print the model summary
print(result.summary())
