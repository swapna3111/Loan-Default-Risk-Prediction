import pandas as pd

# Load the loan dataset
df = pd.read_csv("data/loan_train.csv")

print("DATASET LOADED SUCCESSFULLY!")
print("--------------------------------")

# Show first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Show number of rows and columns
print("\nDataset shape:")
print(df.shape)

# Show column names
print("\nColumn names:")
print(df.columns.tolist())

# Show information about columns
print("\nDataset information:")
print(df.info())

# Show loan status values
print("\nLoan Status:")
print(df["loan_status"].value_counts())