import pandas as pd

# Load the original dataset
df = pd.read_csv("data/loan_train.csv")

print("Original dataset shape:", df.shape)

# Columns that should not be used for prediction
columns_to_remove = [
    "id",
    "member_id",
    "emp_title",
    "issue_d",
    "url",
    "desc",
    "title",
    "zip_code",
    "addr_state",
    "earliest_cr_line",
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "last_credit_pull_d"
]

# Remove unwanted columns
df_clean = df.drop(columns=columns_to_remove)

print("Cleaned dataset shape:", df_clean.shape)

# Show remaining columns
print("\nRemaining columns:")
print(df_clean.columns.tolist())

# Save cleaned dataset
df_clean.to_csv("data/loan_clean.csv", index=False)

print("\nCleaned dataset saved successfully!")

