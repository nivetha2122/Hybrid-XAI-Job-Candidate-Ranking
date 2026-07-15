import pandas as pd

# Load the resume dataset
df = pd.read_csv("data/Resume.csv")

# Display the first 5 rows
print("First 5 Rows:")
print(df.head())

# Display column names
print("\nColumn Names:")
print(df.columns)

# Display dataset information
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())