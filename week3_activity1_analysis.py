import pandas as pd
import numpy as np


# Make large numbers easier to read in the terminal.
pd.set_option(
    "display.float_format",
    lambda value: f"{value:,.2f}"
)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


# Load the dataset
file_path = "Sample_dataset.csv"

df = pd.read_csv(file_path)


# Display the original dataset
print("ORIGINAL DATASET")
print(df)


# Display basic information
print("\nDATASET INFORMATION")
df.info()


# Check missing values
print("\nMISSING VALUES")
print(df.isnull().sum())



# ============================================================
# DATA CLEANING
# ============================================================

print("\nSTARTING DATA CLEANING")


# Remove unnecessary spaces from column names
df.columns = df.columns.str.strip()


# Replace empty strings with missing values
df.replace(r"^\s*$", np.nan, regex=True, inplace=True)


# Convert ID into numeric format
df["ID"] = pd.to_numeric(df["ID"], errors="coerce")


# Replace Age written as text
df["Age"] = df["Age"].replace({
    "thirty-eight": 38
})


# Convert Age to numeric
df["Age"] = pd.to_numeric(
    df["Age"],
    errors="coerce"
)

# Remove commas from Net worth
df["Net worth"] = (
    df["Net worth"]
    .astype(str)
    .str.replace(",", "", regex=False)
)


# Convert Net worth to numeric
df["Net worth"] = pd.to_numeric(
    df["Net worth"],
    errors="coerce"
)

# Replace Salary written as text
df["Salary"] = df["Salary"].replace({
    "sixty five thousand": 65000
})


# Convert Salary to numeric
df["Salary"] = pd.to_numeric(
    df["Salary"],
    errors="coerce"
)

# Standardise country codes
df["Country"] = df["Country"].replace({
    "AU": "AUS",
    "Australia": "AUS",
    "New Zealand": "NZ"
})

# Convert Join Date into datetime format
df["Join Date"] = pd.to_datetime(
    df["Join Date"],
    errors="coerce",
    dayfirst=True,
    format="mixed"
)

# Combine records that have the same ID
df = (
    df.groupby(
        "ID",
        dropna=False,
        as_index=False
    )
    .first()
)

# Assign the missing ID based on the sequence
df.loc[
    df["ID"].isna(),
    "ID"
] = 6


# Convert ID to integer
df["ID"] = df["ID"].astype(int)

# Fill missing numeric values using the median
df["Age"] = df["Age"].fillna(
    df["Age"].median()
)

df["Net worth"] = df["Net worth"].fillna(
    df["Net worth"].median()
)

df["Salary"] = df["Salary"].fillna(
    df["Salary"].median()
)

# Fill missing Country using the most common country
country_mode = df["Country"].mode()[0]

df["Country"] = df["Country"].fillna(
    country_mode
)

# Label missing names
df["Name"] = df["Name"].fillna(
    "Unknown"
)

# Sort the dataset by ID
df = df.sort_values(
    by="ID"
).reset_index(drop=True)


print("\nCLEANED DATASET")
print(df)


print("\nMISSING VALUES AFTER CLEANING")
print(df.isnull().sum())

# Save the cleaned dataset
df.to_csv(
    "Sample_dataset_cleaned.csv",
    index=False
)

print(
    "\nCleaned dataset saved as "
    "Sample_dataset_cleaned.csv"
)

# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

numeric_columns = [
    "Age",
    "Net worth",
    "Salary"
]

numeric_data = df[numeric_columns]

# Calculate mean
mean_values = numeric_data.mean()

print("\nMEAN")
print(mean_values)

# Calculate median
median_values = numeric_data.median()

print("\nMEDIAN")
print(median_values)

# Calculate and display meaningful modes
print("\nMODE")

for column in numeric_columns:
    frequency = numeric_data[column].value_counts()
    highest_frequency = frequency.iloc[0]

    if highest_frequency == 1:
        print(f"{column}: No unique mode")
    else:
        modes = frequency[
            frequency == highest_frequency
        ].index.tolist()

        print(
            f"{column}: {modes} "
            f"(appears {highest_frequency} times)"
        )

# Calculate minimum
minimum_values = numeric_data.min()


# Calculate maximum
maximum_values = numeric_data.max()


# Calculate range
range_values = (
    maximum_values -
    minimum_values
)


print("\nMINIMUM")
print(minimum_values)

print("\nMAXIMUM")
print(maximum_values)

print("\nRANGE")
print(range_values)

# Calculate sample variance
variance_values = numeric_data.var(
    ddof=1
)

print("\nSAMPLE VARIANCE")
print(variance_values)

# Calculate sample standard deviation
standard_deviation_values = (
    numeric_data.std(ddof=1)
)

print("\nSAMPLE STANDARD DEVIATION")
print(standard_deviation_values)

# Calculate sample covariance
covariance_matrix = numeric_data.cov(
    ddof=1
)

print("\nSAMPLE COVARIANCE")
print(covariance_matrix)

# Calculate Pearson correlation
correlation_matrix = numeric_data.corr()

print("\nPEARSON CORRELATION")
print(correlation_matrix)

# Create a statistical summary table
statistical_summary = numeric_data.describe().T


# Add extra metrics
statistical_summary["variance"] = (
    numeric_data.var(ddof=1)
)

statistical_summary["range"] = (
    numeric_data.max() -
    numeric_data.min()
)


print("\nSTATISTICAL SUMMARY")
print(statistical_summary.round(2))
