import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/startup_funding_raw.csv")

# -------------------------------------------------
# 1. Standardize column names
# -------------------------------------------------
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("Columns after cleaning:", df.columns.tolist())

# -------------------------------------------------
# 2. Remove duplicates
# -------------------------------------------------
df = df.drop_duplicates()

# -------------------------------------------------
# 3. Handle missing values (CORRECT WAY)
# -------------------------------------------------

# Categorical columns → mode
categorical_cols = [
    "industry",
    "city",
    "investment_type",
    "city_tier",
    "market_size_category"
]

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Numerical columns → median
df["founded_year"] = df["founded_year"].fillna(df["founded_year"].median())
df["no_of_founders"] = df["no_of_founders"].fillna(df["no_of_founders"].median())
df["amount_in_inr"] = df["amount_in_inr"].fillna(df["amount_in_inr"].median())

# -------------------------------------------------
# 4. Save cleaned data
# -------------------------------------------------
df.to_csv("data/processed/startup_funding_cleaned.csv", index=False)

print("✅ Data cleaning completed successfully")
