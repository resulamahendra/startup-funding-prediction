import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import joblib
from datetime import datetime

# ----------------------------------
# Paths
# ----------------------------------
DATA_PATH = "data/processed/startup_funding_cleaned.csv"
OUTPUT_DIR = "data/processed"
ENCODER_PATH = "models/label_encoders.pkl"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# ----------------------------------
# Load data
# ----------------------------------
df = pd.read_csv(DATA_PATH)

CURRENT_YEAR = datetime.now().year

# ----------------------------------
# TARGET 1: Funded (Binary)
# ----------------------------------
df["funded"] = (df["amount_in_inr"] > 0).astype(int)

# ----------------------------------
# TARGET 2: Funding Level (Multi-class)
# ----------------------------------
def funding_level(amount):
    if amount == 0:
        return 0
    elif amount <= 5_000_000:
        return 1
    elif amount <= 50_000_000:
        return 2
    else:
        return 3

df["funding_level"] = df["amount_in_inr"].apply(funding_level)

# ----------------------------------
# TARGET 3: Funding Amount (Regression)
# ----------------------------------
df["funding_amount"] = df["amount_in_inr"]

# ----------------------------------
# TARGET 4: Startup Age
# ----------------------------------
df["startup_age"] = CURRENT_YEAR - df["founded_year"]

# ----------------------------------
# TARGET 5: High Potential Startup
# ----------------------------------
df["high_potential_startup"] = (
    (df["funded"] == 1) &
    (df["market_size_category"].str.lower() == "large") &
    (df["city_tier"] == "Tier 1")
).astype(int)

# ----------------------------------
# TARGET 6: Founder Strength
# ----------------------------------
def founder_strength(n):
    if n == 1:
        return 0
    elif n <= 3:
        return 1
    else:
        return 2

df["founder_strength"] = df["no_of_founders"].apply(founder_strength)

# ----------------------------------
# Encode categorical features
# ----------------------------------
categorical_cols = [
    "industry",
    "city",
    "investment_type",
    "city_tier",
    "market_size_category"
]

numerical_cols = [
    "founded_year",
    "no_of_founders"
]

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

joblib.dump(encoders, ENCODER_PATH)

# ----------------------------------
# Final feature set
# ----------------------------------
X = df[categorical_cols + numerical_cols]

# ----------------------------------
# Save multiple targets
# ----------------------------------
targets = [
    "funded",
    "funding_level",
    "funding_amount",
    "startup_age",
    "high_potential_startup",
    "founder_strength"
]

X.to_csv(f"{OUTPUT_DIR}/X_features.csv", index=False)

for t in targets:
    df[t].to_csv(f"{OUTPUT_DIR}/y_{t}.csv", index=False)

print("✅ Feature engineering completed with MULTIPLE targets!")
print("Input features:", X.columns.tolist())
print("Targets created:", targets)
