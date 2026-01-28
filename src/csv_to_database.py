import sqlite3
import pandas as pd
from pathlib import Path

# =====================================================
# DATABASE CONFIGURATION
# =====================================================
DB_PATH = "data/startup_funding.db"

CSV_TABLE_MAPPING = {
    "raw_startup_funding": "data/raw/startup_funding_raw.csv",
    "startup_funding_cleaned": "data/processed/startup_funding_cleaned.csv",
    "X_features": "data/processed/X_features.csv",
    "y_funded": "data/processed/y_funded.csv",
    "y_funding_amount": "data/processed/y_funding_amount.csv",
    "y_funding_level": "data/processed/y_funding_level.csv",
    "y_founder_strength": "data/processed/y_founder_strength.csv",
    "y_high_potential_startup": "data/processed/y_high_potential_startup.csv",
    "y_startup_age": "data/processed/y_startup_age.csv"
}

# =====================================================
# CONNECT TO DATABASE
# =====================================================
print("\n===== CONNECTING TO DATABASE =====")
conn = sqlite3.connect(DB_PATH)
print(f"Database path: {DB_PATH}\n")

# =====================================================
# LOAD CSV FILES INTO DATABASE
# =====================================================
for table_name, csv_path in CSV_TABLE_MAPPING.items():
    if not Path(csv_path).exists():
        print(f"❌ Missing CSV: {csv_path}")
        continue

    print(f"Loading `{csv_path}` into table `{table_name}`")

    df = pd.read_csv(csv_path)

    df.to_sql(
        table_name,
        conn,
        if_exists="replace",
        index=False
    )

    print(f"✅ {table_name}: {df.shape[0]} rows, {df.shape[1]} columns\n")

# =====================================================
# VERIFY DATABASE TABLES
# =====================================================
print("===== DATABASE TABLES CREATED =====")

tables = pd.read_sql(
    "SELECT name FROM sqlite_master WHERE type='table'",
    conn
)
print(tables)

conn.close()

print("\n===== CSV TO DATABASE CONNECTION COMPLETED =====")
