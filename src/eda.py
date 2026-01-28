import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="Startup Funding EDA Dashboard",
    layout="wide"
)

st.title("📊 Startup Funding EDA & Data Quality Dashboard")

# ----------------------------------
# Paths
# ----------------------------------
RAW_DATA = "data/raw/startup_funding_raw.csv"
CLEAN_DATA = "data/processed/startup_funding_cleaned.csv"
OUTPUT_DIR = "outputs"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(PLOT_DIR, exist_ok=True)

# ----------------------------------
# Load Data
# ----------------------------------
raw_df = pd.read_csv(RAW_DATA)
clean_df = pd.read_csv(CLEAN_DATA)

# ----------------------------------
# DATA QUALITY COMPARISON
# ----------------------------------
st.header("🧹 Data Quality Comparison")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Before Cleaning")
    st.metric("Rows", raw_df.shape[0])
    st.metric("Missing Values", raw_df.isnull().sum().sum())
    st.metric("Duplicate Rows", raw_df.duplicated().sum())

with col2:
    st.subheader("After Cleaning")
    st.metric("Rows", clean_df.shape[0])
    st.metric("Missing Values", clean_df.isnull().sum().sum())
    st.metric("Duplicate Rows", clean_df.duplicated().sum())

# ----------------------------------
# Missing Values Table
# ----------------------------------
st.subheader("🔎 Missing Values (Before vs After)")

missing_comparison = pd.DataFrame({
    "Before Cleaning": raw_df.isnull().sum(),
    "After Cleaning": clean_df.isnull().sum()
})

st.dataframe(missing_comparison)

# ----------------------------------
# Save EDA Summary
# ----------------------------------
with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
    f.write("EDA & DATA QUALITY REPORT\n")
    f.write("=========================\n\n")
    f.write("RAW DATASET\n")
    f.write(str(raw_df.describe(include="all")))
    f.write("\n\nCLEANED DATASET\n")
    f.write(str(clean_df.describe(include="all")))

# ----------------------------------
# EDA SECTION
# ----------------------------------
st.header("📈 Exploratory Data Analysis")

sns.set_style("whitegrid")

# ---- Funding Distribution
st.subheader("Funding Amount Distribution")

plt.figure()
sns.histplot(clean_df["amount_in_inr"], bins=30, kde=True)
plt.title("Funding Amount Distribution (INR)")
plt.savefig(os.path.join(PLOT_DIR, "funding_distribution.png"))
st.pyplot(plt)
plt.close()

# ---- Industry Count
st.subheader("Startups by Industry")

plt.figure()
sns.countplot(
    y="industry",
    data=clean_df,
    order=clean_df["industry"].value_counts().index
)
plt.title("Startups by Industry")
plt.savefig(os.path.join(PLOT_DIR, "industry_distribution.png"))
st.pyplot(plt)
plt.close()

# ---- City Tier vs Funding
st.subheader("Funding vs City Tier")

plt.figure()
sns.boxplot(
    x="city_tier",
    y="amount_in_inr",
    data=clean_df
)
plt.title("Funding vs City Tier")
plt.savefig(os.path.join(PLOT_DIR, "city_tier_vs_funding.png"))
st.pyplot(plt)
plt.close()

# ---- Correlation Heatmap
st.subheader("Correlation Heatmap")

plt.figure(figsize=(8, 6))
numeric_df = clean_df.select_dtypes(include=["int64", "float64"])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig(os.path.join(PLOT_DIR, "correlation_heatmap.png"))
st.pyplot(plt)
plt.close()

# ----------------------------------
# Data Samples
# ----------------------------------
st.header("📄 Sample Records")

tab1, tab2 = st.tabs(["Raw Data", "Cleaned Data"])

with tab1:
    st.dataframe(raw_df.head(10))

with tab2:
    st.dataframe(clean_df.head(10))

st.success("✅ EDA completed, website displayed, and outputs saved successfully!")
