# =====================================================
# GLOBAL SETUP
# =====================================================
import warnings
warnings.filterwarnings("ignore")

import sqlite3
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.stats import ttest_rel


# =====================================================
# PART 1: CSV-BASED TRAINING PIPELINE
# =====================================================
print("\n========== STARTUP FUNDING PROJECT (TERMINAL MODE) ==========")

print("\n===== STEP 1: DATA PREPARATION =====")
df = pd.read_csv("data/processed/startup_funding_cleaned.csv")

print("Available columns:")
print(df.columns.tolist())

# Encode categorical features
categorical_cols = df.select_dtypes(include="object").columns
label_encoders = {}

print("\nEncoding categorical columns...")
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print("Encoded columns:", list(categorical_cols))

# Targets
y_reg = df["amount_in_inr"]
df["funding_level"] = pd.qcut(df["amount_in_inr"], q=4, labels=[0, 1, 2, 3])
y_cls = df["funding_level"]

print("\nFunding level distribution:")
print(y_cls.value_counts())

# Features
X = df.drop(columns=["amount_in_inr", "funding_level"])

# Train–test split
X_train, X_test, y_train_cls, y_test_cls = train_test_split(
    X, y_cls,
    test_size=0.2,
    random_state=42,
    stratify=y_cls
)

_, _, y_train_reg, y_test_reg = train_test_split(
    X, y_reg,
    test_size=0.2,
    random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)

# =====================================================
# STEP 2: BASELINE MODEL TRAINING
# =====================================================
print("\n===== BASELINE MODEL TRAINING =====\n")

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=3000))
])
lr_pipeline.fit(X_train, y_train_cls)
lr_pred = lr_pipeline.predict(X_test)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train_cls)
dt_pred = dt.predict(X_test)

rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, y_train_cls)
rf_pred = rf.predict(X_test)

print("Logistic Regression")
print("Accuracy :", round(accuracy_score(y_test_cls, lr_pred), 4))
print("F1 Score :", round(f1_score(y_test_cls, lr_pred, average="weighted"), 4), "\n")

print("Decision Tree")
print("Accuracy :", round(accuracy_score(y_test_cls, dt_pred), 4))
print("F1 Score :", round(f1_score(y_test_cls, dt_pred, average="weighted"), 4), "\n")

print("Random Forest")
print("Accuracy :", round(accuracy_score(y_test_cls, rf_pred), 4))
print("F1 Score :", round(f1_score(y_test_cls, rf_pred, average="weighted"), 4), "\n")

print("Observation:")
print("Random Forest performs best among baseline models")
print("\n===== BASELINE MODELING COMPLETED =====")

# =====================================================
# STEP 3: MODEL EVALUATION
# =====================================================
print("\n===== MODEL EVALUATION =====\n")

print("Classification Evaluation:")
print("Accuracy :", round(accuracy_score(y_test_cls, rf_pred), 4))
print("F1 Score :", round(f1_score(y_test_cls, rf_pred, average="weighted"), 4), "\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test_cls, rf_pred))

cv_scores_rf = cross_val_score(rf, X, y_cls, cv=5, scoring="accuracy")
print("\nCross Validation Accuracy Scores:")
print(np.round(cv_scores_rf, 4))
print("Mean CV Accuracy :", round(cv_scores_rf.mean(), 4))

# Regression
rf_reg = RandomForestRegressor(n_estimators=300, random_state=42)
rf_reg.fit(X_train, y_train_reg)
reg_pred = rf_reg.predict(X_test)

mae = np.mean(np.abs(y_test_reg - reg_pred))
r2 = rf_reg.score(X_test, y_test_reg)

print("\nRegression Evaluation:")
print("MAE :", f"{mae:.2e}")
print("R2  :", round(r2, 4))

print("\n===== MODEL EVALUATION COMPLETED =====")

# =====================================================
# STEP 4: OPTIMIZATION INSIGHTS
# =====================================================
print("\n===== OPTIMIZATION INSIGHTS =====")

baseline_scores = cross_val_score(
    lr_pipeline, X, y_cls, cv=5, scoring="accuracy"
)

print("\nBaseline (Logistic Regression) Mean Accuracy :", round(baseline_scores.mean(), 4))
print("Optimized (Random Forest) Mean Accuracy      :", round(cv_scores_rf.mean(), 4))

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 5 Important Features:")
print(feature_importance.head(5))

# =====================================================
# STEP 5: STATISTICAL VALIDATION (PAIRED T-TEST)
# =====================================================
print("\n===== STATISTICAL VALIDATION =====")

t_stat, p_value = ttest_rel(cv_scores_rf, baseline_scores)

print("t-statistic :", round(t_stat, 4))
print("p-value     :", round(p_value, 6))

if p_value < 0.05:
    print("\nDecision: Reject Null Hypothesis (H₀)")
    print("Optimized model performs SIGNIFICANTLY better")
else:
    print("\nDecision: Fail to Reject Null Hypothesis (H₀)")
    print("No statistically significant improvement")

print("\n===== STATISTICAL VALIDATION COMPLETED =====")

# =====================================================
# PART 2: DATABASE-BASED TERMINAL EVALUATION
# =====================================================
print("\n===== DATABASE-BASED MODEL EVALUATION =====")

DB_PATH = "data/startup_funding.db"

conn = sqlite3.connect(DB_PATH)
X_db = pd.read_sql("SELECT * FROM X_features", conn)
y_level_db = pd.read_sql("SELECT * FROM y_funding_level", conn).values.ravel()
y_amount_db = pd.read_sql("SELECT * FROM y_funding_amount", conn).values.ravel()
conn.close()

# 🔑 CRITICAL FIX: ALIGN FEATURE ORDER
X_db = X_db[X.columns]

X_tr, X_te, y_lvl_tr, y_lvl_te, y_amt_tr, y_amt_te = train_test_split(
    X_db, y_level_db, y_amount_db,
    test_size=0.2,
    random_state=42,
    stratify=y_level_db
)

lvl_pred_db = rf.predict(X_te)
amt_pred_db = rf_reg.predict(X_te)

print("\nFunding Level Evaluation (DB Data):")
print("Accuracy :", round(accuracy_score(y_lvl_te, lvl_pred_db), 4))
print("F1 Score :", round(f1_score(y_lvl_te, lvl_pred_db, average="weighted"), 4))
print("Confusion Matrix:")
print(confusion_matrix(y_lvl_te, lvl_pred_db))

mae_db = np.mean(np.abs(y_amt_te - amt_pred_db))
print("\nFunding Amount MAE (DB Data):", f"{mae_db:.2e}")

print("\n========== END OF MAIN EXECUTION ==========")
# =====================================================
# STEP 6: INTERACTIVE STARTUP PREDICTION (TERMINAL)
# =====================================================
print("\n===== STARTUP FUNDING PREDICTION =====\n")

def get_valid_input(feature_name, encoder):
    print(f"\nAvailable {feature_name} options:")
    for val in encoder.classes_:
        print(f"- {val}")
    
    while True:
        value = input(f"\nEnter {feature_name}: ").strip()
        if value in encoder.classes_:
            return value
        else:
            print(f"Invalid {feature_name}. Please choose from above options.")

# --- Get categorical inputs safely ---
industry = get_valid_input("industry", label_encoders["industry"])
city = get_valid_input("city", label_encoders["city"])
investment_type = get_valid_input("investment_type", label_encoders["investment_type"])
city_tier = get_valid_input("city_tier", label_encoders["city_tier"])
market_size = get_valid_input("market_size_category", label_encoders["market_size_category"])

# --- Get numerical inputs ---
founded_year = int(input("\nEnter founded_year (e.g. 2020): "))
no_of_founders = int(input("Enter no_of_founders (e.g. 2): "))

# --- Encode inputs ---
startup_encoded = {
    "industry": label_encoders["industry"].transform([industry])[0],
    "city": label_encoders["city"].transform([city])[0],
    "investment_type": label_encoders["investment_type"].transform([investment_type])[0],
    "city_tier": label_encoders["city_tier"].transform([city_tier])[0],
    "market_size_category": label_encoders["market_size_category"].transform([market_size])[0],
    "founded_year": founded_year,
    "no_of_founders": no_of_founders
}

startup_df = pd.DataFrame([startup_encoded])

# 🔑 IMPORTANT: ensure feature order matches training
startup_df = startup_df[X.columns]

# --- Predict ---
funding_level_pred = rf.predict(startup_df)[0]
funding_amount_pred = rf_reg.predict(startup_df)[0]

funding_level_map = {
    0: "Low Funding",
    1: "Medium Funding",
    2: "High Funding",
    3: "Very High Funding"
}

print("\n===== PREDICTION RESULT =====")
print("Funding Level    :", funding_level_map[funding_level_pred])
print("Estimated Amount :", f"₹ {funding_amount_pred:,.0f}")
print("\n===== PREDICTION COMPLETED =====")


joblib.dump(rf, "models/funding_level_model.pkl")
joblib.dump(rf_reg, "models/funding_amount_model.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")
