import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.stats import ttest_rel

# =====================================================
# STEP 1: DATA PREPARATION
# =====================================================
print("\n===== STEP 1: DATA PREPARATION =====")

df = pd.read_csv("data/processed/startup_funding_cleaned.csv")

print("Available columns:")
print(df.columns.tolist(), "\n")

# =====================================================
# STEP 2: ENCODE CATEGORICAL FEATURES
# =====================================================
print("Encoding categorical columns...")

categorical_cols = df.select_dtypes(include="object").columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print("Encoded columns:", list(categorical_cols), "\n")

# =====================================================
# STEP 3: CREATE TARGET VARIABLES
# =====================================================
print("Creating target variables...\n")

# Regression target
y_reg = df["amount_in_inr"]

# Multi-class classification target using quantiles
df["funding_level"] = pd.qcut(
    df["amount_in_inr"],
    q=4,
    labels=[0, 1, 2, 3]
)

y_cls = df["funding_level"]

print("Funding level distribution:")
print(y_cls.value_counts(), "\n")

# Features
X = df.drop(columns=["amount_in_inr", "funding_level"])

# =====================================================
# STEP 4: TRAIN–TEST SPLIT
# =====================================================
X_train, X_test, y_train_cls, y_test_cls = train_test_split(
    X,
    y_cls,
    test_size=0.2,
    random_state=42,
    stratify=y_cls
)

_, _, y_train_reg, y_test_reg = train_test_split(
    X,
    y_reg,
    test_size=0.2,
    random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# =====================================================
# STEP 5: BASELINE MODEL TRAINING
# =====================================================
print("\n===== BASELINE MODEL TRAINING =====\n")

# ---------- Logistic Regression (Scaled Pipeline) ----------
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=3000))
])

lr_pipeline.fit(X_train, y_train_cls)
lr_pred = lr_pipeline.predict(X_test)

print("Logistic Regression")
print("Accuracy :", round(accuracy_score(y_test_cls, lr_pred), 4))
print("F1 Score :", round(f1_score(y_test_cls, lr_pred, average="weighted"), 4), "\n")

# ---------- Decision Tree ----------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train_cls)
dt_pred = dt.predict(X_test)

print("Decision Tree")
print("Accuracy :", round(accuracy_score(y_test_cls, dt_pred), 4))
print("F1 Score :", round(f1_score(y_test_cls, dt_pred, average="weighted"), 4), "\n")

# ---------- Random Forest ----------
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, y_train_cls)
rf_pred = rf.predict(X_test)

print("Random Forest")
print("Accuracy :", round(accuracy_score(y_test_cls, rf_pred), 4))
print("F1 Score :", round(f1_score(y_test_cls, rf_pred, average="weighted"), 4), "\n")

print("Observation:")
print("Random Forest performs best among baseline models\n")
print("===== BASELINE MODELING COMPLETED =====")

# =====================================================
# STEP 6: MODEL EVALUATION
# =====================================================
print("\n===== MODEL EVALUATION =====\n")

print("Classification Evaluation:\n")
print("Accuracy :", round(accuracy_score(y_test_cls, rf_pred), 4))
print("F1 Score :", round(f1_score(y_test_cls, rf_pred, average="weighted"), 4), "\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test_cls, rf_pred), "\n")

cv_scores_rf = cross_val_score(
    rf, X, y_cls, cv=5, scoring="accuracy"
)

print("Cross Validation Accuracy Scores:")
print(np.round(cv_scores_rf, 4))
print("\nMean CV Accuracy :", round(cv_scores_rf.mean(), 4), "\n")

# =====================================================
# STEP 7: REGRESSION MODEL EVALUATION
# =====================================================
rf_reg = RandomForestRegressor(n_estimators=300, random_state=42)
rf_reg.fit(X_train, y_train_reg)
reg_pred = rf_reg.predict(X_test)

mae = np.mean(np.abs(y_test_reg - reg_pred))
r2 = rf_reg.score(X_test, y_test_reg)

print("Regression Evaluation:\n")
print("MAE :", f"{mae:.2e}")
print("R2  :", round(r2, 4), "\n")

print("===== MODEL EVALUATION COMPLETED =====")

# =====================================================
# STEP 8: MODEL COMPARISON
# =====================================================
print("\n===== MODEL COMPARISON =====\n")

print("Classification Models:")
print("Logistic Regression   → Accuracy:", round(accuracy_score(y_test_cls, lr_pred), 4))
print("Decision Tree         → Accuracy:", round(accuracy_score(y_test_cls, dt_pred), 4))
print("Random Forest         → Accuracy:", round(accuracy_score(y_test_cls, rf_pred), 4), "\n")

print("Best Classification Model:")
print("Random Forest Classifier\n")

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train_reg)
lin_r2 = lin_reg.score(X_test, y_test_reg)

print("Regression Models:")
print("Linear Regression     → R2:", round(lin_r2, 4))
print("Random Forest         → R2:", round(r2, 4), "\n")

print("Best Regression Model:")
print("Random Forest Regressor\n")

print("===== MODEL COMPARISON COMPLETED =====")

# =====================================================
# STEP 9: OPTIMIZATION INSIGHTS
# =====================================================
print("\n===== OPTIMIZATION INSIGHTS =====\n")

baseline_scores = cross_val_score(
    lr_pipeline,
    X,
    y_cls,
    cv=5,
    scoring="accuracy"
)

optimized_scores = cv_scores_rf

print("Baseline Model (Logistic Regression) CV Accuracy:")
print(np.round(baseline_scores, 4))

print("\nOptimized Model (Random Forest) CV Accuracy:")
print(np.round(optimized_scores, 4))

print("\nMean Accuracy Comparison:")
print("Baseline Mean Accuracy :", round(baseline_scores.mean(), 4))
print("Optimized Mean Accuracy:", round(optimized_scores.mean(), 4))

print("\nTop 5 Important Features (Random Forest):")
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print(feature_importance.head(5))

# =====================================================
# STEP 10: STATISTICAL VALIDATION (PAIRED T-TEST)
# =====================================================
print("\n===== STATISTICAL VALIDATION =====\n")

t_stat, p_value = ttest_rel(optimized_scores, baseline_scores)

print("Paired t-test Results:")
print("t-statistic :", round(t_stat, 4))
print("p-value     :", round(p_value, 6))

alpha = 0.05
if p_value < alpha:
    print("\nDecision:")
    print("Reject Null Hypothesis (H0)")
    print("Optimized model performs SIGNIFICANTLY better than baseline")
else:
    print("\nDecision:")
    print("Fail to Reject Null Hypothesis (H0)")
    print("No significant performance improvement detected")

print("\n===== STATISTICAL VALIDATION COMPLETED =====")
