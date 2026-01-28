import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    mean_absolute_error,
    r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ------------------------------------------------
# Paths
# ------------------------------------------------
X_PATH = "data/processed/X_features.csv"
Y_LEVEL_PATH = "data/processed/y_funding_level.csv"
Y_AMOUNT_PATH = "data/processed/y_funding_amount.csv"

MODEL_DIR = "models"
OUTPUT_PLOT_DIR = "outputs/plots"

os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# ------------------------------------------------
# Load data
# ------------------------------------------------
X = pd.read_csv(X_PATH)
y_level = pd.read_csv(Y_LEVEL_PATH).values.ravel()
y_amount = pd.read_csv(Y_AMOUNT_PATH).values.ravel()

# ------------------------------------------------
# Load trained models
# ------------------------------------------------
level_model = joblib.load(f"{MODEL_DIR}/funding_level_model.pkl")
amount_model = joblib.load(f"{MODEL_DIR}/funding_amount_model.pkl")

# ------------------------------------------------
# Train-test split
# ------------------------------------------------
X_train, X_test, y_train_lvl, y_test_lvl = train_test_split(
    X, y_level, test_size=0.2, random_state=42
)

_, _, y_train_amt, y_test_amt = train_test_split(
    X, y_amount, test_size=0.2, random_state=42
)

# =================================================
# 1️⃣ CLASSIFICATION EVALUATION (Funding Level)
# =================================================
print("\n========== FUNDING LEVEL CLASSIFICATION ==========")

y_pred_lvl = level_model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test_lvl, y_pred_lvl))

# Confusion Matrix (TERMINAL OUTPUT)
cm = confusion_matrix(y_test_lvl, y_pred_lvl)
print("\nConfusion Matrix (Terminal Output):")
print(cm)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test_lvl, y_pred_lvl))

# Save confusion matrix image
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Funding Level")
plt.tight_layout()
plt.savefig(f"{OUTPUT_PLOT_DIR}/confusion_matrix_funding_level.png")
plt.close()

# ------------------------------------------------
# Cross Validation (Classification)
# ------------------------------------------------
cv_scores_cls = cross_val_score(
    RandomForestClassifier(n_estimators=300, random_state=42),
    X,
    y_level,
    cv=5,
    scoring="accuracy"
)

print("\nCross-Validation Accuracy Scores:", cv_scores_cls)
print("Mean CV Accuracy:", cv_scores_cls.mean())

# =================================================
# 2️⃣ REGRESSION EVALUATION (Funding Amount)
# =================================================
print("\n========== FUNDING AMOUNT REGRESSION ==========")

y_pred_amt = amount_model.predict(X_test)

print("MAE:", mean_absolute_error(y_test_amt, y_pred_amt))
print("R² Score:", r2_score(y_test_amt, y_pred_amt))

# ------------------------------------------------
# Cross Validation (Regression)
# ------------------------------------------------
cv_scores_reg = cross_val_score(
    RandomForestRegressor(n_estimators=300, random_state=42),
    X,
    y_amount,
    cv=5,
    scoring="neg_mean_absolute_error"
)

print("Cross-Validation MAE:", -cv_scores_reg.mean())

print("\n✅ Evaluation completed")
