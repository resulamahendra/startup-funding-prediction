import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

# ----------------------------------
# Paths (STRICTLY MATCH YOUR STRUCTURE)
# ----------------------------------
X_PATH = "data/processed/X_features.csv"
Y_FUNDED_PATH = "data/processed/y_funded.csv"
Y_LEVEL_PATH = "data/processed/y_funding_level.csv"
Y_AMOUNT_PATH = "data/processed/y_funding_amount.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------------------------
# Load features
# ----------------------------------
X = pd.read_csv(X_PATH)

# ==================================================
# 1️⃣ FUNDED MODEL (BINARY CLASSIFICATION)
# ==================================================
y_funded = pd.read_csv(Y_FUNDED_PATH).values.ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y_funded, test_size=0.2, random_state=42, stratify=y_funded
)

funded_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

funded_model.fit(X_train, y_train)

y_pred = funded_model.predict(X_test)
print("Funded Model Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(funded_model, "models/funded_model.pkl")
print("✅ funded_model.pkl saved successfully")

# ==================================================
# 2️⃣ FUNDING LEVEL MODEL
# ==================================================
y_level = pd.read_csv(Y_LEVEL_PATH).values.ravel()

level_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)
level_model.fit(X, y_level)

joblib.dump(level_model, "models/funding_level_model.pkl")
print("✅ funding_level_model.pkl saved")

# ==================================================
# 3️⃣ FUNDING AMOUNT REGRESSOR
# ==================================================
y_amount = pd.read_csv(Y_AMOUNT_PATH).values.ravel()

amount_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)
amount_model.fit(X, y_amount)

joblib.dump(amount_model, "models/funding_amount_model.pkl")
print("✅ funding_amount_model.pkl saved")

print("🎯 ALL MODELS TRAINED & SAVED SUCCESSFULLY")
