import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---------------------------------
# Load data
# ---------------------------------
X = pd.read_csv("data/processed/X_features.csv")
y_level = pd.read_csv("data/processed/y_funding_level.csv").values.ravel()
y_amount = pd.read_csv("data/processed/y_funding_amount.csv").values.ravel()

# ---------------------------------
# Train-test split
# ---------------------------------
X_train, X_test, y_train_lvl, y_test_lvl = train_test_split(
    X, y_level, test_size=0.2, random_state=42
)

_, _, y_train_amt, y_test_amt = train_test_split(
    X, y_amount, test_size=0.2, random_state=42
)

# =================================================
# 🔵 CLASSIFICATION COMPARISON
# =================================================
classification_models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            solver="saga",
            max_iter=1000,
            n_jobs=-1
        ))
    ]),
    "Random Forest Classifier": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
}

cls_results = []

for name, model in classification_models.items():
    model.fit(X_train, y_train_lvl)
    preds = model.predict(X_test)

    cls_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test_lvl, preds),
        "F1": f1_score(y_test_lvl, preds, average="weighted")
    })

cls_df = pd.DataFrame(cls_results)

print("\n📊 CLASSIFICATION MODEL COMPARISON")
print(cls_df)

# ---- Select best classification model
best_cls = cls_df.sort_values(
    by=["F1", "Accuracy"], ascending=False
).iloc[0]

print("\n🏆 BEST CLASSIFICATION MODEL")
print(f"Model: {best_cls['Model']}")
print(f"Reason: Higher F1-score ({best_cls['F1']:.4f}) "
      f"and Accuracy ({best_cls['Accuracy']:.4f})")

# =================================================
# 🟢 REGRESSION COMPARISON
# =================================================
reg_results = []

# Linear Regression
lin_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])
lin_reg.fit(X_train, y_train_amt)
lin_preds = lin_reg.predict(X_test)

reg_results.append({
    "Model": "Linear Regression",
    "MAE": mean_absolute_error(y_test_amt, lin_preds),
    "R2": r2_score(y_test_amt, lin_preds)
})

# Random Forest Regressor
rf_reg = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
rf_reg.fit(X_train, y_train_amt)
rf_preds = rf_reg.predict(X_test)

reg_results.append({
    "Model": "Random Forest Regressor",
    "MAE": mean_absolute_error(y_test_amt, rf_preds),
    "R2": r2_score(y_test_amt, rf_preds)
})

reg_df = pd.DataFrame(reg_results)

print("\n📈 REGRESSION MODEL COMPARISON")
print(reg_df)

# ---- Select best regression model
best_reg = reg_df.sort_values(
    by=["R2", "MAE"], ascending=[False, True]
).iloc[0]

print("\n🏆 BEST REGRESSION MODEL")
print(f"Model: {best_reg['Model']}")
print(f"Reason: Higher R² ({best_reg['R2']:.4f}) "
      f"and Lower MAE ({best_reg['MAE']:.2f})")

print("\n✅ Model comparison completed successfully!")
