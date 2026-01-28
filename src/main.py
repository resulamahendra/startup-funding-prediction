# =====================================================
# GLOBAL SETUP
# =====================================================
import warnings
warnings.filterwarnings("ignore")

import sqlite3
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

from fpdf import FPDF
from datetime import datetime


# =====================================================
# PDF LOGGER (TERMINAL + PDF)
# =====================================================
class PDFLogger:
    def __init__(self, filename):
        self.lines = []
        self.filename = filename

    def log(self, text=""):
        print(text)
        self.lines.append(str(text))

    def save(self):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=10)

        for line in self.lines:
            pdf.multi_cell(0, 6, line)

        pdf.output(self.filename)


# Initialize logger
logger = PDFLogger(
    filename=f"Startup_Funding_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
)


# =====================================================
# PART 1: CSV-BASED TRAINING PIPELINE
# =====================================================
logger.log("========== STARTUP FUNDING PROJECT (TERMINAL MODE) ==========")

logger.log("\n===== STEP 1: DATA PREPARATION =====")
df = pd.read_csv("data/processed/startup_funding_cleaned.csv")

logger.log("Available columns:")
logger.log(", ".join(df.columns.tolist()))

# Encode categorical features
categorical_cols = df.select_dtypes(include="object").columns
label_encoders = {}

logger.log("\nEncoding categorical columns...")
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

logger.log(f"Encoded columns: {list(categorical_cols)}")

# Targets
y_reg = df["amount_in_inr"]
df["funding_level"] = pd.qcut(df["amount_in_inr"], q=4, labels=[0, 1, 2, 3])
y_cls = df["funding_level"]

logger.log("\nFunding level distribution:")
logger.log(str(y_cls.value_counts()))

# Features
X = df.drop(columns=["amount_in_inr", "funding_level"])

# Train-test split
X_train, X_test, y_train_cls, y_test_cls = train_test_split(
    X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)
_, _, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

logger.log(f"\nTrain shape: {X_train.shape}")
logger.log(f"Test shape : {X_test.shape}")

# =====================================================
# STEP 2: BASELINE MODEL TRAINING
# =====================================================
logger.log("\n===== BASELINE MODEL TRAINING =====")

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

logger.log("\nLogistic Regression")
logger.log(f"Accuracy : {accuracy_score(y_test_cls, lr_pred):.4f}")
logger.log(f"F1 Score : {f1_score(y_test_cls, lr_pred, average='weighted'):.4f}")

logger.log("\nDecision Tree")
logger.log(f"Accuracy : {accuracy_score(y_test_cls, dt_pred):.4f}")
logger.log(f"F1 Score : {f1_score(y_test_cls, dt_pred, average='weighted'):.4f}")

logger.log("\nRandom Forest")
logger.log(f"Accuracy : {accuracy_score(y_test_cls, rf_pred):.4f}")
logger.log(f"F1 Score : {f1_score(y_test_cls, rf_pred, average='weighted'):.4f}")

logger.log("\nObservation:")
logger.log("Random Forest performs best among baseline models")
logger.log("===== BASELINE MODELING COMPLETED =====")

# =====================================================
# STEP 3: MODEL EVALUATION
# =====================================================
logger.log("\n===== MODEL EVALUATION =====")

logger.log("\nClassification Evaluation:")
logger.log(f"Accuracy : {accuracy_score(y_test_cls, rf_pred):.4f}")
logger.log(f"F1 Score : {f1_score(y_test_cls, rf_pred, average='weighted'):.4f}")

logger.log("\nConfusion Matrix:")
logger.log(str(confusion_matrix(y_test_cls, rf_pred)))

cv_scores_rf = cross_val_score(rf, X, y_cls, cv=5, scoring="accuracy")
logger.log("\nCross Validation Accuracy Scores:")
logger.log(str(np.round(cv_scores_rf, 4)))
logger.log(f"Mean CV Accuracy : {cv_scores_rf.mean():.4f}")

# Regression
rf_reg = RandomForestRegressor(n_estimators=300, random_state=42)
rf_reg.fit(X_train, y_train_reg)
reg_pred = rf_reg.predict(X_test)

mae = np.mean(np.abs(y_test_reg - reg_pred))
r2 = rf_reg.score(X_test, y_test_reg)

logger.log("\nRegression Evaluation:")
logger.log(f"MAE : {mae:.2e}")
logger.log(f"R2  : {r2:.4f}")

logger.log("===== MODEL EVALUATION COMPLETED =====")

# =====================================================
# STEP 4: OPTIMIZATION INSIGHTS
# =====================================================
logger.log("\n===== OPTIMIZATION INSIGHTS =====")

baseline_scores = cross_val_score(
    lr_pipeline, X, y_cls, cv=5, scoring="accuracy"
)

logger.log(f"Baseline Mean Accuracy : {baseline_scores.mean():.4f}")
logger.log(f"Optimized Mean Accuracy: {cv_scores_rf.mean():.4f}")

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

logger.log("\nTop 5 Important Features:")
logger.log(str(feature_importance.head(5)))

# =====================================================
# STEP 5: STATISTICAL VALIDATION
# =====================================================
logger.log("\n===== STATISTICAL VALIDATION =====")

t_stat, p_value = ttest_rel(cv_scores_rf, baseline_scores)

logger.log(f"t-statistic : {t_stat:.4f}")
logger.log(f"p-value     : {p_value:.6f}")

if p_value < 0.05:
    logger.log("Decision: Reject Null Hypothesis (Significant improvement)")
else:
    logger.log("Decision: Fail to Reject Null Hypothesis")

logger.log("===== STATISTICAL VALIDATION COMPLETED =====")

# =====================================================
# STEP 6: DATABASE-BASED MODEL EVALUATION
# =====================================================
logger.log("\n===== DATABASE-BASED MODEL EVALUATION =====")

conn = sqlite3.connect("data/startup_funding.db")
X_db = pd.read_sql("SELECT * FROM X_features", conn)
y_level_db = pd.read_sql("SELECT * FROM y_funding_level", conn).values.ravel()
y_amount_db = pd.read_sql("SELECT * FROM y_funding_amount", conn).values.ravel()
conn.close()

# Align feature order
X_db = X_db[X.columns]

X_tr, X_te, y_lvl_tr, y_lvl_te, y_amt_tr, y_amt_te = train_test_split(
    X_db, y_level_db, y_amount_db,
    test_size=0.2,
    random_state=42,
    stratify=y_level_db
)

lvl_pred_db = rf.predict(X_te)
amt_pred_db = rf_reg.predict(X_te)

logger.log("\nFunding Level Evaluation (DB):")
logger.log(f"Accuracy : {accuracy_score(y_lvl_te, lvl_pred_db):.4f}")
logger.log(f"F1 Score : {f1_score(y_lvl_te, lvl_pred_db, average='weighted'):.4f}")
logger.log("Confusion Matrix:")
logger.log(str(confusion_matrix(y_lvl_te, lvl_pred_db)))

mae_db = np.mean(np.abs(y_amt_te - amt_pred_db))
logger.log(f"\nFunding Amount MAE (DB): {mae_db:.2e}")

# =====================================================
# STEP 7: INTERACTIVE TERMINAL PREDICTION
# =====================================================
logger.log("\n===== STARTUP FUNDING PREDICTION =====")

def get_valid_input(name, encoder):
    logger.log(f"\nAvailable {name} options:")
    for v in encoder.classes_:
        logger.log(f"- {v}")
    while True:
        val = input(f"Enter {name}: ").strip()
        if val in encoder.classes_:
            return val
        print("Invalid input, try again.")

industry = get_valid_input("industry", label_encoders["industry"])
city = get_valid_input("city", label_encoders["city"])
investment_type = get_valid_input("investment_type", label_encoders["investment_type"])
city_tier = get_valid_input("city_tier", label_encoders["city_tier"])
market_size = get_valid_input("market_size_category", label_encoders["market_size_category"])

founded_year = int(input("Enter founded_year: "))
no_of_founders = int(input("Enter no_of_founders: "))

startup_encoded = {
    "industry": label_encoders["industry"].transform([industry])[0],
    "city": label_encoders["city"].transform([city])[0],
    "investment_type": label_encoders["investment_type"].transform([investment_type])[0],
    "founded_year": founded_year,
    "no_of_founders": no_of_founders,
    "city_tier": label_encoders["city_tier"].transform([city_tier])[0],
    "market_size_category": label_encoders["market_size_category"].transform([market_size])[0],
}

startup_df = pd.DataFrame([startup_encoded])[X.columns]

funding_level_pred = rf.predict(startup_df)[0]
funding_amount_pred = rf_reg.predict(startup_df)[0]

level_map = {
    0: "Low Funding",
    1: "Medium Funding",
    2: "High Funding",
    3: "Very High Funding"
}

logger.log("\nPrediction Result:")
logger.log(f"Funding Level    : {level_map[funding_level_pred]}")
logger.log(f"Estimated Amount : ₹ {funding_amount_pred:,.0f}")

# =====================================================
# SAVE PDF
# =====================================================
logger.log("\n===== OUTPUT SAVED TO PDF =====")
logger.save()
logger.log("PDF generated successfully.")

logger.log("\n========== END OF MAIN EXECUTION ==========")
