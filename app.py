import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
import streamlit as st

# =====================================================
# LOAD MODELS & ENCODERS
# =====================================================
@st.cache_resource
def load_resources():
    funding_level_model = joblib.load("models/funding_level_model.pkl")
    funding_amount_model = joblib.load("models/funding_amount_model.pkl")
    label_encoders = joblib.load("models/label_encoders.pkl")
    return funding_level_model, funding_amount_model, label_encoders


# =====================================================
# STREAMLIT APP
# =====================================================
st.set_page_config(page_title="Startup Funding Prediction", layout="centered")

st.title("🚀 Startup Funding Prediction")
st.write("Enter startup details to predict funding level and funding amount.")

funding_level_model, funding_amount_model, label_encoders = load_resources()

# =====================================================
# INPUT SECTION
# =====================================================
st.subheader("📥 Enter Startup Details")

industry = st.selectbox(
    "Industry", label_encoders["industry"].classes_
)
city = st.selectbox(
    "City", label_encoders["city"].classes_
)
investment_type = st.selectbox(
    "Investment Type", label_encoders["investment_type"].classes_
)
city_tier = st.selectbox(
    "City Tier", label_encoders["city_tier"].classes_
)
market_size = st.selectbox(
    "Market Size Category", label_encoders["market_size_category"].classes_
)

founded_year = st.number_input(
    "Founded Year", min_value=1990, max_value=2030, value=2020
)
no_of_founders = st.number_input(
    "Number of Founders", min_value=1, max_value=10, value=2
)

# =====================================================
# PREDICTION
# =====================================================
if st.button("🔮 Predict Funding"):
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

    # 🔑 IMPORTANT: SAME FEATURE ORDER AS TRAINING
    FEATURE_ORDER = [
        "industry",
        "city",
        "investment_type",
        "founded_year",
        "no_of_founders",
        "city_tier",
        "market_size_category"
    ]
    startup_df = startup_df[FEATURE_ORDER]

    # Predictions
    funding_level_pred = funding_level_model.predict(startup_df)[0]
    funding_amount_pred = funding_amount_model.predict(startup_df)[0]

    funding_level_map = {
        0: "Low Funding",
        1: "Medium Funding",
        2: "High Funding",
        3: "Very High Funding"
    }

    st.success("✅ Prediction Completed")
    st.subheader("📊 Prediction Result")
    st.write("**Funding Level:**", funding_level_map[funding_level_pred])
    st.write("**Estimated Amount:** ₹ {:,.0f}".format(funding_amount_pred))
