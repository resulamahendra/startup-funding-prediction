import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
import streamlit as st
from fpdf import FPDF
from datetime import datetime
import matplotlib.pyplot as plt
import os
import time
import random
import gdown   # ✅ ADDED (ONLY NEW IMPORT)

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Startup Funding Prediction", layout="centered")

# =====================================================
# CUSTOM CSS (DESIGN + ANIMATIONS)
# =====================================================
st.markdown("""
<style>
.main { background-color: #f7f9fc; }
h1 { font-weight: 700; }

.section-header {
    font-size: 22px;
    font-weight: 600;
    margin-top: 25px;
    margin-bottom: 10px;
    animation: bounce 0.6s ease-out;
}

@keyframes bounce {
    0% { transform: scale(0.9); opacity: 0; }
    60% { transform: scale(1.05); }
    100% { transform: scale(1); opacity: 1; }
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.08); }
    100% { transform: scale(1); }
}

@keyframes flyUp {
    from { bottom: 0; opacity: 1; }
    to { bottom: 260px; opacity: 0; }
}

@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 15px;
    animation: bounce 0.6s ease-out;
}

.prediction {
    font-size: 20px;
    font-weight: 700;
    color: #065f46;
    animation: pulse 1.5s infinite;
}

.stButton > button {
    background: linear-gradient(135deg,#6366f1,#22c55e,#f59e0b,#ec4899);
    background-size: 300% 300%;
    color: white;
    font-size: 16px;
    font-weight: 700;
    border-radius: 14px;
    height: 50px;
    animation: gradientMove 4s ease infinite, pulse 2s infinite;
}

.fly {
    position: fixed;
    font-size: 28px;
    animation: flyUp 2.5s ease-out forwards;
    z-index: 9999;
}

.footer {
    text-align: center;
    color: #6b7280;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODELS FROM GOOGLE DRIVE (ONLY FIX)
# =====================================================
@st.cache_resource
def load_resources():

    FILES = {
        "models/funding_level_model.pkl": "1tmnPz_Q9Acw9Dq8AMd4dpxHLjHMcTNnP",
        "models/funding_amount_model.pkl": "1s5bXqUBU30529e8Ejx0r0vUrfkHYWpEP",
        "models/label_encoders.pkl": "1s5bXqUBU30529e8Ejx0r0vUrfkHYWpEP"
    }

    os.makedirs("models", exist_ok=True)

    for path, file_id in FILES.items():
        if not os.path.exists(path):
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                path,
                quiet=False
            )

    return (
        joblib.load("models/funding_level_model.pkl"),
        joblib.load("models/funding_amount_model.pkl"),
        joblib.load("models/label_encoders.pkl")
    )

funding_level_model, funding_amount_model, label_encoders = load_resources()

# =====================================================
# HELPERS
# =====================================================
def sanitize_text(text):
    return str(text).replace("₹", "INR").replace("–", "-").replace("—", "-")

# =====================================================
# PDF CHARTS
# =====================================================
def generate_charts(level_text, encoded_features):
    paths = []
    level_map = {"Low Funding":1,"Medium Funding":2,"High Funding":3,"Very High Funding":4}

    plt.figure()
    plt.bar(["Funding Level"], [level_map[level_text]])
    plt.ylim(0,5)
    plt.savefig("funding_level.png")
    plt.close()
    paths.append("funding_level.png")

    plt.figure(figsize=(8,4))
    plt.bar(encoded_features.keys(), encoded_features.values())
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("encoded_features.png")
    plt.close()
    paths.append("encoded_features.png")

    return paths

# =====================================================
# PDF GENERATOR
# =====================================================
def generate_prediction_pdf(raw, enc, level_text, amount, insights, charts):

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial","B",16)
    pdf.cell(0,10,"Startup Funding Prediction Report",ln=True,align="C")

    pdf.set_font("Arial",size=11)
    pdf.cell(0,8,f"Generated on: {datetime.now()}",ln=True)

    for k,v in raw.items():
        pdf.cell(0,8,f"{k}: {sanitize_text(v)}",ln=True)

    pdf.cell(0,8,f"Funding Level: {level_text}",ln=True)
    pdf.cell(0,8,f"Estimated Amount: INR {amount:,.0f}",ln=True)

    for c in charts:
        if os.path.exists(c):
            pdf.image(c, w=170)

    for ins in insights:
        pdf.multi_cell(0,8,f"- {sanitize_text(ins)}")

    return pdf.output(dest="S").encode("latin-1")

# =====================================================
# APP UI
# =====================================================
st.title("🚀 Startup Funding Prediction Dashboard")
st.write("A machine learning based system for funding prediction & analytics")

industry = st.selectbox("Industry", label_encoders["industry"].classes_)
city = st.selectbox("City", label_encoders["city"].classes_)
investment_type = st.selectbox("Investment Type", label_encoders["investment_type"].classes_)
city_tier = st.selectbox("City Tier", label_encoders["city_tier"].classes_)
market_size = st.selectbox("Market Size Category", label_encoders["market_size_category"].classes_)
founded_year = st.number_input("Founded Year", 1990, 2030, 2020)
founders = st.number_input("Number of Founders", 1, 10, 2)

# =====================================================
# PREDICT
# =====================================================
if st.button("🔮 Predict Funding", use_container_width=True):

    with st.spinner("🧠 Brain is analyzing startup data..."):
        time.sleep(1.3)

    encoded = {
        "industry": label_encoders["industry"].transform([industry])[0],
        "city": label_encoders["city"].transform([city])[0],
        "investment_type": label_encoders["investment_type"].transform([investment_type])[0],
        "founded_year": founded_year,
        "no_of_founders": founders,
        "city_tier": label_encoders["city_tier"].transform([city_tier])[0],
        "market_size_category": label_encoders["market_size_category"].transform([market_size])[0]
    }

    df = pd.DataFrame([encoded])

    level = funding_level_model.predict(df)[0]
    amount = funding_amount_model.predict(df)[0]

    level_map = {0:"Low Funding",1:"Medium Funding",2:"High Funding",3:"Very High Funding"}
    level_text = level_map[level]

    if level_text in ["High Funding","Very High Funding"]:
        st.balloons()

    for _ in range(5):
        st.markdown(
            f"<div class='fly' style='left:{random.randint(10,90)}%'>🚀💰</div>",
            unsafe_allow_html=True
        )

    st.markdown(f"<div class='prediction'>Funding Level: {level_text}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='prediction'>Estimated Amount: INR {amount:,.0f}</div>", unsafe_allow_html=True)

    insights = [
        f"{industry} sector has strong investor interest",
        f"{city} has a growing startup ecosystem",
        f"{investment_type} indicates funding stage",
        f"{market_size} market shows scalability"
    ]

    charts = generate_charts(level_text, encoded)

    pdf = generate_prediction_pdf(
        {"Industry":industry,"City":city,"Founders":founders},
        encoded,
        level_text,
        amount,
        insights,
        charts
    )

    st.download_button(
        "⬇ Download Complete Prediction Report (PDF)",
        pdf,
        "Startup_Funding_Report.pdf",
        "application/pdf",
        use_container_width=True
    )

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">
Startup Funding Prediction & Investment Analytics<br>
Machine Learning Project
</div>
""", unsafe_allow_html=True)
