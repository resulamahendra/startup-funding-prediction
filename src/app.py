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

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Startup Funding Prediction",
    layout="centered"
)

# =====================================================
# CUSTOM CSS (DESIGN + ANIMATIONS)
# =====================================================
st.markdown("""
<style>

/* ---------- BASE ---------- */
.main { background-color: #f7f9fc; }
h1 { font-weight: 700; }

/* ---------- SECTIONS ---------- */
.section-header {
    font-size: 22px;
    font-weight: 600;
    margin-top: 25px;
    margin-bottom: 10px;
    animation: bounce 0.6s ease-out;
}

/* ---------- ANIMATIONS ---------- */
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

/* ---------- CARDS ---------- */
.card {
    background-color: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 15px;
    animation: bounce 0.6s ease-out;
    transition: transform 0.25s ease;
}
.card:hover { transform: translateY(-6px); }

/* ---------- METRIC ---------- */
.prediction {
    font-size: 20px;
    font-weight: 700;
    color: #065f46;
    animation: pulse 1.5s infinite;
}

/* ---------- BUTTONS ---------- */
.stButton > button {
    background: linear-gradient(
        135deg,
        #6366f1,
        #22c55e,
        #f59e0b,
        #ec4899
    );
    background-size: 300% 300%;
    color: white;
    font-size: 16px;
    font-weight: 700;
    border-radius: 14px;
    height: 50px;
    border: none;
    animation: gradientMove 4s ease infinite, pulse 2s infinite;
    box-shadow: 0 8px 20px rgba(99,102,241,0.35);
}

.stButton > button:hover {
    transform: scale(1.08);
    box-shadow: 0 10px 26px rgba(236,72,153,0.45);
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #0ea5e9, #22c55e);
    color: white;
    font-weight: 700;
    border-radius: 14px;
    height: 50px;
    box-shadow: 0 8px 22px rgba(14,165,233,0.4);
}

/* ---------- FLYING EMOJIS ---------- */
.fly {
    position: fixed;
    font-size: 28px;
    animation: flyUp 2.5s ease-out forwards;
    z-index: 9999;
}

/* ---------- FOOTER ---------- */
.footer {
    text-align: center;
    color: #6b7280;
    margin-top: 40px;
    animation: bounce 0.8s ease-out;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODELS & ENCODERS
# =====================================================
@st.cache_resource
def load_resources():
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
    if not isinstance(text, str):
        text = str(text)
    for u, r in {"₹": "INR", "–": "-", "—": "-"}.items():
        text = text.replace(u, r)
    return text

# =====================================================
# PDF CHARTS
# =====================================================
def generate_charts(funding_level_text, encoded_features):
    paths = []
    level_map = {"Low Funding":1,"Medium Funding":2,"High Funding":3,"Very High Funding":4}

    plt.figure()
    plt.bar(["Funding Level"], [level_map[funding_level_text]])
    plt.ylim(0,5)
    plt.title("Predicted Funding Level")
    plt.ylabel("Score")
    plt.savefig("funding_level.png")
    plt.close()
    paths.append("funding_level.png")

    plt.figure(figsize=(8,4))
    plt.bar(encoded_features.keys(), encoded_features.values())
    plt.xticks(rotation=45, ha="right")
    plt.title("Encoded Feature Values Used by Model")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig("encoded_features.png")
    plt.close()
    paths.append("encoded_features.png")

    return paths

# =====================================================
# SHOW DASHBOARD CHARTS
# =====================================================
def show_dashboard_charts(level_text, encoded_features):
    st.markdown('<div class="section-header">📊 Visual Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    level_map = {"Low Funding":1,"Medium Funding":2,"High Funding":3,"Very High Funding":4}

    fig1, ax1 = plt.subplots()
    ax1.bar(["Funding Level"], [level_map[level_text]])
    ax1.set_ylim(0,5)
    ax1.set_ylabel("Score")
    ax1.set_title("Predicted Funding Level")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.bar(encoded_features.keys(), encoded_features.values())
    ax2.set_ylabel("Encoded Value")
    ax2.set_title("Encoded Feature Values Used by Model")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig2)

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# PDF GENERATOR
# =====================================================
def generate_prediction_pdf(raw_features, encoded_features, feature_order,
                            funding_level, funding_amount, insights, chart_paths):

    pdf = FPDF()
    pdf.set_auto_page_break(True,15)
    pdf.add_page()

    pdf.set_font("Arial","B",16)
    pdf.cell(0,10,"Startup Funding Prediction Report",ln=True,align="C")

    pdf.set_font("Arial",size=11)
    pdf.cell(0,8,f"Generated on: {datetime.now()}",ln=True)

    def section(t):
        pdf.ln(4)
        pdf.set_font("Arial","B",12)
        pdf.cell(0,8,t,ln=True)
        pdf.set_font("Arial",size=11)

    section("1. Raw Features")
    for k,v in raw_features.items():
        pdf.cell(0,8,sanitize_text(f"{k}: {v}"),ln=True)

    section("2. Encoded Features")
    for k,v in encoded_features.items():
        pdf.cell(0,8,f"{k}: {v}",ln=True)

    section("3. Prediction Output")
    pdf.cell(0,8,f"Funding Level: {funding_level}",ln=True)
    pdf.cell(0,8,f"Estimated Amount: INR {funding_amount:,.0f}",ln=True)

    section("4. Charts")
    for c in chart_paths:
        if os.path.exists(c):
            pdf.image(c, w=170)

    section("5. Insights")
    for ins in insights:
        pdf.multi_cell(0,8,f"- {sanitize_text(ins)}")

    return pdf.output(dest="S").encode("latin-1")

# =====================================================
# APP TITLE
# =====================================================
st.title("🚀 Startup Funding Prediction Dashboard")
st.write("A machine learning based system for funding prediction & analytics")

# =====================================================
# INPUTS
# =====================================================
st.markdown('<div class="section-header">📥 Startup Details</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

industry = st.selectbox("Industry", label_encoders["industry"].classes_)
city = st.selectbox("City", label_encoders["city"].classes_)
investment_type = st.selectbox("Investment Type", label_encoders["investment_type"].classes_)
city_tier = st.selectbox("City Tier", label_encoders["city_tier"].classes_)
market_size = st.selectbox("Market Size Category", label_encoders["market_size_category"].classes_)
founded_year = st.number_input("Founded Year", 1990, 2030, 2020)
no_of_founders = st.number_input("Number of Founders", 1, 10, 2)

st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# PREDICT
# =====================================================
if st.button("🔮 Predict Funding", use_container_width=True):

    with st.spinner("🧠 Brain is analyzing startup data..."):
        time.sleep(1.4)

    encoded_features = {
        "industry": label_encoders["industry"].transform([industry])[0],
        "city": label_encoders["city"].transform([city])[0],
        "investment_type": label_encoders["investment_type"].transform([investment_type])[0],
        "founded_year": founded_year,
        "no_of_founders": no_of_founders,
        "city_tier": label_encoders["city_tier"].transform([city_tier])[0],
        "market_size_category": label_encoders["market_size_category"].transform([market_size])[0]
    }

    df = pd.DataFrame([encoded_features])

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

    st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    placeholder = st.empty()
    for i in range(0, int(amount)+1, max(1,int(amount/15))):
        placeholder.markdown(
            f"<div class='prediction'>INR {i:,.0f} ({level_text})</div>",
            unsafe_allow_html=True
        )
        time.sleep(0.05)

    st.markdown('</div>', unsafe_allow_html=True)

    insights = [
        f"{industry} startups attract investors",
        f"{city} has a strong funding ecosystem",
        f"{investment_type} indicates startup stage",
        f"{market_size} market shows scalability"
    ]

    show_dashboard_charts(level_text, encoded_features)

    charts = generate_charts(level_text, encoded_features)

    pdf = generate_prediction_pdf(
        {
            "Industry":industry,"City":city,"Investment Type":investment_type,
            "City Tier":city_tier,"Market Size":market_size,
            "Founded Year":founded_year,"Founders":no_of_founders
        },
        encoded_features,
        list(encoded_features.keys()),
        level_text,
        amount,
        insights,
        charts
    )

    st.download_button(
        "⬇ Download Complete Prediction Report (PDF)",
        pdf,
        "Startup_Funding_Complete_Report.pdf",
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
