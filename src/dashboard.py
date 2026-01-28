import streamlit as st
from pdf_utils import generate_prediction_pdf

st.set_page_config(page_title="Startup Funding Dashboard")

st.title("Startup Funding Prediction Dashboard")

# ----- Inputs -----
sector = st.selectbox("Sector", ["FinTech", "EdTech", "HealthTech"])
location = st.selectbox("Location", ["Bangalore", "Hyderabad", "Mumbai"])
founding_year = st.number_input("Founding Year", 2000, 2025)
experience = st.selectbox("Founder Experience", ["Low", "Medium", "High"])

# ----- Prediction (dummy example) -----
prediction_result = "Startup is LIKELY to receive funding."

st.success(prediction_result)

inputs = {
    "Sector": sector,
    "Location": location,
    "Founding Year": founding_year,
    "Founder Experience": experience
}

# ----- PDF DATA -----
pdf_bytes = generate_prediction_pdf(inputs, prediction_result)

# ✅ THIS BUTTON WILL ALWAYS SHOW
st.download_button(
    label="📄 Download Prediction Output (PDF)",
    data=pdf_bytes,
    file_name="Startup_Funding_Prediction.pdf",
    mime="application/pdf"
)
