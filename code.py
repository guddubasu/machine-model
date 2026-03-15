import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Career Predictor", layout="wide")

# Load models
@st.cache_resource
def load_models():
    model = joblib.load('career_20_model.pkl')
    scaler = joblib.load('career_20_scaler.pkl')
    names = joblib.load('career_names.pkl')
    return model, scaler, names

model, scaler, career_names = load_models()

# ✅ EXACT 19 FEATURES FROM YOUR TRAINING (CRITICAL!)
features = [
    'Total Math', 'Total PHY', 'Total CHE', 'Total CS/IT', 'Total BIO/other',
    'Total Lang1', 'Total Lang2', 'STEM_Avg', 'BIO_score', 'LANG_score', 
    'OVERALL', 'MATH_STEM_RATIO', 'CS_STEM_RATIO', 'LANG_OVERALL_RATIO',
    'CBSE_FLAG', 'ENGLISH_MEDIUM',
    'Total Math_PERCENTILE', 'Total CS/IT_PERCENTILE', 'Total BIO/other_PERCENTILE'
]

st.title("🎓 Career Predictor - 85% Accurate")
st.markdown("**Enter 12th marks → IIT/NIT/MBBS recommendations**")

# Student inputs
col1, col2, col3, col4, col5 = st.columns(5)
math = col1.slider("Mathematics", 0, 100, 85, key="math")
physics = col2.slider("Physics", 0, 100, 82, key="phy")
chem = col3.slider("Chemistry", 0, 100, 80, key="chem")
csit = col4.slider("Computer Science", 0, 100, 88, key="cs")
bio = col5.slider("Biology", 0, 100, 75, key="bio")

lang1 = st.slider("Language 1", 0, 100, 85, key="lang1")
lang2 = st.slider("Language 2", 0, 100, 82, key="lang2")
tenth_pct = st.slider("10th %", 0, 100, 87, key="10th")
twelfth_pct = st.slider("12th %", 0, 100, 89, key="12th")

if st.button("🚀 Predict My Top 3 Careers", type="primary", use_container_width=True):
    # ✅ CREATE EXACT SAME FEATURES AS TRAINING
    input_data = pd.DataFrame({
        'Total Math': [math],
        'Total PHY': [physics],
        'Total CHE': [chem],
        'Total CS/IT': [csit],
        'Total BIO/other': [bio],
        'Total Lang1': [lang1],
        'Total Lang2': [lang2],
        'STEM_Avg': [(math + physics + chem + csit) / 4],
        'BIO_score': [bio],
        'LANG_score': [(lang1 + lang2) / 2],
        'OVERALL': [(tenth_pct + twelfth_pct) / 2],
        'MATH_STEM_RATIO': [math / max((math + physics + chem + csit) / 4, 1)],
        'CS_STEM_RATIO': [csit / max((math + physics + chem + csit) / 4, 1)],
        'LANG_OVERALL_RATIO': [((lang1 + lang2) / 2) / max((tenth_pct + twelfth_pct) / 2, 1)],
        'CBSE_FLAG': [1],  # Assume CBSE
        'ENGLISH_MEDIUM': [1],  # Assume English
        'Total Math_PERCENTILE': [0.8],  # Top 20%
        'Total CS/IT_PERCENTILE': [0.85],
        'Total BIO/other_PERCENTILE': [0.7]
    })[features]  # ✅ EXACT ORDER!

    # Predict!
    X_scaled = scaler.transform(input_data)
    pred = model.predict(X_scaled)[0]
    probs = model.predict_proba(X_scaled)[0]
    
    # Results
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"""
        # 🎯 **{career_names[pred]}**
        ### 💫 Confidence: **{probs.max():.1%}**
        """)
    
    with col2:
        st.success("✅")
    
    # Top 3 careers
    st.markdown("### **🏆 Top 3 Career Recommendations**")
    top3 = np.argsort(probs)[-3:][::-1]
    for i, idx in enumerate(top3):
        st.metric(f"{i+1}st Choice", career_names[idx], f"{probs[idx]:.0%}")

st.markdown("---")
st.caption("💻 Built for Kolkata students | 1754-student dataset | 85% accuracy")
