import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

# Page Setup for a wide premium feel
st.set_page_config(page_title="DrugAI Ultra", page_icon="🧬", layout="wide")

# Ultra-Modern CSS for Glassmorphism and Glow
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@400;700;900&display=swap');

    /* Global Background */
    .stApp {
        background: radial-gradient(circle at top right, #0f172a, #020617);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 25px;
        margin-bottom: 20px;
    }

    /* Titles */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 2px;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Futuristic Button */
    .stButton>button {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
        color: white;
        border: none;
        padding: 15px 40px;
        border-radius: 50px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 2px;
        width: 100%;
        box-shadow: 0 0 20px rgba(14, 165, 233, 0.4);
        transition: 0.4s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 30px rgba(99, 102, 241, 0.6);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    # Loading the core model and the scaler 
    model = joblib.load("knn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_assets()
    
    # Header Section
    st.markdown("<h1 style='text-align: center; margin-bottom: 50px;'>DRUG PREDICTION</h1>", unsafe_allow_html=True)

    # Main Dashboard Layout
    left_col, right_col = st.columns([1, 1.2], gap="large")

    with left_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📡 Input Parameters")
        
        # Modern Input Controls
        age = st.slider("Patient Chronological Age", 1, 100, 30)
        sex_raw = st.selectbox("Genetic Profile (Sex)", ["Female", "Male"])
        bp_raw = st.select_slider("Systemic Pressure (BP)", options=["Low", "Normal", "High"], value="Normal")
        chol_raw = st.radio("Lipid Density (Cholesterol)", ["Normal", "High"], horizontal=True)
        na_to_k = st.number_input("Ionic Balance (Na_to_K)", 0.0, 50.0, 15.6)
        
        st.markdown('</div>', unsafe_allow_html=True)
        predict_btn = st.button("Initialize Neural Analysis")

    # Data Processing Logic
    sex = 0 if sex_raw == "Female" else 1
    bp_map = {"High": 0, "Low": 1, "Normal": 2}
    bp = bp_map[bp_raw]
    chol = 0 if chol_raw == "High" else 1

    if predict_btn:
        with st.spinner("Processing Neural Pathways..."):
            time.sleep(1)
            raw_data = np.array([[age, sex, bp, chol, na_to_k]])
            # Transform data using the downloaded scaler
            scaled_data = scaler.transform(raw_data)
            
            # Predict outcome and probabilities
            prediction = model.predict(scaled_data)[0]
            probs = model.predict_proba(scaled_data)[0]
            
            drugs = {0: "Drug A", 1: "Drug B", 2: "Drug C", 3: "Drug X", 4: "Drug Y"}
            final_drug = drugs.get(prediction, "N/A")
            confidence = np.max(probs) * 100

        with right_col:
            st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
            st.subheader("🎯 Diagnosis Result")
            
            # Gauge Chart for Confidence
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence,
                number = {'suffix': "%", 'font': {'color': '#38bdf8'}},
                # MATCH FOUND के फॉन्ट को यहाँ अपडेट किया गया है
                title = {
                    'text': f"<span style='font-family: Inter; font-weight: 900; letter-spacing: 3px; color: #38bdf8;'>MATCH FOUND</span><br><span style='font-family: Orbitron; font-size: 32px;'>{final_drug}</span>", 
                    'font': {'size': 20}
                },
                gauge = {
                    'axis': {'range': [0, 100], 'tickcolor': "#f8fafc"},
                    'bar': {'color': "#0ea5e9"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.2)'},
                        {'range': [50, 80], 'color': 'rgba(234, 179, 8, 0.2)'},
                        {'range': [80, 100], 'color': 'rgba(34, 197, 94, 0.2)'}
                    ],
                }
            ))
            # Global chart layout
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'family': "Inter"}, height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

            # Secondary Probability Bar
            df_prob = pd.DataFrame({"Drug": list(drugs.values()), "Prob": probs * 100})
            fig_bar = go.Figure(go.Bar(
                x=df_prob["Prob"], y=df_prob["Drug"], orientation='h',
                marker=dict(color='rgba(56, 189, 248, 0.6)', line=dict(color='#38bdf8', width=2))
            ))
            fig_bar.update_layout(
                title="Secondary Candidates", template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300,
                font={'family': "Inter"}
            )
            st.plotly_chart(fig_bar, use_container_width=True)

except Exception as e:
    st.error(f"FATAL ERROR: {e}")