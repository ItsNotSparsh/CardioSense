import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
from fpdf import FPDF  # NEW: Import for PDF generation
import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CardioSense AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #D93030;
        color: white;
        border-color: #D93030;
    }
    h1 {
        color: #2C3E50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h3 {
        color: #34495E;
        border-bottom: 2px solid #3498DB;
        padding-bottom: 10px;
    }
    .disclaimer-box {
        border: 1px solid #e0e0e0;
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 5px;
        color: #856404;
        font-size: 0.9em;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- PDF GENERATION CLASS (NEW) ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'CardioCast AI - Heart Health Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf_report(input_data, probability, prediction, status):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Date
    pdf.cell(0, 10, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'R')
    pdf.ln(10)
    
    # Result Section
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Assessment Results", 0, 1)
    pdf.set_font("Arial", size=12)
    
    risk_text = f"Risk Probability: {probability:.1%}"
    status_text = f"Classification: {status}"
    
    pdf.cell(0, 10, status_text, 0, 1)
    pdf.cell(0, 10, risk_text, 0, 1)
    pdf.ln(5)
    
    # Advice
    pdf.set_text_color(200, 0, 0)
    pdf.multi_cell(0, 10, "RECOMMENDATION: Please consult a doctor or cardiologist. This report is for informational purposes only and is not a medical diagnosis.")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    
    # Patient Data Table
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Patient Clinical Data", 0, 1)
    pdf.set_font("Arial", size=11)
    
    # Loop through inputs to print them nicely
    for key, value in input_data.items():
        # Format key for display
        display_key = key.replace('_', ' ').title()
        pdf.cell(90, 10, f"{display_key}: {value}", 1, 0)
        # Simple logic to create 2 columns
        if list(input_data.keys()).index(key) % 2 != 0:
            pdf.ln()
            
    return pdf.output(dest='S').encode('latin-1')

# --- LOAD MODEL ---
model_path = "new_model.pkl"

if not os.path.exists(model_path):
    st.error("⚠️ Critical Error: Model file not found!")
    st.info("Please run the 'generate_data.py' and 'train_model.py' scripts first.")
    st.stop()

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=100)
    st.title("CardioSense AI")
    st.markdown("### 🏥 Clinical Decision Support")
    st.info(
        """
        This AI tool assists medical professionals in estimating the 
        probability of Coronary Artery Disease (CAD) based on 15 clinical parameters.
        """
    )
    
    # --- RESET FUNCTIONALITY ---
    def reset_form():
        for key in st.session_state.keys():
            del st.session_state[key]
            
    st.button("🔄 Reset Form", on_click=reset_form)
    st.markdown("---")
    
    # --- NEW: MODEL INFO & CREDITS ---
    st.markdown("### 🤖 Model Info")
    st.caption("This app predicts the risk of heart disease based on user input using a trained **Random Forest Classifier**.")
    
    st.markdown("### 👨‍💻 Creator")
    st.success("Made by Sparsh")
    
    st.markdown("---")
    st.caption("v2.4 | Clean Start UI")

# --- MAIN HEADER ---
st.title("🫀 Heart Disease Risk Assessment")
st.markdown("Enter patient vitals and test results below to generate a risk profile.")

# --- INPUT FORM ---
with st.form("prediction_form"):
    
    # Section 1: Demographics & Lifestyle
    st.markdown("### 1. Patient Demographics & Lifestyle")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        age = st.number_input("Age", min_value=20, max_value=100, value=None, placeholder="e.g. 50")
    with c2:
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", index=None, placeholder="Select gender...")
    with c3:
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=None, step=0.1, placeholder="e.g. 24.5", help="Body Mass Index")
    with c4:
        smoking = st.selectbox("Smoker?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=None, placeholder="Select status...")

    st.markdown("") # Spacer

    # Section 2: Vitals & Blood Work
    st.markdown("### 2. Vitals & Blood Work")
    c1, c2, c3 = st.columns(3)
    with c1:
        trestbps = st.number_input("Resting BP (mm Hg)", min_value=80, max_value=220, value=None, placeholder="e.g. 120")
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "False" if x == 0 else "True", index=None, placeholder="Select...")
    with c2:
        chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=None, placeholder="e.g. 200")
        restecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x], index=None, placeholder="Select result...")
    with c3:
        thalach = st.number_input("Max Heart Rate", min_value=50, max_value=230, value=None, placeholder="e.g. 150")
        exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=None, placeholder="Select...")

    st.markdown("")

    # Section 3: Cardiac Tests (The heavy lifters)
    st.markdown("### 3. Specialized Cardiac Tests")
    c1, c2, c3 = st.columns(3)
    with c1:
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                          format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"][x], index=None, placeholder="Select type...")
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=None, step=0.1, placeholder="e.g. 1.0")
    with c2:
        slope = st.selectbox("ST Slope", [0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x], index=None, placeholder="Select slope...")
        ca = st.selectbox("Major Vessels Colored", [0, 1, 2, 3], index=None, placeholder="Select count (0-3)...")
    with c3:
        thal = st.selectbox("Thalassemia", [1, 2, 3], 
                            format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}.get(x, str(x)), index=None, placeholder="Select type...")

    # Submit Button
    st.markdown("---")
    submit_btn = st.form_submit_button("GENERATE RISK REPORT")

# --- PREDICTION LOGIC ---
if submit_btn:
    missing_fields = []
    
    # Check inputs
    if age is None: missing_fields.append("Age")
    if sex is None: missing_fields.append("Sex")
    if bmi is None: missing_fields.append("BMI")
    if smoking is None: missing_fields.append("Smoking Status")
    if cp is None: missing_fields.append("Chest Pain Type")
    if trestbps is None: missing_fields.append("Resting BP")
    if chol is None: missing_fields.append("Cholesterol")
    if fbs is None: missing_fields.append("Fasting BS")
    if restecg is None: missing_fields.append("Resting ECG")
    if thalach is None: missing_fields.append("Max Heart Rate")
    if exang is None: missing_fields.append("Exercise Angina")
    if oldpeak is None: missing_fields.append("Oldpeak")
    if slope is None: missing_fields.append("ST Slope")
    if ca is None: missing_fields.append("Major Vessels")
    if thal is None: missing_fields.append("Thalassemia")
    
    if missing_fields:
        st.warning(f"⚠️ Please fill in the following missing fields: {', '.join(missing_fields)}")
    else:
        # 1. Define features for the MODEL (Numbers)
        feature_order = ['age', 'sex', 'smoking', 'bmi', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        input_data = {
            'age': age, 'sex': sex, 'smoking': smoking, 'bmi': bmi, 'cp': cp, 
            'trestbps': trestbps, 'chol': chol, 'fbs': fbs, 'restecg': restecg, 
            'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 
            'ca': ca, 'thal': thal
        }
        
        input_df = pd.DataFrame([input_data])[feature_order]
        
        # Predict
        prob = model.predict_proba(input_df)[:, 1][0]
        prediction = model.predict(input_df)[0]

        # 2. Create human-readable data for the PDF REPORT (Text)
        report_data = input_data.copy()

        # Map the coded numbers back to text descriptions
        report_data['sex'] = "Male" if input_data['sex'] == 1 else "Female"
        report_data['smoking'] = "Yes" if input_data['smoking'] == 1 else "No"
        report_data['fbs'] = "True (>120 mg/dl)" if input_data['fbs'] == 1 else "False (<120 mg/dl)"
        report_data['exang'] = "Yes" if input_data['exang'] == 1 else "No"
        
        # Dictionary mappings for multi-option fields
        cp_map = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal", 3: "Asymptomatic"}
        restecg_map = {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}
        slope_map = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
        thal_map = {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}
        
        report_data['cp'] = cp_map.get(input_data['cp'], str(input_data['cp']))
        report_data['restecg'] = restecg_map.get(input_data['restecg'], str(input_data['restecg']))
        report_data['slope'] = slope_map.get(input_data['slope'], str(input_data['slope']))
        report_data['thal'] = thal_map.get(input_data['thal'], str(input_data['thal']))


        # --- RESULTS DASHBOARD ---
        st.markdown("### 📋 Analysis Result")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if prob < 0.2:
                status = "Low Risk"
                color = "green"
            elif prob < 0.6:
                status = "Moderate Risk"
                color = "orange"
            else:
                status = "High Risk"
                color = "red"
                
            st.markdown(f"""
                <div style="background-color: white; padding: 20px; border-radius: 10px; border-left: 10px solid {color}; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
                    <h4 style="margin:0; color: grey;">Diagnosis</h4>
                    <h2 style="margin:0; color: {color};">{status}</h2>
                    <p style="margin-top:10px; font-size: 14px;">Probability Score: <b>{prob:.1%}</b></p>
                </div>
            """, unsafe_allow_html=True)
            
            if prediction == 1:
                st.error("⚠️ The model indicates a significant likelihood of heart disease presence.")
            else:
                st.success("✅ The model indicates a low likelihood of heart disease.")

            # --- DOWNLOAD BUTTON ---
            st.markdown("---")
            # Pass 'report_data' (Text) instead of 'input_data' (Numbers)
            pdf_bytes = create_pdf_report(report_data, prob, prediction, status)
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name="heart_health_report.pdf",
                mime="application/pdf"
            )

        with col2:
            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                title = {'text': "Risk Probability (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 20], 'color': "#d4edda"},
                        {'range': [20, 60], 'color': "#fff3cd"},
                        {'range': [60, 100], 'color': "#f8d7da"}],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': prob * 100}
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

        # Detailed advice expander
        with st.expander("ℹ️ View Interpretation Guide"):
            st.write("""
            - **0-20% (Low Risk):** Maintain healthy lifestyle. Routine checkups.
            - **20-60% (Moderate Risk):** Monitor BP and Cholesterol. Consult a cardiologist if symptoms appear.
            - **60-100% (High Risk):** Urgent medical consultation recommended. Diagnostics (Stress test/Angiography) may be required.
            """)

# --- NEW: DISCLAIMER ON MAIN PAGE ---
st.markdown("""
    <div class="disclaimer-box">
        <b>Disclaimer:</b> This prediction is based on a machine learning model and is not a substitute for professional medical advice. 
        Please consult a doctor for any health concerns.
    </div>
""", unsafe_allow_html=True)