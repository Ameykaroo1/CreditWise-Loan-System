import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. SET PAGE CONFIG (Must be first)
st.set_page_config(page_title="CreditWise | Loan Prediction", page_icon="🏦", layout="wide")

# --- PREMIUM CUSTOM CSS INJECTION ---
st.markdown("""
    <style>
    /* Premium Background and Typography */
    .stApp {
        background-color: #F4F7F6;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    /* Center the main container and add padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Elegant Headers */
    h1 {
        color: #0F5132 !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }
    h2, h3 {
        color: #198754 !important;
        font-weight: 600 !important;
    }

    /* Style the Predict Button with a Premium Gradient */
    .stButton>button {
        background: linear-gradient(135deg, #198754 0%, #146C43 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 20px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(25, 135, 84, 0.3);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        width: 100%;
    }
    
    /* Button Hover Floating Animation */
    .stButton>button:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 25px rgba(25, 135, 84, 0.5);
        color: white;
    }
    
    /* Custom Animated Result Cards */
    .result-card {
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        margin-top: 20px;
    }
    
    .result-success {
        background: linear-gradient(135deg, #0F5132 0%, #198754 100%);
    }
    
    .result-error {
        background: linear-gradient(135deg, #842029 0%, #DC3545 100%);
    }
    
    .result-card h2 {
        color: white !important;
        margin-bottom: 10px;
        font-size: 32px;
    }
    
    .result-card p {
        font-size: 18px;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Slide up and fade in animation */
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# 2. Load the trained model and scaler
@st.cache_resource
def load_models():
    model = joblib.load("criditwise_loan.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_models()

feature_names = [
    'Applicant_ID', 'Applicant_Income', 'Coapplicant_Income', 'Age',
    'Dependents', 'Existing_Loans', 'Savings', 'Collateral_Value', 'Loan_Amount',
    'Loan_Term', 'Education_Level', 'Employment_Status_Salaried', 'Employment_Status_Self-employed',
    'Employment_Status_Unemployed', 'Marital_Status_Single', 'Loan_Purpose_Car',
    'Loan_Purpose_Education', 'Loan_Purpose_Home', 'Loan_Purpose_Personal',
    'Property_Area_Semiurban', 'Property_Area_Urban', 'Gender_Male',
    'Employer_Category_Government', 'Employer_Category_MNC', 'Employer_Category_Private',
    'Employer_Category_Unemployed', 'DTI_Ratio_sq', 'Credit_Score_sq'
]

def main():
    
    # --- HEADER ---
    st.markdown("<h1 style='text-align: center;'>🏦 CreditWise System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6c757d; font-size: 18px; margin-bottom: 30px;'>Intelligent Loan Approval Prediction Powered by Machine Learning</p>", unsafe_allow_html=True)

    # --- SECTION 1: Applicant Details (Enclosed in a visual card) ---
    with st.container(border=True):
        st.subheader("👤 Personal Profile")
        col1, col2, col3 = st.columns(3)
        with col1:
            applicant_id = st.text_input("Applicant ID", placeholder="e.g. CW-1029")
            age = st.number_input("Applicant Age", min_value=18.0, max_value=100.0, value=30.0, step=1.0)
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        with col3:
            dependents = st.number_input("Number of Dependents", min_value=0.0, max_value=10.0, value=0.0, step=1.0)
            education_level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    
    st.write("") # Small spacer

    # --- SECTION 2: Employment & Income ---
    with st.container(border=True):
        st.subheader("💼 Financial Profile")
        col4, col5, col6 = st.columns(3)
        with col4:
            employment_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])
            employer_category = st.selectbox("Employer Category", ["Private", "Government", "MNC", "Unemployed"])
        with col5:
            applicant_income = st.number_input("Applicant Income ($)", min_value=0.0, value=5000.0, step=100.0)
            coapplicant_income = st.number_input("Co-Applicant Income ($)", min_value=0.0, value=0.0, step=100.0)
        with col6:
            savings = st.number_input("Current Savings ($)", min_value=0.0, value=2000.0, step=100.0)
            existing_loans = st.number_input("Number of Existing Loans", min_value=0.0, value=1.0, step=1.0)

    st.write("")

    # --- SECTION 3: Loan Details ---
    with st.container(border=True):
        st.subheader("📝 Loan Request Details")
        col7, col8, col9 = st.columns(3)
        with col7:
            loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0.0, value=10000.0, step=100.0)
            loan_term = st.number_input("Loan Term (Months)", min_value=12.0, max_value=360.0, value=36.0, step=12.0)
        with col8:
            loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Car", "Education", "Home", "Business"])
            collateral_value = st.number_input("Collateral Value ($)", min_value=0.0, value=15000.0, step=100.0)
        with col9:
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            
    st.write("")

    # --- SECTION 4: Risk Assessment ---
    with st.container(border=True):
        st.subheader("⚖️ Risk Assessment Metrics")
        col10, col11 = st.columns(2)
        with col10:
            credit_score = st.number_input("Credit Score", min_value=300.0, max_value=850.0, value=700.0, step=10.0)
        with col11:
            dti_ratio = st.number_input("Debt-to-Income (DTI) Ratio", min_value=0.0, max_value=1.0, value=0.30, step=0.01)

    st.write("")
    st.write("")

    # --- PREDICT BUTTON ---
    # Centering the button using columns
    _, center_col, _ = st.columns([1, 2, 1])
    
    with center_col:
        if st.button("Analyze & Predict Loan Status", use_container_width=True):
            
            with st.spinner("Processing applicant data through CreditWise AI..."): 
                
                # --- Preprocess inputs ---
                ed_level_val = 1.0 if education_level == "Graduate" else 0.0
                mar_single = 1.0 if marital_status == "Single" else 0.0
                gen_male = 1.0 if gender == "Male" else 0.0
                emp_salaried = 1.0 if employment_status == "Salaried" else 0.0
                emp_self = 1.0 if employment_status == "Self-employed" else 0.0
                emp_unemp = 1.0 if employment_status == "Unemployed" else 0.0
                lp_car = 1.0 if loan_purpose == "Car" else 0.0
                lp_edu = 1.0 if loan_purpose == "Education" else 0.0
                lp_home = 1.0 if loan_purpose == "Home" else 0.0
                lp_per = 1.0 if loan_purpose == "Personal" else 0.0
                pa_semi = 1.0 if property_area == "Semiurban" else 0.0
                pa_urban = 1.0 if property_area == "Urban" else 0.0
                ec_gov = 1.0 if employer_category == "Government" else 0.0
                ec_mnc = 1.0 if employer_category == "MNC" else 0.0
                ec_priv = 1.0 if employer_category == "Private" else 0.0
                ec_unemp = 1.0 if employer_category == "Unemployed" else 0.0

                dti_ratio_sq = dti_ratio ** 2
                credit_score_sq = credit_score ** 2

                input_df = pd.DataFrame([[
                    1.0, applicant_income, coapplicant_income, age, dependents, existing_loans,
                    savings, collateral_value, loan_amount, loan_term, ed_level_val, emp_salaried,
                    emp_self, emp_unemp, mar_single, lp_car, lp_edu, lp_home, lp_per, pa_semi, pa_urban,
                    gen_male, ec_gov, ec_mnc, ec_priv, ec_unemp, dti_ratio_sq, credit_score_sq
                ]], columns=feature_names)
                
                # Predict
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)
                
                # Output results with Premium HTML cards
                if prediction[0] == 1:
                    st.markdown("""
                    <div class="result-card result-success">
                        <h2>🎉 Approved</h2>
                        <p>Based on our AI analysis, this applicant meets the criteria for a loan.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown("""
                    <div class="result-card result-error">
                        <h2>⚠️ Declined</h2>
                        <p>Based on our risk assessment, this application does not meet the approval criteria at this time.</p>
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
