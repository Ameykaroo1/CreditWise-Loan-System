import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

st.set_page_config(
    page_title="CreditWise — Loan Approval Predictor",
    page_icon="💳",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.main { background: #f0f4ff; }

.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    color: white;
}
.hero h1 { font-size: 2.2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
.hero p  { font-size: 1rem; margin: 0.5rem 0 0; opacity: 0.88; }

.section-card {
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}
.sec-income  { background: #fff7ed; border-left: 5px solid #f97316; }
.sec-loan    { background: #f0fdf4; border-left: 5px solid #22c55e; }
.sec-profile { background: #eff6ff; border-left: 5px solid #3b82f6; }
.sec-employ  { background: #fdf4ff; border-left: 5px solid #a855f7; }

.section-title {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.75rem;
}
.sec-income  .section-title { color: #c2410c; }
.sec-loan    .section-title { color: #15803d; }
.sec-profile .section-title { color: #1d4ed8; }
.sec-employ  .section-title { color: #7e22ce; }

.result-approved {
    background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
    border: 2px solid #10b981;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}
.result-rejected {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    border: 2px solid #ef4444;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}
.result-title { font-size: 2rem; font-weight: 700; margin: 0.5rem 0; }
.result-approved .result-title { color: #065f46; }
.result-rejected .result-title { color: #7f1d1d; }

.confidence-bar-wrap {
    background: rgba(255,255,255,0.6);
    border-radius: 999px;
    height: 14px;
    margin: 0.75rem 0;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}
.approved-fill { background: linear-gradient(90deg, #10b981, #059669); }
.rejected-fill { background: linear-gradient(90deg, #ef4444, #dc2626); }

.factor-positive {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
    margin: 0.4rem 0;
    font-size: 0.88rem;
    color: #14532d;
}
.factor-negative {
    background: #fff1f2;
    border: 1px solid #fca5a5;
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
    margin: 0.4rem 0;
    font-size: 0.88rem;
    color: #7f1d1d;
}

.stat-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
}
.pill-blue   { background: #dbeafe; color: #1e40af; }
.pill-purple { background: #ede9fe; color: #5b21b6; }
.pill-green  { background: #dcfce7; color: #166534; }
.pill-orange { background: #ffedd5; color: #9a3412; }

div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] select {
    border-radius: 10px !important;
    border: 1.5px solid #e2e8f0 !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stSelectbox"] select:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.12) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2.5rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    font-family: 'Poppins', sans-serif !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

.reset-btn > button {
    background: white !important;
    color: #6b7280 !important;
    border: 1.5px solid #e5e7eb !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("ohe.pkl", "rb") as f:
        ohe = pickle.load(f)
    with open("feature_cols.json", "r") as f:
        feature_cols = json.load(f)
    return model, scaler, ohe, feature_cols


def build_input(data, ohe, feature_cols):
    num_fields = ["Applicant_Income", "Coapplicant_Income", "Age", "Dependents",
                  "Credit_Score", "Existing_Loans", "DTI_Ratio", "Savings",
                  "Collateral_Value", "Loan_Amount", "Loan_Term"]
    edu_map = {"Graduate": 1, "Not Graduate": 0}

    ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose",
                "Property_Area", "Gender", "Employer_Category"]
    ohe_input = pd.DataFrame([[data[c] for c in ohe_cols]], columns=ohe_cols)
    ohe_encoded = ohe.transform(ohe_input)
    ohe_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out(ohe_cols))

    base = {f: [data[f]] for f in num_fields}
    base["Education_Level"] = [edu_map[data["Education_Level"]]]
    base_df = pd.DataFrame(base)

    full_df = pd.concat([base_df.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)
    full_df = full_df.reindex(columns=feature_cols, fill_value=0)
    return full_df


model, scaler, ohe, feature_cols = load_model()

st.markdown("""
<div class="hero">
    <h1>💳 CreditWise</h1>
    <p>AI-powered loan approval predictor · Naive Bayes · 86.5% accuracy</p>
</div>
""", unsafe_allow_html=True)

with st.form("loan_form"):

    st.markdown('<div class="section-card sec-income"><div class="section-title">💰 Income & Financial Details</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    applicant_income    = c1.number_input("Applicant Income (₹)", min_value=2009, max_value=19988, value=10000, step=100)
    coapplicant_income  = c2.number_input("Co-applicant Income (₹)", min_value=0, max_value=9996, value=0, step=100)
    savings             = c3.number_input("Savings (₹)", min_value=65, max_value=19996, value=5000, step=100)
    c4, c5 = st.columns(2)
    collateral_value    = c4.number_input("Collateral Value (₹)", min_value=36, max_value=49954, value=20000, step=500)
    dti_ratio           = c5.number_input("Debt-to-Income Ratio", min_value=0.10, max_value=0.60, value=0.30, step=0.01, format="%.2f")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card sec-loan"><div class="section-title">🏦 Loan Details</div>', unsafe_allow_html=True)
    l1, l2, l3 = st.columns(3)
    loan_amount  = l1.number_input("Loan Amount (₹)", min_value=1015, max_value=39995, value=15000, step=100)
    loan_term    = l2.number_input("Loan Term (months)", min_value=12, max_value=84, value=36, step=6)
    loan_purpose = l3.selectbox("Loan Purpose", ["Home", "Car", "Education", "Business", "Personal"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card sec-profile"><div class="section-title">👤 Applicant Profile</div>', unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    age            = p1.number_input("Age", min_value=21, max_value=59, value=30)
    credit_score   = p2.number_input("Credit Score", min_value=550, max_value=799, value=680)
    dependents     = p3.number_input("Dependents", min_value=0, max_value=3, value=0)
    existing_loans = p4.number_input("Existing Loans", min_value=0, max_value=4, value=0)
    p5, p6, p7, p8 = st.columns(4)
    gender          = p5.selectbox("Gender", ["Male", "Female"])
    marital_status  = p6.selectbox("Marital Status", ["Married", "Single"])
    education_level = p7.selectbox("Education Level", ["Graduate", "Not Graduate"])
    property_area   = p8.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card sec-employ"><div class="section-title">🏢 Employment Details</div>', unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    employment_status  = e1.selectbox("Employment Status", ["Salaried", "Self-employed", "Contract", "Unemployed"])
    employer_category  = e2.selectbox("Employer Category", ["Private", "Government", "MNC", "Business", "Unemployed"])
    st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("🔍  Predict Loan Approval")


if submitted:
    data = {
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Age": age,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Education_Level": education_level,
        "Employment_Status": employment_status,
        "Marital_Status": marital_status,
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Gender": gender,
        "Employer_Category": employer_category,
    }

    input_df   = build_input(data, ohe, feature_cols)
    scaled     = scaler.transform(input_df)
    prediction = model.predict(scaled)[0]
    proba      = model.predict_proba(scaled)[0]
    confidence = int(round(proba[prediction] * 100))
    decision   = "Approved" if prediction == 1 else "Rejected"

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    col_res, col_factors = st.columns([1, 1])

    with col_res:
        css_class   = "result-approved" if decision == "Approved" else "result-rejected"
        icon        = "✅" if decision == "Approved" else "❌"
        bar_class   = "approved-fill" if decision == "Approved" else "rejected-fill"
        st.markdown(f"""
        <div class="{css_class}">
            <div style="font-size:3rem">{icon}</div>
            <div class="result-title">Loan {decision}</div>
            <div style="font-size:0.9rem; opacity:0.75; margin-top:0.25rem">Naive Bayes Classifier</div>
            <div class="confidence-bar-wrap">
                <div class="confidence-bar-fill {bar_class}" style="width:{confidence}%"></div>
            </div>
            <div style="font-size:1.4rem; font-weight:700; color:{'#065f46' if decision=='Approved' else '#7f1d1d'}">
                {confidence}% confidence
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📈 Key Metrics")
        m1, m2 = st.columns(2)
        m1.metric("Credit Score", credit_score, delta="Good" if credit_score >= 700 else "Low")
        m2.metric("DTI Ratio", f"{dti_ratio:.2f}", delta="Low" if dti_ratio < 0.3 else "High", delta_color="inverse")
        m3, m4 = st.columns(2)
        m3.metric("Monthly Income", f"₹{applicant_income:,}")
        m4.metric("Loan Amount",    f"₹{loan_amount:,}")

    with col_factors:
        st.markdown("#### 🔍 Influencing Factors")

        factors = []
        if credit_score >= 720:
            factors.append(("positive", "Credit Score", f"{credit_score} — excellent credit history"))
        elif credit_score >= 650:
            factors.append(("positive", "Credit Score", f"{credit_score} — decent credit score"))
        else:
            factors.append(("negative", "Credit Score", f"{credit_score} — below ideal threshold"))

        if dti_ratio < 0.3:
            factors.append(("positive", "DTI Ratio", f"{dti_ratio:.2f} — low debt burden"))
        elif dti_ratio < 0.45:
            factors.append(("neutral",  "DTI Ratio", f"{dti_ratio:.2f} — moderate debt burden"))
        else:
            factors.append(("negative", "DTI Ratio", f"{dti_ratio:.2f} — high debt-to-income ratio"))

        income_ratio = applicant_income / max(loan_amount, 1)
        if income_ratio > 0.8:
            factors.append(("positive", "Income vs Loan", f"Ratio {income_ratio:.2f} — strong repayment capacity"))
        elif income_ratio > 0.4:
            factors.append(("positive", "Income vs Loan", f"Ratio {income_ratio:.2f} — adequate income"))
        else:
            factors.append(("negative", "Income vs Loan", f"Ratio {income_ratio:.2f} — income may be insufficient"))

        if existing_loans == 0:
            factors.append(("positive", "Existing Loans", "No existing loans — clean slate"))
        elif existing_loans <= 2:
            factors.append(("neutral",  "Existing Loans", f"{existing_loans} existing loan(s)"))
        else:
            factors.append(("negative", "Existing Loans", f"{existing_loans} existing loans — high debt load"))

        if employment_status in ["Salaried"] and employer_category in ["Government", "MNC"]:
            factors.append(("positive", "Employment", f"{employment_status} at {employer_category} — very stable"))
        elif employment_status == "Unemployed":
            factors.append(("negative", "Employment", "Unemployed — no regular income"))
        else:
            factors.append(("positive", "Employment", f"{employment_status} — {employer_category}"))

        if savings >= 10000:
            factors.append(("positive", "Savings", f"₹{savings:,} — strong savings buffer"))
        elif savings >= 3000:
            factors.append(("positive", "Savings", f"₹{savings:,} — moderate savings"))
        else:
            factors.append(("negative", "Savings", f"₹{savings:,} — low savings"))

        if education_level == "Graduate":
            factors.append(("positive", "Education", "Graduate — slight approval advantage"))

        for impact, name, note in factors:
            css = "factor-positive" if impact == "positive" else "factor-negative"
            icon_f = "✅" if impact == "positive" else "⚠️"
            st.markdown(f'<div class="{css}">{icon_f} <strong>{name}</strong> — {note}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📋 Applicant Summary")
    cols = st.columns(4)
    pills = [
        ("pill-blue",   f"👤 {gender}, {age} yrs"),
        ("pill-purple", f"🎓 {education_level}"),
        ("pill-green",  f"💼 {employment_status} · {employer_category}"),
        ("pill-orange", f"📍 {property_area}"),
        ("pill-blue",   f"💍 {marital_status}"),
        ("pill-purple", f"👨‍👩‍👧 {dependents} dependent(s)"),
        ("pill-green",  f"🏠 {loan_purpose} loan"),
        ("pill-orange", f"📅 {int(loan_term)} months"),
    ]
    for i, (cls, text) in enumerate(pills):
        cols[i % 4].markdown(f'<span class="stat-pill {cls}">{text}</span>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; margin-top:2.5rem; color:#9ca3af; font-size:0.8rem;">
    CreditWise · Naive Bayes Classifier · Trained on 1,000 applicant records
</div>
""", unsafe_allow_html=True)

