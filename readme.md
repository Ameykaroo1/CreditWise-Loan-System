# 🏦 CreditWise Loan System

## 📌 Overview

CreditWise Loan System is a Machine Learning-powered web application that predicts loan approval eligibility based on an applicant's financial and personal information. The system helps streamline the loan screening process by providing fast, data-driven, and unbiased predictions through an interactive Streamlit interface.

🔗 Live Demo: https://creditwise-loan-system-ttsd93kt2jvbrtzfkasegm.streamlit.app/

---

## 🚀 Features

* Real-time loan approval prediction
* User-friendly Streamlit dashboard
* Financial risk assessment based on applicant details
* Automated data preprocessing and feature scaling
* Machine Learning model integration using Pickle files
* Clean and responsive UI
* Fast prediction results

---

## 🛠️ Tech Stack

### Frontend

* Streamlit

### Backend

* Python

### Libraries Used

* Pandas
* NumPy
* Scikit-learn
* Joblib
* Pickle

### Machine Learning

* Naive Bayes Classifier

---

## 📊 Input Parameters

The model uses multiple applicant-related features, including:

* Applicant Income
* Coapplicant Income
* Age
* Dependents
* Existing Loans
* Loan Amount
* Loan Term
* Credit Score
* Employment Status
* Education Level
* Property Area
* Marital Status
* Other financial indicators

---

## 🔄 Workflow

1. User enters applicant information.
2. Input data is preprocessed.
3. Features are scaled using the saved scaler.
4. The trained Naive Bayes model analyzes the data.
5. Prediction is generated instantly.
6. Loan approval status is displayed to the user.

---

## 📁 Project Structure

```bash
CreditWise-Loan-System/
│
├── app.py
├── criditwise_loan.pkl
├── scaler.pkl
├── label_encoder.pkl
├── loan_approval_data.csv
├── requirements.txt
├── README.md
└── assets/
```

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/Ameykaroo1/CreditWise-Loan-System.git
cd CreditWise-Loan-System
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 📈 Machine Learning Pipeline

* Data Cleaning
* Missing Value Handling
* Feature Engineering
* Label Encoding
* Feature Scaling
* Model Training
* Model Evaluation
* Deployment with Streamlit

---

## 🎯 Future Enhancements

* Loan approval probability score
* Explainable AI (Why Approved/Rejected)
* PDF report generation
* Database integration
* User authentication system
* Loan recommendation engine

---

## 👨‍💻 Author

**Amey Karoo**

* GitHub: https://github.com/Ameykaroo1
* LinkedIn: Add your LinkedIn profile here

---

## 📜 License

This project is developed for educational and learning purposes. Feel free to fork, modify, and enhance the project.
