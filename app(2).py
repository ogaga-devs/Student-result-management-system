# ================================
# app.py
# ================================
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ================================
# Load pre-trained model and scaler
# ================================
# Make sure model.pkl and scaler.pkl are uploaded to the repo
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ================================
# Streamlit UI
# ================================
st.title("Student Result Management System")
st.write("Predict whether a student is At Risk or Not At Risk")

# Inputs (match your dataset order)
student_id = st.number_input("Student ID", 0)
age = st.number_input("Age", 10, 40)
gender = st.selectbox("Gender", ["Male", "Female"])
school_type = st.selectbox("School Type", ["Public", "Private"])
parent_education = st.number_input("Parent Education Level", 0, 5)
study_hours = st.number_input("Study Hours", 0)
attendance = st.slider("Attendance Percentage", 0, 100)
internet = st.selectbox("Internet Access", ["No", "Yes"])
travel_time = st.number_input("Travel Time", 0)
extra = st.selectbox("Extra Activities", ["No", "Yes"])
study_method = st.selectbox("Study Method", ["Self", "Group"])
math = st.number_input("Math Score")
science = st.number_input("Science Score")
english = st.number_input("English Score")
overall = st.number_input("Overall Score")

# Manual encoding
gender = 1 if gender == "Male" else 0
school_type = 1 if school_type == "Private" else 0
internet = 1 if internet == "Yes" else 0
extra = 1 if extra == "Yes" else 0
study_method = 1 if study_method == "Group" else 0

if st.button("Predict"):
    data = np.array([[student_id, age, gender, school_type,
                      parent_education, study_hours, attendance,
                      internet, travel_time, extra, study_method,
                      math, science, english, overall]])
    
    scaled = scaler.transform(data)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][pred]

    if pred == 1:
        st.success(f"Not At Risk (Confidence: {prob:.2%})")
    else:
        st.error(f"At Risk (Confidence: {prob:.2%})")
