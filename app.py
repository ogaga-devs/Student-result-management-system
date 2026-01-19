# IMPORT LIBRARIES
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Students Result Management System", layout="wide")

# LOAD DATA
st.title("🎓 Students Result Management System")

@st.cache_data
def load_data():
    return pd.read_csv("Student_Performance.csv")

srms = load_data()

# Create Student ID if not exists
if 'Student_ID' not in srms.columns:
    srms['Student_ID'] = (srms.index + 1).astype(str)

# PREVIEW DATA
st.subheader("Dataset Preview")
st.dataframe(srms.head())

# MISSING VALUES
st.subheader("Missing Values Heatmap")
fig, ax = plt.subplots()
sns.heatmap(srms.isnull(), cbar=False, ax=ax)
st.pyplot(fig)

# DESCRIPTIVE STATISTICS
st.subheader("📊 Descriptive Statistics")
st.dataframe(srms.describe().T)

# GRADE → RISK RULE
def grade_based_risk(grade):
    return 1 if grade in ['A', 'B', 'C'] else 0  # 1=Not At Risk, 0=At Risk

# Preserve grade for display
srms['Student_Grade'] = srms['Final_grade']

# Create Risk label
srms['Risk'] = srms['Final_grade'].apply(grade_based_risk)

# Drop original grade for modeling
srms.drop(columns=['Final_grade'], inplace=True)

# EDA
st.subheader("Risk Distribution")
fig, ax = plt.subplots()
sns.countplot(x='Risk', data=srms, ax=ax)
ax.set_xticklabels(["At Risk", "Not At Risk"])
st.pyplot(fig)

st.subheader("Overall Score vs Risk")
fig, ax = plt.subplots()
sns.boxplot(x='Risk', y='Overall_score', data=srms, ax=ax)
ax.set_xticklabels(["At Risk", "Not At Risk"])
st.pyplot(fig)

# ENCODING
categorical_cols = srms.select_dtypes(include='object').columns
categorical_cols = categorical_cols.drop(['Student_ID', 'Student_Grade'], errors='ignore')

for col in categorical_cols:
    le = LabelEncoder()
    srms[col] = le.fit_transform(srms[col])

# CORRELATION
st.subheader("Correlation Matrix")
numeric_cols = srms.select_dtypes(include=np.number)  # only numeric
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# MODEL TRAINING
features = [
    'Math_score',
    'Science_score',
    'English_score',
    'Overall_score',
    'Extra_activities',
    'Study_method'
]

X = srms[features]
y = srms['Risk']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# MODEL PERFORMANCE
st.subheader("Model Performance")
y_pred = model.predict(X_test)

st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred))

st.subheader("Classification Report")
st.code(classification_report(y_test, y_pred))

# FEATURE IMPORTANCE
st.subheader("📊 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.coef_[0]
}).sort_values(by="Importance")

fig, ax = plt.subplots()
sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
st.pyplot(fig)

# SIDEBAR INPUTS
st.sidebar.header("🔮 Manual Prediction")

Math_score = st.sidebar.slider("Math Score", 0, 100, int(srms['Math_score'].mean()))
Science_score = st.sidebar.slider("Science Score", 0, 100, int(srms['Science_score'].mean()))
English_score = st.sidebar.slider("English Score", 0, 100, int(srms['English_score'].mean()))
Overall_score = st.sidebar.slider("Overall Score", 0, 100, int(srms['Overall_score'].mean()))
Extra_activities = st.sidebar.slider("Extra Activities", 0, 5, int(srms['Extra_activities'].mean()))
Study_method = st.sidebar.slider("Study Method", 0, 5, int(srms['Study_method'].mean()))

# STUDENT ID CHECK
st.sidebar.header("🔎 Check Student by ID")
student_id_input = st.sidebar.text_input("Enter Student ID")

# STUDENT ID PREDICTION
if student_id_input:
    if student_id_input in srms['Student_ID'].values:
        student = srms[srms['Student_ID'] == student_id_input].iloc[0]

        # Predict probabilities
        input_scaled = scaler.transform(student[features].values.reshape(1, -1))
        probabilities = model.predict_proba(input_scaled)[0]

        final_risk = np.argmax(probabilities)
        risk_status = "Not At Risk ✅" if final_risk == 1 else "At Risk ⚠️"

        # Display student info
        st.subheader(f"Student ID: {student_id_input}")
        st.markdown(f"**Final Grade:** {student['Student_Grade']}")
        st.markdown(f"**Risk Status:** {risk_status}")
        st.markdown(f"**Model Confidence:** {probabilities[final_risk]:.2f}")

        # ✅ Display subject scores / numeric features
        st.markdown("### 📊 Student Scores & Features")
        st.dataframe(student[features].to_frame().T)

        # Display prediction probabilities
        st.markdown("### 📌 Risk Probabilities")
        st.table(pd.DataFrame({
            "Risk Status": ["At Risk", "Not At Risk"],
            "Probability": probabilities
        }))

    else:
        st.error("❌ Student ID not found")

# MANUAL PREDICTION
elif st.sidebar.button("Predict Risk"):
    input_df = pd.DataFrame([[Math_score, Science_score, English_score,
                              Overall_score, Extra_activities, Study_method]],
                            columns=features)

    input_scaled = scaler.transform(input_df)
    probabilities = model.predict_proba(input_scaled)[0]

    pred_risk = np.argmax(probabilities)
    risk_status = "Not At Risk ✅" if pred_risk == 1 else "At Risk ⚠️"

    st.subheader("📊 Manual Prediction Result")
    st.markdown(f"**Risk Status:** {risk_status}")
    st.markdown(f"**Model Confidence:** {probabilities[pred_risk]:.2f}")

    st.table(pd.DataFrame({
        "Risk Status": ["At Risk", "Not At Risk"],
        "Probability": probabilities
    }))



