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

st.set_page_config(page_title="Student Result Management System", layout="wide")

# LOAD DATA
st.title("🎓 Student Result Management System")

@st.cache_data
def load_data():
    return pd.read_csv("Student_Performance.csv")

srms = load_data()

st.subheader("Dataset Preview")
st.dataframe(srms.head())

# MISSING VALUES HEATMAP
st.subheader("Missing Values Heatmap")
fig, ax = plt.subplots()
sns.heatmap(srms.isnull(), cbar=False, ax=ax)
st.pyplot(fig)

# DESCRIPTIVE STATISTICAL SUMMARY
st.subheader("📊 Descriptive Statistics of the Dataset")

# Generate summary
desc_summary = srms.describe().T  # Transpose for better readability

# Display in Streamlit
st.dataframe(desc_summary)

# GRADE → RISK RULE (REQUIRED)
def grade_based_risk(grade):
    if grade in ['A', 'B', 'C']:
        return 1   # Not At Risk
    else:
        return 0   # At Risk

# RISK LABELING
# 0 = At Risk | 1 = Not At Risk
def risk_label(grade):
    if grade in ['A', 'B', 'C']:
        return 1
    else:
        return 0

srms['Risk'] = srms['Final_grade'].apply(risk_label)
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

for col in categorical_cols:
    le = LabelEncoder()
    srms[col] = le.fit_transform(srms[col])

# CORRELATION HEATMAP
st.subheader("Correlation Map")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(srms.corr(), annot=False, cmap='coolwarm', ax=ax)
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

report = classification_report(
    y_test, y_pred,
    target_names=["At Risk (0)", "Not At Risk (1)"]
)

st.subheader("Classification Report")
st.code(report)

# FEATURE IMPORTANCE
st.subheader("📊 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.coef_[0]
}).sort_values(by="Importance")

fig, ax = plt.subplots()
sns.barplot(
    x="Importance",
    y="Feature",
    data=importance_df,
    ax=ax
)
st.pyplot(fig)


# SIDEBAR INPUT (WITH GRADE)
# ===============================
st.sidebar.header("🔮 Student Information")

Math_score = st.sidebar.slider("Math Score", 0, 100, 70)
Science_score = st.sidebar.slider("Science Score", 0, 100, 70)
English_score = st.sidebar.slider("English Score", 0, 100, 70)
Overall_score = st.sidebar.slider("Overall Score", 0, 100, 70)
Extra_activities = st.sidebar.slider("Extra Activities", 0, 5, 2)
Study_method = st.sidebar.slider("Study Method", 0, 5, 2)

Final_grade = st.sidebar.selectbox(
    "Final Grade",
    ['A', 'B', 'C', 'D', 'E', 'F']
)

# PREDICTION
# ===============================
st.subheader("📌 Prediction Result")

if st.sidebar.button("Predict Risk"):

    # Prepare model input
    input_df = pd.DataFrame([[
        Math_score,
        Science_score,
        English_score,
        Overall_score,
        Extra_activities,
        Study_method
    ]], columns=features)

    input_scaled = scaler.transform(input_df)
    probability = model.predict_proba(input_scaled)

    # Grade-based final decision
    final_risk = grade_based_risk(Final_grade)
    confidence = probability[0][final_risk]

    if final_risk == 1:
        st.success(
            f"✅ **Not At Risk**\n\n"
            f"Grade Entered: {Final_grade}\n\n"
            f"Model Confidence: {confidence:.2f}"
        )

    else:
        st.error(
            f"⚠️ **At Risk**\n\n"
            f"Grade Entered: {Final_grade}\n\n"
            f"Model Confidence: {confidence:.2f}"
        )

        # WHY STUDENT IS AT RISK
        # ===============================
        st.markdown("### 📌 Key Risk Indicators")

        if Math_score < 50:
            st.write("- Low performance in Mathematics")

        if English_score < 50:
            st.write("- Weak English language score")

        if Overall_score < 55:
            st.write("- Poor overall academic performance")

        if Extra_activities < 1:
            st.write("- Limited participation in extracurricular activities")

    # Prepare model input
    input_df = pd.DataFrame([[
        Math_score,
        Science_score,
        English_score,
        Overall_score,
        Extra_activities,
        Study_method
    ]], columns=features)

    # Get prediction probabilities
    probabilities = model.predict_proba(input_scaled)[0]  # [At Risk, Not At Risk]

    # Display probabilities
    prob_df = pd.DataFrame({
        "Risk Status": ["At Risk", "Not At Risk"],
        "Probability": [probabilities[0], probabilities[1]]
    })
    st.table(prob_df)

