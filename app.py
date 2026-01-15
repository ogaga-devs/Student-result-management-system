
import pandas as pd
import numpy as np
import streamlit as st
!pip install scikit-learn
import sklearn


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("Student_Performance.csv")
df.drop(columns=["student_id"], inplace=True)

# Function to convert grades to pass/fail
def grade_to_pass_fail(grade):
    if grade.upper() in ['A', 'B', 'C']:
        return 1
    else:
        return 0

df["pass_fail"] = df["final_grade"].apply(grade_to_pass_fail)

# Features and target
X = df.drop(columns=["final_grade", "pass_fail"])
y = df["pass_fail"]

# Identify categorical and numerical columns
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()

# Preprocessing pipelines
num_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
cat_pipeline = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, numerical_features),
        ("cat", cat_pipeline, categorical_features)
    ]
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

# Train model
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
st.write(f"Model Accuracy: {accuracy*100:.2f}%")

# Streamlit UI
st.title("Student Pass/Fail Predictor")

st.sidebar.header("Enter Student Details")

def user_input_features():
    data = {}
    for col in numerical_features:
        data[col] = st.sidebar.number_input(f"{col}", min_value=0.0, value=0.0)
    for col in categorical_features:
        options = df[col].unique().tolist()
        data[col] = st.sidebar.selectbox(f"{col}", options)
    return pd.DataFrame([data])

input_df = user_input_features()

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction")
st.write("Pass" if prediction[0] == 1 else "Fail")

st.subheader("Prediction Probability")
st.write(f"Pass Probability: {prediction_proba[0][1]*100:.2f}%")
st.write(f"Fail Probability: {prediction_proba[0][0]*100:.2f}%")
