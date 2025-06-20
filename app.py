import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import joblib

st.set_page_config(page_title="Income Prediction App", layout="centered")
st.title("Income Level Prediction")
st.markdown("Predict whether an individual earns more than $50K per year based on their demographic and employment information.")

# Load model and pre-fitted scaler
model = CatBoostClassifier()
model.load_model("best_catboost_model.cbm")
scaler = joblib.load("scaler.pkl")
expected_cols = joblib.load("expected_columns.pkl")

# Education level mapping
education_order = [
    'Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th',
    '11th', '12th', 'HS-grad', 'Some-college',
    'Assoc-voc', 'Assoc-acdm', 'Bachelors',
    'Masters', 'Prof-school', 'Doctorate'
]
edu_mapping = {level: idx for idx, level in enumerate(education_order)}

# User input form
def user_input():
    data = {
        'age': st.slider("Age", 17, 90, 30),
        'education': st.selectbox("Education", education_order),
        'education-num': st.slider("Education Number", 1, 16, 10),
        'capital-gain': st.number_input("Capital Gain", 0),
        'capital-loss': st.number_input("Capital Loss", 0),
        'hours-per-week': st.slider("Hours per week", 1, 99, 40),
        'workclass': st.selectbox("Workclass", [
            'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
            'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', 'Unknown'
        ]),
        'marital-status': st.selectbox("Marital Status", [
            'Married-civ-spouse', 'Divorced', 'Never-married',
            'Separated', 'Widowed', 'Married-spouse-absent'
        ]),
        'occupation': st.selectbox("Occupation", [
            'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
            'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
            'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
            'Transport-moving', 'Priv-house-serv', 'Protective-serv',
            'Armed-Forces', 'Unknown'
        ]),
        'relationship': st.selectbox("Relationship", [
            'Wife', 'Own-child', 'Husband', 'Not-in-family',
            'Other-relative', 'Unmarried'
        ]),
        'race': st.selectbox("Race", [
            'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'
        ]),
        'sex': st.selectbox("Sex", ['Female', 'Male']),
        'native-country': st.selectbox("Native Country", [
            'United-States', 'Mexico', 'Philippines', 'Germany',
            'Canada', 'India', 'Other', 'Unknown'
        ])
    }
    return pd.DataFrame(data, index=[0])

# Get input
df_input = user_input()

# Feature engineering
df_input['education_level'] = df_input['education'].map(edu_mapping)
df_input.drop(columns='education', inplace=True)
df_input['log_capital_gain'] = np.log1p(df_input['capital-gain'])
df_input['age_hours_interaction'] = df_input['age'] * df_input['hours-per-week']

# One-hot encoding
df_encoded = pd.get_dummies(df_input, drop_first=True)

# Add missing columns (set to 0)
for col in expected_cols:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

# Ensure column order
df_encoded = df_encoded[expected_cols]

# Scale numeric columns
numeric_cols = ['age', 'education-num', 'capital-gain', 'capital-loss',
                'hours-per-week', 'education_level', 'log_capital_gain', 'age_hours_interaction']
df_encoded[numeric_cols] = scaler.transform(df_encoded[numeric_cols])

# Predict
if st.button("Predict Income Level"):
    pred = model.predict(df_encoded)[0]
    if pred == 1:
        st.success("Prediction: Income is **> $50K**")
    else:
        st.info("Prediction: Income is **≤ $50K**")
