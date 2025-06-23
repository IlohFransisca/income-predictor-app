import streamlit as st
import pandas as pd
import numpy as np
import os
from catboost import CatBoostClassifier
import joblib

st.set_page_config(page_title="Income Prediction App", layout="centered")

# Load model and preprocessing files
model = CatBoostClassifier()
model.load_model("best_catboost_model.cbm")
scaler = joblib.load("scaler.pkl")
expected_cols = joblib.load("expected_columns.pkl")

# Sidebar navigation
menu = st.sidebar.selectbox(
    "Navigate",
    ("🏠 Home", "📊 Dashboard", "ℹ️ About the App", "👩‍💻 About the Developer", "✉️ Feedback")
)

# Education mapping
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

# 🏠 Home Page
if menu == "🏠 Home":
    st.title("Income Level Prediction")

    # Show image only if it exists
    if os.path.exists("money_image.png"):
        st.image("money_image.png", caption="Predicting Income Levels", use_column_width=True)
    else:
        st.warning("📸 Image not found. Upload `money_image.png` to the app folder or GitHub repo.")

    st.markdown("Predict whether an individual earns more than $50K per year based on demographic and employment information.")

    df_input = user_input()

    # Feature Engineering
    df_input['education_level'] = df_input['education'].map(edu_mapping)
    df_input.drop(columns='education', inplace=True)
    df_input['log_capital_gain'] = np.log1p(df_input['capital-gain'])
    df_input['age_hours_interaction'] = df_input['age'] * df_input['hours-per-week']

    # One-hot encoding
    df_encoded = pd.get_dummies(df_input, drop_first=True)

    # Add missing columns
    for col in expected_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Ensure correct column order
    df_encoded = df_encoded[expected_cols]

    # Scale numeric features
    numeric_cols = ['age', 'education-num', 'capital-gain', 'capital-loss',
                    'hours-per-week', 'education_level', 'log_capital_gain', 'age_hours_interaction']
    df_encoded[numeric_cols] = scaler.transform(df_encoded[numeric_cols])

    # Predict income
    if st.button("Predict Income Level"):
        pred = model.predict(df_encoded)[0]
        if pred == 1:
            st.success("✅ Prediction: Income is **> $50K**")
        else:
            st.info("📉 Prediction: Income is **≤ $50K**")

# 📊 Dashboard
elif menu == "📊 Dashboard":
    st.title("📊 Dashboard")
    st.markdown("Coming soon: model insights, feature importance, and SHAP values.")
    st.info("This section will help you understand how predictions are made.")

# ℹ️ About the App
elif menu == "ℹ️ About the App":
    st.title("ℹ️ About This App")
    st.write("""
    This machine learning app predicts whether a person earns more than $50,000 per year.

    **Technologies used**:
    - Python, Pandas, NumPy
    - CatBoost (for classification)
    - Streamlit (for UI)
    - Scikit-learn (for preprocessing)
    """)

# 👩‍💻 About the Developer
elif menu == "👩‍💻 About the Developer":
    st.title("👩‍💻 About the Developer")
    st.markdown("""
    **Name:** Iloh Fransisca  
    **Role:** Data Science Enthusiast  
    **Focus:** Building real-world ML solutions using Python  
    **Goal:** Help people make data-informed decisions.
    """)

# ✉️ Feedback
elif menu == "✉️ Feedback":
    st.title("✉️ Feedback")
    feedback = st.text_area("Share your feedback, suggestions, or improvements:")
    if st.button("Submit Feedback"):
        st.success("✅ Thank you for your feedback! (Note: This is a demo, feedback not stored.)")
