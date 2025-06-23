import streamlit as st
import pandas as pd
import numpy as np
import os
from catboost import CatBoostClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Income Prediction App", layout="centered")

# Load model and preprocessing tools
model = CatBoostClassifier()
model.load_model("best_catboost_model.cbm")
scaler = joblib.load("scaler.pkl")
expected_cols = joblib.load("expected_columns.pkl")

# Sidebar Navigation
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

    if os.path.exists("money_image.png"):
        st.image("money_image.png", caption="Predicting Income Levels", use_container_width=True)
    else:
        st.warning("📸 Image not found. Upload `money_image.png` to this folder or your GitHub repo.")

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

    # Predict
    if st.button("Predict Income Level"):
        pred = model.predict(df_encoded)[0]
        if pred == 1:
            st.success("✅ Prediction: Income is **> $50K**")
        else:
            st.info("📉 Prediction: Income is **≤ $50K**")

# 📊 Dashboard Page
elif menu == "📊 Dashboard":
    st.title("📊 Dashboard")
    st.subheader("Welcome to Income Prediction App")

    if os.path.exists("money_image.png"):
        st.image("money_image.png", caption="Income Prediction Visual", use_container_width=True)

    if os.path.exists("income_data.csv"):
        df_dash = pd.read_csv("income_data.csv")

        st.markdown("### 1. Income Distribution")
        fig1, ax1 = plt.subplots()
        income_counts = df_dash['income'].value_counts()
        bars = ax1.bar(income_counts.index.astype(str), income_counts.values, color='skyblue')
        ax1.set_ylabel("Count")
        ax1.set_title("Income Distribution")
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        st.pyplot(fig1)

        st.markdown("### 2. Education Level vs Income")
        edu_income = pd.crosstab(df_dash['education'], df_dash['income'])
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        edu_income.plot(kind='bar', stacked=True, ax=ax2, colormap="viridis")
        ax2.set_ylabel("Count")
        ax2.set_title("Education vs Income")
        st.pyplot(fig2)

        st.markdown("### 3. Workclass vs Income")
        work_income = pd.crosstab(df_dash['workclass'], df_dash['income'])
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        work_income.plot(kind='bar', stacked=True, ax=ax3, colormap="plasma")
        ax3.set_ylabel("Count")
        ax3.set_title("Workclass vs Income")
        st.pyplot(fig3)

        st.markdown("### 4. Relationship vs Income")
        rel_income = pd.crosstab(df_dash['relationship'], df_dash['income'])
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        rel_income.plot(kind='bar', stacked=True, ax=ax4, colormap="Set2")
        ax4.set_ylabel("Count")
        ax4.set_title("Relationship vs Income")
        st.pyplot(fig4)

        st.markdown("### 5. Sex vs Income")
        sex_income = pd.crosstab(df_dash['sex'], df_dash['income'])
        fig5, ax5 = plt.subplots()
        sex_income.plot(kind='bar', stacked=True, ax=ax5, colormap="coolwarm")
        ax5.set_ylabel("Count")
        ax5.set_title("Sex vs Income")
        for p in ax5.patches:
            ax5.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 3),
                         textcoords='offset points')
        st.pyplot(fig5)

    else:
        st.warning("📂 Dashboard data not found. Please upload `income_data.csv` to display analysis graphs.")

# ℹ️ About the App
elif menu == "ℹ️ About the App":
    st.title("ℹ️ About This App")
    st.write("""
    This is a machine learning-powered app that predicts whether a person earns more than $50K per year.

    **Built with:**
    - Python, Pandas, NumPy
    - CatBoost Classifier (ML Model)
    - Streamlit (Web Interface)
    - Scikit-learn (Preprocessing)
    """)

# 👩‍💻 About the Developer
elif menu == "👩‍💻 About the Developer":
    st.title("👩‍💻 About the Developer")
    st.markdown("""
    **Name:** Iloh Fransisca  
    **Role:** Data Science Enthusiast  
    **Focus:** Predictive modeling, data visualization, and Python-based tools  
    **Goal:** Build intelligent solutions to real-world problems using data.
    """)

# ✉️ Feedback
elif menu == "✉️ Feedback":
    st.title("✉️ Feedback")
    feedback = st.text_area("💬 Share your thoughts, suggestions, or report issues:")
    if st.button("Submit Feedback"):
        st.success("✅ Thank you for your feedback! (Note: not stored in this demo.)")
