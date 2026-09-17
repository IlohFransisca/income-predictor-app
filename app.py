import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import joblib
import os

# ============================================================
# PAGE SETTINGS
# ============================================================
st.set_page_config(
    page_title="Income Level Predictor",
    page_icon="💰",
    layout="wide" # Use wide layout for a better dashboard feel
)

# ============================================================
# SIDEBAR - ABOUT & CONTACT
# ============================================================
st.sidebar.title("💰 Income Predictor")

st.sidebar.header("👩‍🔬 About the Author")
st.sidebar.write("**Iloh Fransisca Onyinyechukwu**")
st.sidebar.write("Data Scientist Enthusiast")
st.sidebar.write(
    "Passionate about using machine learning to uncover insights "
    "using real world dataset and build accessible tools for everyone."
)

st.sidebar.markdown("---")

st.sidebar.header("📱 About the App")
st.sidebar.write(
    "This app uses a **CatBoost Classifier** trained on the U.S. Census "
    "Income dataset (48,842 records). It predicts whether an individual "
    "earns more than $50K per year based on demographic and employment data."
)

st.sidebar.markdown("---")

st.sidebar.header("📧 Contact")
st.sidebar.write("For collaborations or feedback:")
st.sidebar.markdown("[Send an Email](mailto:ilohfransisca2014@gmail.com)")
st.sidebar.write("ilohfransisca2014@gmail.com")

# ============================================================
# LOAD MODEL & PREPROCESSING ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    # Wrap in try-except to catch missing file errors gracefully
    try:
        model = CatBoostClassifier()
        model.load_model("best_catboost_model.cbm")
        scaler = joblib.load("scaler.pkl")
        expected_cols = joblib.load("expected_columns.pkl")
        return model, scaler, expected_cols
    except Exception as e:
        st.error(f"Error loading model files: {e}. Please ensure all .pkl and .cbm files are in the same folder.")
        st.stop()

model, scaler, expected_cols = load_artifacts()

# ============================================================
# EDUCATION MAPPING
# ============================================================
education_order = [
    'Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th',
    '11th', '12th', 'HS-grad', 'Some-college',
    'Assoc-voc', 'Assoc-acdm', 'Bachelors',
    'Masters', 'Prof-school', 'Doctorate'
]
edu_mapping = {level: idx for idx, level in enumerate(education_order)}

# ============================================================
# MAIN PAGE - TITLE
# ============================================================
st.title("💰 Income Level Prediction")
st.markdown(
    "Predict whether an individual earns **more than $50K per year** based on "
    "their demographic and employment information."
)

# ============================================================
# INPUT FORM
# ============================================================
st.header("👤 Personal & Employment Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 17, 90, 30)
    education = st.selectbox("Education Level", education_order)
    education_num = st.slider("Education Number (1-16)", 1, 16, 10)
    sex = st.selectbox("Sex", ['Female', 'Male'])
    race = st.selectbox("Race", [
        'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'
    ])

with col2:
    workclass = st.selectbox("Workclass", [
        'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
        'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', 'Unknown'
    ])
    occupation = st.selectbox("Occupation", [
        'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
        'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
        'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
        'Transport-moving', 'Priv-house-serv', 'Protective-serv',
        'Armed-Forces', 'Unknown'
    ])
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)

with col3:
    marital_status = st.selectbox("Marital Status", [
        'Married-civ-spouse', 'Divorced', 'Never-married',
        'Separated', 'Widowed', 'Married-spouse-absent'
    ])
    relationship = st.selectbox("Relationship", [
        'Wife', 'Own-child', 'Husband', 'Not-in-family',
        'Other-relative', 'Unmarried'
    ])
    native_country = st.selectbox("Native Country", [
        'United-States', 'Mexico', 'Philippines', 'Germany',
        'Canada', 'India', 'Other', 'Unknown'
    ])

st.subheader("💵 Financial Information")
col4, col5 = st.columns(2)
with col4:
    capital_gain = st.number_input("Capital Gain", 0, value=0, step=100)
with col5:
    capital_loss = st.number_input("Capital Loss", 0, value=0, step=100)

# ============================================================
# PREDICTION LOGIC
# ============================================================
if st.button("Predict Income Level", type="primary"):

    # 1. Build Input DataFrame
    data = {
        'age': [age],
        'education': [education],
        'education-num': [education_num],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week],
        'workclass': [workclass],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'sex': [sex],
        'native-country': [native_country]
    }
    df_input = pd.DataFrame(data)

    # 2. Feature Engineering (Must match training)
    df_input['education_level'] = df_input['education'].map(edu_mapping)
    df_input.drop(columns='education', inplace=True)
    df_input['log_capital_gain'] = np.log1p(df_input['capital-gain'])
    df_input['age_hours_interaction'] = df_input['age'] * df_input['hours-per-week']

    # 3. One-Hot Encoding
    df_encoded = pd.get_dummies(df_input, drop_first=True)

    # 4. Align with Training Columns
    for col in expected_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[expected_cols]

    # 5. Scaling
    numeric_cols = ['age', 'education-num', 'capital-gain', 'capital-loss',
                    'hours-per-week', 'education_level', 'log_capital_gain', 'age_hours_interaction']
    
    df_encoded[numeric_cols] = scaler.transform(df_encoded[numeric_cols])

    # 6. Predict
    pred = model.predict(df_encoded)[0]
    proba = model.predict_proba(df_encoded)[0]

    # 7. Display Result
    st.header("📊 Prediction Result")
    
    if pred == 1:
        st.success(f"**Income is > $50K** (Probability: {proba[1]:.1%})")
    else:
        st.info(f"**Income is ≤ $50K** (Probability: {proba[0]:.1%})")

    # 8. Details Expander
    with st.expander("See detailed probability breakdown"):
        st.write(f"Probability of earning ≤ $50K: **{proba[0]:.4f}**")
        st.write(f"Probability of earning > $50K: **{proba[1]:.4f}**")
        st.write(f"Model used: CatBoost Classifier (Accuracy: ~88%, ROC AUC: ~0.93)")

    # 9. Disclaimer
    st.warning(
        "⚠️ **Disclaimer:** This is a machine learning prediction based on historical "
        "census data. It is for educational purposes and should not be considered "
        "financial or career advice."
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "Built with Streamlit | Model: CatBoost | Dataset: UCI Census Income (48K records)"
)
