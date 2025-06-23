import streamlit as st
import pandas as pd
import numpy as np
import os
from catboost import CatBoostClassifier
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Income Prediction App", layout="centered")

# Load model and tools
model = CatBoostClassifier()
model.load_model("best_catboost_model.cbm")
scaler = joblib.load("scaler.pkl")
expected_cols = joblib.load("expected_columns.pkl")

# Sidebar menu
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

# Form input
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

# Home Page
if menu == "🏠 Home":
    st.title("Income Level Prediction App")

    if os.path.exists("money_image.png"):
        st.image("money_image.png", caption="Predicting Income Levels", use_container_width=True)
    else:
        st.warning("⚠️ money_image.png not found. Upload it to display here.")

    st.markdown("Predict whether an individual earns more than $50K per year based on demographic and employment information.")

    df_input = user_input()

    # Feature engineering
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

    # Ensure column order
    df_encoded = df_encoded[expected_cols]

    # Scale numeric features
    numeric_cols = ['age', 'education-num', 'capital-gain', 'capital-loss',
                    'hours-per-week', 'education_level', 'log_capital_gain', 'age_hours_interaction']
    df_encoded[numeric_cols] = scaler.transform(df_encoded[numeric_cols])

    if st.button("Predict Income Level"):
        pred = model.predict(df_encoded)[0]
        if pred == 1:
            st.success("✅ Prediction: Income is **> $50K**")
        else:
            st.info("📉 Prediction: Income is **≤ $50K**")

# Dashboard Page
elif menu == "📊 Dashboard":
    st.title("📊 Dashboard")
    st.subheader("Welcome to Income Prediction App")

    if os.path.exists("money_image.png"):
        st.image("money_image.png", caption="Income Data Overview", use_container_width=True)

    if os.path.exists("income_data.csv"):
        df_dash = pd.read_csv("income_data.csv")

        def plot_bar_chart(df, feature, title, colormap="Set2"):
            ct = pd.crosstab(df[feature], df['income'])
            fig, ax = plt.subplots(figsize=(10, 5))
            ct.plot(kind='bar', stacked=False, ax=ax, colormap=colormap, width=0.7)
            ax.set_title(title)
            ax.set_ylabel("Count")
            ax.set_xlabel(feature)
            ax.legend(title="Income")
            plt.xticks(rotation=45, ha='right')

            # Add value labels
            for container in ax.containers:
                for bar in container:
                    height = bar.get_height()
                    if height > 0:
                        ax.annotate(
                            f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center',
                            va='bottom',
                            fontsize=8
                        )
            st.pyplot(fig)

        st.markdown("### 1. Income Distribution")
        fig1, ax1 = plt.subplots()
        income_counts = df_dash['income'].value_counts()
        bars = ax1.bar(income_counts.index.astype(str), income_counts.values, color='skyblue')
        ax1.set_title("Income Distribution")
        ax1.set_ylabel("Count")
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        st.pyplot(fig1)

        st.markdown("### 2. Education Level vs Income")
        plot_bar_chart(df_dash, 'education', 'Education vs Income')

        st.markdown("### 3. Workclass vs Income")
        plot_bar_chart(df_dash, 'workclass', 'Workclass vs Income')

        st.markdown("### 4. Marital Status vs Income")
        plot_bar_chart(df_dash, 'marital-status', 'Marital Status vs Income')

        st.markdown("### 5. Relationship vs Income")
        plot_bar_chart(df_dash, 'relationship', 'Relationship vs Income')

        st.markdown("### 6. Race vs Income")
        plot_bar_chart(df_dash, 'race', 'Race vs Income')

        st.markdown("### 7. Sex vs Income")
        plot_bar_chart(df_dash, 'sex', 'Sex vs Income')

        st.markdown("### 8. Top 10 Native Countries vs Income")
        top_countries = df_dash['native-country'].value_counts().head(10).index
        df_native_top = df_dash[df_dash['native-country'].isin(top_countries)]
        plot_bar_chart(df_native_top, 'native-country', 'Top Countries vs Income')

    else:
        st.warning("⚠️ `income_data.csv` not found. Upload it for dashboard analysis.")

# About the App
elif menu == "ℹ️ About the App":
    st.title("ℹ️ About This App")
    st.write("""
    A machine learning app to predict whether an individual earns more than $50K/year.

    **Built with**:
    - CatBoost Classifier
    - Streamlit
    - Pandas, Scikit-learn, Matplotlib
    """)

# About the Developer
elif menu == "👩‍💻 About the Developer":
    st.title("👩‍💻 About the Developer")
    st.write("""
    **Iloh Fransisca**  
    Data Science Enthusiast passionate about solving real-world problems through machine learning and data-driven insights.
    """)

# Feedback
elif menu == "✉️ Feedback":
    st.title("✉️ Feedback")
    feedback = st.text_area("Leave your feedback or suggestions:")
    if st.button("Submit"):
        st.success("✅ Thanks for your feedback! (Not stored in demo version)")
