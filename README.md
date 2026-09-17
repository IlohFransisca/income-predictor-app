# Income Level Prediction App

A machine learning web application that predicts whether an individual earns more than $50K per year based on U.S. census data. The app uses a **CatBoost Classifier** trained on 48,842 records with 15 demographic and economic features.

## Live Demo
**[Click here to view the live app](https://income-predictor-app-hzt2hzylvcn8qf5wb8hgrr.streamlit.app/)

---

## 👩‍🔬 About the Author
**Iloh Fransisca Onyinyechukwu**  
*Data Scientist Enthusiast*  
Passionate about using data to solve real-world problems.

- **Email:** [ilohfransisca2014@gmail.com](mailto:ilohfransisca2014@gmail.com)
- **GitHub:** [ilohfransisca](https://github.com/ilohfransisca)

---

## About the Dataset
The model was trained on the **UCI Census Income Dataset** (often referred to as the Adult dataset), containing 48,842 rows and 15 columns:

**Numerical Features:**
- `age`: Age of the individual
- `fnlwgt`: Final weight (dropped during preprocessing)
- `education-num`: Number of years of education
- `capital-gain`: Capital gains
- `capital-loss`: Capital losses
- `hours-per-week`: Hours worked per week

**Categorical Features:**
- `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`

**Target Variable:**
- `income`: `<=50K` (0) or `>50K` (1)

---

## Project Methodology

### 1. Data Cleaning
- **Missing Values:** Filled missing values in `workclass`, `occupation`, and `native-country` with `'Unknown'`.
- **Target Encoding:** Mapped `<=50K` to 0 and `>50K` to 1.
- **Drop Irrelevant:** Removed `fnlwgt` as it is a sampling weight and not a predictor.

### 2. Exploratory Data Analysis (EDA)
- **Cramér's V Analysis:** Revealed that `relationship` (0.45), `marital-status` (0.45), and `education` (0.37) are the strongest categorical predictors of income.
- **Class Imbalance:** The dataset is imbalanced (75% earn ≤50K, 25% earn >50K). This was addressed using stratified splitting.
- **Visualizations:** Generated count plots, grouped bar plots, and correlation heatmaps to understand income distribution across demographics.

### 3. Feature Engineering
- **Ordinal Encoding:** `education` was mapped into a numerical `education_level` (1-16) based on the natural progression of degrees.
- **Log Transformation:** Created `log_capital_gain` to handle the heavy right-skewness of capital gains.
- **Interaction Feature:** Created `age_hours_interaction` to capture the combined effect of age and working hours.
- **One-Hot Encoding:** Nominal categorical variables were one-hot encoded (`drop_first=True`) to avoid multicollinearity.
- **Scaling:** Used `StandardScaler` on all numerical features.

### 4. Model Development & Evaluation
Multiple models were trained and evaluated:

| Model | Accuracy | ROC AUC |
| :--- | :--- | :--- |
| Logistic Regression | 85% | 0.91 |
| Random Forest | 85% | 0.90 |
| XGBoost | 88% | 0.93 |
| LightGBM | 87% | 0.93 |
| **CatBoost (Best)** | **88%** | **0.93** |
| SVM | 86% | 0.89 |

**Best Model:** **CatBoost** was selected after hyperparameter tuning (`depth=6`, `iterations=200`, `l2_leaf_reg=1`, `learning_rate=0.1`), achieving the highest ROC AUC of **0.93**.

---

## App Features

1. **Interactive Input Form:** Users can input demographic, employment, and financial details.
2. **Real-Time Prediction:** Uses the trained CatBoost model to predict income level instantly.
3. **Probability Score:** Displays the confidence level of the prediction (e.g., "Probability of earning > $50K: 78.5%").
4. **Sidebar Insights:** Includes author information, app description, and contact details.
5. **Model Transparency:** An expander shows the detailed probability breakdown.

---

## Limitations & Disclaimer
Historical Bias: The model is trained on historical U.S. census data. It may reflect historical biases related to race, sex, and socioeconomic status. The predictions should not be used for discriminatory purposes.

Accuracy: The model achieves ~88% accuracy but is not perfect. It should be used as a supplementary tool, not a definitive guide.

Disclaimer: This app is a machine learning prediction, not financial or career advice. Always consult with professionals for career decisions.

## Acknowledgements
Dataset Source: UCI Machine Learning Repository - Adult Dataset

Built with: Streamlit, CatBoost, Scikit-Learn, Pandas, NumPy

## 📁 Repository Structure

```text
income-prediction-app/
├── .devcontainer/             # Dev container configuration
├── app.py                     # Main Streamlit application
├── best_catboost_model.cbm    # Trained CatBoost model
├── expected_columns.pkl       # List of columns after one-hot encoding
├── income_data.csv            # Raw UCI Census Income dataset
├── money_image.png            # Header image for the app
├── scaler.pkl                 # Fitted StandardScaler
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
