# 💳 Loan Default Risk Prediction

A machine learning-based system that predicts the probability of loan
default using borrower and loan-related information.

The project covers the complete machine learning workflow, including
data cleaning, feature engineering, preprocessing, model training,
evaluation, and loan-risk prediction through a Streamlit application.

# 📌 Overview

Loan default is a major source of financial risk for lending
institutions. Being able to identify borrowers who may be at higher
risk of default can help support better lending decisions.

This project uses historical loan data obtained from Kaggle to train an
XGBoost classification model that estimates the probability of a loan
defaulting.

The trained model is integrated into a prediction system that provides
a risk-based lending decision.

> **Note:** This project is intended for educational and research
> purposes and should not be used as the sole basis for real-world
> lending decisions.

 🎯 Objectives

- Analyze historical loan data
- Clean and preprocess the dataset
- Identify relevant features for loan default prediction
- Create additional financial features
- Handle missing values and categorical variables
- Address class imbalance
- Train an XGBoost classification model
- Evaluate model performance using classification metrics
- Develop a simple interface for loan-risk prediction

---

## 📊 Dataset

The dataset used in this project was obtained from **Kaggle** and is
provided as a CSV file.

The original dataset is large and is therefore **not included in this
repository**.

### Target Variable

The target variable is created from the original `loan_status` column.

Loans with the following statuses are treated as defaults:

- `Charged Off`
- `Default`

Loans with the following status are treated as non-defaults:

- `Fully Paid`

Other loan statuses are excluded from the training data.

 Features Used

The final model uses the following features:

| Feature | Description |
|---|---|
| `loan_amnt` | Loan amount |
| `annual_inc` | Annual income |
| `int_rate` | Interest rate |
| `term` | Loan repayment term |
| `grade` | Loan grade |
| `dti` | Debt-to-income ratio |
| `loan_to_income` | Loan amount relative to annual income |

 Feature Engineering

A new feature called `loan_to_income` is created:

loan_to_income = loan_amnt / annual_inc