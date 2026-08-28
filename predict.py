import pandas as pd
import joblib

# Load trained pipeline
pipeline = joblib.load("loan_pipeline.pkl")

# Default values
base_sample = {
    "loan_amnt": 10000,
    "annual_inc": 50000,
    "int_rate": 10.0,
    "term": 36,
    "grade": "C",
    "dti": 20.0
}

def predict_loan(data):
    sample = base_sample.copy()

    # Override with user input
    sample['loan_amnt'] = min(float(data['loan_amnt']), 40000)
    sample['annual_inc'] = max(float(data['annual_inc']), 1000)
    sample['int_rate'] = float(data['int_rate'])
    sample['term'] = int(data['term'])
    sample['grade'] = str(data['grade']).upper()
    sample['dti'] = float(data['dti'])

    # Derived feature (IMPORTANT: same as training)
    sample['loan_to_income'] = sample['loan_amnt'] / sample['annual_inc']

    # Convert to DataFrame
    df_input = pd.DataFrame([sample])

    # Prediction
    prob = pipeline.predict_proba(df_input)[:, 1][0]

    # Decision logic (realistic thresholds)
    if prob < 0.3:
        decision = "Accept"
    elif prob < 0.5:
        decision = "Waiting for review"
    else:
        decision = "Reject"

    return round(prob, 4), decision