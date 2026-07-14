"""
app.py — Flask Web Application
Step 6 of the project instructions: Home Page, Prediction Interface,
Input Forms, Result Display.

Run locally with:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# ---------------------------------------------------------------
# Load trained model + preprocessing artifacts once at startup
# ---------------------------------------------------------------
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')
encoders = joblib.load('models/encoders.pkl')
feature_cols = joblib.load('models/feature_cols.pkl')

import json
with open('models/results_summary.json') as f:
    RESULTS = json.load(f)
BEST_MODEL_NAME = RESULTS['best_model']
USES_SCALED_INPUT = RESULTS['uses_scaled_input']

CATEGORICAL_COLS = ['Gender', 'Own_Car', 'Own_Realty', 'Income_Type',
                     'Education_Level', 'Family_Status', 'Housing_Type']


@app.route('/')
def home():
    """Home Page Introduction"""
    return render_template(
        'index.html',
        model_name=BEST_MODEL_NAME,
        accuracy=round(RESULTS[BEST_MODEL_NAME]['accuracy'] * 100, 2),
        encoders={col: list(le.classes_) for col, le in encoders.items()}
    )


@app.route('/predict', methods=['POST'])
def predict():
    """Credit Card Approval Prediction Interface"""
    form = request.form

    raw_input = {
        'Gender': form.get('Gender'),
        'Own_Car': form.get('Own_Car'),
        'Own_Realty': form.get('Own_Realty'),
        'Children_Count': float(form.get('Children_Count')),
        'Annual_Income': float(form.get('Annual_Income')),
        'Income_Type': form.get('Income_Type'),
        'Education_Level': form.get('Education_Level'),
        'Family_Status': form.get('Family_Status'),
        'Housing_Type': form.get('Housing_Type'),
        'Age_Years': float(form.get('Age_Years')),
        'Employment_Years': float(form.get('Employment_Years')),
        'Has_Work_Phone': int(form.get('Has_Work_Phone')),
        'Has_Phone': int(form.get('Has_Phone')),
        'Has_Email': int(form.get('Has_Email')),
        'Family_Members': float(form.get('Family_Members')),
        'Overdue_Months': int(form.get('Overdue_Months')),
    }

    row = {}
    for col in feature_cols:
        val = raw_input[col]
        if col in CATEGORICAL_COLS:
            le = encoders[col]
            val = le.transform([val])[0]
        row[col] = val

    X = pd.DataFrame([row])[feature_cols]

    if USES_SCALED_INPUT:
        X = scaler.transform(X)

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = round(float(max(proba)) * 100, 2)

    result = {
        'decision': 'REJECTED' if pred == 1 else 'APPROVED',
        'is_rejected': bool(pred == 1),
        'confidence': confidence,
        'model_name': BEST_MODEL_NAME,
    }

    return render_template(
        'result.html',
        result=result,
        inputs=raw_input
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
