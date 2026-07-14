"""
generate_data.py
-----------------
Generates a realistic synthetic Credit Card Approval dataset that mirrors
the well-known Kaggle "Credit Card Approval Prediction" dataset schema
(application_record + credit_record merged & feature engineered).

If you have internet access, you can instead download the real dataset from:
https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction
and place application_record.csv + credit_record.csv in this /data folder.
The merge logic is explained in notebooks/01_EDA_and_Preprocessing.ipynb.
Everything downstream (preprocessing, training, Flask app) works the same
either way, as long as the final CSV has the columns generated below.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 8000

genders = np.random.choice(['M', 'F'], N, p=[0.45, 0.55])
income_type = np.random.choice(
    ['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student'],
    N, p=[0.51, 0.23, 0.17, 0.08, 0.01]
)
education = np.random.choice(
    ['Secondary', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree'],
    N, p=[0.65, 0.24, 0.06, 0.03, 0.02]
)
family_status = np.random.choice(
    ['Married', 'Single', 'Civil marriage', 'Separated', 'Widow'],
    N, p=[0.6, 0.2, 0.08, 0.08, 0.04]
)
housing_type = np.random.choice(
    ['House / apartment', 'With parents', 'Rented apartment', 'Municipal apartment', 'Co-op apartment'],
    N, p=[0.85, 0.06, 0.04, 0.03, 0.02]
)
own_car = np.random.choice(['Y', 'N'], N, p=[0.4, 0.6])
own_realty = np.random.choice(['Y', 'N'], N, p=[0.65, 0.35])
children_count = np.random.poisson(0.5, N).clip(0, 5)
family_members = (children_count + np.random.choice([1, 2], N, p=[0.4, 0.6])).clip(1, 8)

annual_income = np.round(np.random.lognormal(mean=11.2, sigma=0.5, size=N), -2).clip(20000, 500000)

employment_years = np.where(
    income_type == 'Pensioner', 0,
    np.random.exponential(scale=6, size=N).clip(0, 40)
)
age_years = np.random.normal(40, 11, N).clip(21, 70)

has_work_phone = np.random.choice([0, 1], N, p=[0.7, 0.3])
has_phone = np.random.choice([0, 1], N, p=[0.3, 0.7])
has_email = np.random.choice([0, 1], N, p=[0.6, 0.4])

overdue_months = np.random.poisson(0.8, N)

risk_score = (
    -0.00002 * annual_income
    + 0.35 * overdue_months
    - 0.05 * employment_years
    + 0.02 * (age_years < 25).astype(int) * 5
    + 0.4 * (income_type == 'Student').astype(int)
    + 0.3 * (income_type == 'Pensioner').astype(int) * (employment_years == 0)
    - 0.3 * (education == 'Higher education').astype(int)
    - 0.2 * (education == 'Academic degree').astype(int)
    + 0.15 * (own_realty == 'N').astype(int)
    + np.random.normal(0, 0.6, N)
)

# 1 = Rejected, 0 = Approved
target_reject = (risk_score > np.percentile(risk_score, 78)).astype(int)

df = pd.DataFrame({
    'Gender': genders,
    'Own_Car': own_car,
    'Own_Realty': own_realty,
    'Children_Count': children_count,
    'Annual_Income': annual_income,
    'Income_Type': income_type,
    'Education_Level': education,
    'Family_Status': family_status,
    'Housing_Type': housing_type,
    'Age_Years': np.round(age_years, 1),
    'Employment_Years': np.round(employment_years, 1),
    'Has_Work_Phone': has_work_phone,
    'Has_Phone': has_phone,
    'Has_Email': has_email,
    'Family_Members': family_members,
    'Overdue_Months': overdue_months,
    'Target_Reject': target_reject
})

for col in ['Annual_Income', 'Employment_Years', 'Education_Level']:
    idx = np.random.choice(df.index, size=int(0.01 * N), replace=False)
    df.loc[idx, col] = np.nan

dupes = df.sample(50, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

df.to_csv('data/credit_card_approval.csv', index=False)
print("Dataset generated:", df.shape)
print(df['Target_Reject'].value_counts(normalize=True))
