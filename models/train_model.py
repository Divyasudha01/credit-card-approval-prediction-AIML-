"""
train_model.py
---------------
Step 4 (Preprocessing & Feature Engineering) + Step 5 (Model Building)
of the project instructions.

- Handles missing values
- Removes duplicate records
- Encodes categorical variables into numerical format
- Trains Logistic Regression, Decision Tree, Random Forest, XGBoost
- Evaluates with accuracy, confusion matrix, classification report
- Saves the best-performing model + the fitted encoders/scaler to /models
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    # Falls back to sklearn's GradientBoostingClassifier if xgboost isn't
    # installed in this environment. Run `pip install xgboost` and re-run
    # this script to use real XGBoost (recommended for the actual project
    # submission, since the assignment explicitly asks for it).
    HAS_XGBOOST = False
    print("[warning] xgboost not installed — using GradientBoostingClassifier "
          "as a stand-in. Run `pip install xgboost` for the real thing.")

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv('data/credit_card_approval.csv')

# ---------------------------------------------------------------
# 2. Data Preprocessing & Feature Engineering
# ---------------------------------------------------------------
df = df.drop_duplicates().reset_index(drop=True)

# Handle missing values: numeric -> median, categorical -> mode
numeric_cols = ['Annual_Income', 'Employment_Years']
categorical_cols = ['Gender', 'Own_Car', 'Own_Realty', 'Income_Type',
                     'Education_Level', 'Family_Status', 'Housing_Type']

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical variables into numerical format
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# "Converting payment status categories into binary labels" — already
# captured by Target_Reject (0 = Approved, 1 = Rejected).

feature_cols = [c for c in df.columns if c != 'Target_Reject']
X = df[feature_cols]
y = df['Target_Reject']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 3. Train & evaluate 4 classification models
# ---------------------------------------------------------------
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=8, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42),
}
if HAS_XGBOOST:
    models['XGBoost'] = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        use_label_encoder=False, eval_metric='logloss', random_state=42
    )
else:
    models['XGBoost (fallback: GradientBoosting)'] = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42
    )

results = {}
best_model_name, best_model, best_acc = None, None, -1

for name, model in models.items():
    # Logistic Regression benefits from scaled features; tree ensembles don't need it
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds).tolist()
    report = classification_report(y_test, preds, output_dict=True)

    results[name] = {'accuracy': acc, 'confusion_matrix': cm, 'report': report}
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))

    if acc > best_acc:
        best_acc = acc
        best_model_name = name
        best_model = model

print(f"\nBest model: {best_model_name} (accuracy={best_acc:.4f})")

# ---------------------------------------------------------------
# 4. Save best model + preprocessing artifacts
# ---------------------------------------------------------------
joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(encoders, 'models/encoders.pkl')
joblib.dump(feature_cols, 'models/feature_cols.pkl')

with open('models/results_summary.json', 'w') as f:
    json.dump(
        {k: {'accuracy': v['accuracy']} for k, v in results.items()} |
        {'best_model': best_model_name, 'uses_scaled_input': best_model_name == 'Logistic Regression'},
        f, indent=2
    )

print("\nSaved: models/best_model.pkl, scaler.pkl, encoders.pkl, feature_cols.pkl, results_summary.json")
