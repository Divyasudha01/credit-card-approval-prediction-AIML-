"""
01_eda.py — Exploratory Data Analysis
Run this after data/generate_data.py.
Produces count plots and distribution plots saved into /plots.

(In the actual submitted project, copy these cells into a Jupyter Notebook
named 01_EDA_and_Preprocessing.ipynb — Jupyter is required by the project
instructions. The .py version here is provided so everything is runnable
end-to-end from the command line too.)
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

df = pd.read_csv('data/credit_card_approval.csv')

print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nTarget distribution:\n", df['Target_Reject'].value_counts())

# 1. Target class balance
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x='Target_Reject')
plt.title('Approved (0) vs Rejected (1)')
plt.savefig('plots/01_target_balance.png', bbox_inches='tight')
plt.close()

# 2. Count plot: Income Type vs Target
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Income_Type', hue='Target_Reject')
plt.title('Income Type vs Approval Outcome')
plt.xticks(rotation=30)
plt.savefig('plots/02_income_type_vs_target.png', bbox_inches='tight')
plt.close()

# 3. Count plot: Education Level vs Target
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Education_Level', hue='Target_Reject')
plt.title('Education Level vs Approval Outcome')
plt.xticks(rotation=30)
plt.savefig('plots/03_education_vs_target.png', bbox_inches='tight')
plt.close()

# 4. Distribution plot: Annual Income
plt.figure(figsize=(7, 5))
sns.histplot(df['Annual_Income'].dropna(), kde=True, bins=40)
plt.title('Distribution of Annual Income')
plt.savefig('plots/04_income_distribution.png', bbox_inches='tight')
plt.close()

# 5. Distribution plot: Age
plt.figure(figsize=(7, 5))
sns.histplot(df['Age_Years'], kde=True, bins=30)
plt.title('Distribution of Applicant Age')
plt.savefig('plots/05_age_distribution.png', bbox_inches='tight')
plt.close()

# 6. Correlation heatmap (numeric features only)
plt.figure(figsize=(9, 7))
numeric_df = df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('plots/06_correlation_heatmap.png', bbox_inches='tight')
plt.close()

print("\nEDA complete. Plots saved to /plots")
