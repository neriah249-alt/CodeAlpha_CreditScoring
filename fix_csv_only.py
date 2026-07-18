import pandas as pd
import os

print("🔧 Reparation du CSV...")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
columns = [
    'status', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings', 'employment', 'installment_rate', 'personal_status', 'other_debtors',
    'residence_since', 'property', 'age', 'other_installments', 'housing',
    'num_credits', 'job', 'num_dependents', 'telephone', 'foreign_worker', 'target'
]

df = pd.read_csv(url, sep=' ', header=None, names=columns)
df['target'] = df['target'].map({1: 0, 2: 1})

print(f"Shape: {df.shape}")
print(f"NaN avant sauvegarde: {df.isnull().sum().sum()}")

os.makedirs('data', exist_ok=True)
df.to_csv('data/german_credit.csv', index=False)

# Verification
df_check = pd.read_csv('data/german_credit.csv')
print(f"NaN apres sauvegarde: {df_check.isnull().sum().sum()}")
print("✅ CSV repare!")