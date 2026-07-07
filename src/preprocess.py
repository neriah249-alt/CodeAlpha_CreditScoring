
---

## 📄 `src/preprocess.py`

```python
"""
Module de prétraitement pour le Credit Scoring Model.
Gère le chargement, le nettoyage, l'encodage et la normalisation des données.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle


def load_data(filepath='data/german_credit.csv'):
    """Charge le dataset German Credit."""
    df = pd.read_csv(filepath)
    # Convert target: 1->0 (good), 2->1 (bad)
    df['target'] = df['target'].map({1: 0, 2: 1})
    return df


def clean_data(df):
    """Nettoie les données (vérifie les valeurs manquantes, doublons)."""
    print(f"Shape initial: {df.shape}")
    print(f"Valeurs manquantes: {df.isnull().sum().sum()}")
    print(f"Doublons: {df.duplicated().sum()}")
    
    df = df.drop_duplicates()
    return df


def encode_categorical(df, fit=True, encoders=None):
    """Encode les variables catégorielles avec LabelEncoder."""
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if fit:
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        return df, encoders
    else:
        for col in categorical_cols:
            if col in encoders:
                df[col] = encoders[col].transform(df[col])
        return df


def split_and_scale(df, target_col='target', test_size=0.2, random_state=42, fit=True, scaler=None):
    """Sépare les features/target et normalise."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    if fit:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test
    else:
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test


def get_feature_names():
    """Retourne les noms des features."""
    return [
        'status', 'duration', 'credit_history', 'purpose', 'credit_amount',
        'savings', 'employment', 'installment_rate', 'personal_status', 
        'other_debtors', 'residence_since', 'property', 'age', 
        'other_installments', 'housing', 'num_credits', 'job', 
        'num_dependents', 'telephone', 'foreign_worker'
    ]


if __name__ == '__main__':
    # Test du pipeline
    df = load_data()
    df = clean_data(df)
    df, encoders = encode_categorical(df)
    X_train_s, X_test_s, y_train, y_test, scaler, X_train, X_test = split_and_scale(df)
    print(f"\nTrain: {X_train_s.shape}, Test: {X_test_s.shape}")