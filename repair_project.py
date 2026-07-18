"""
Script de reparation complet - Credit Scoring Model
Repare le CSV, les fichiers .pkl, et corrige les erreurs de syntaxe
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

print("=" * 60)
print("🔧 REPARATION DU PROJET CREDIT SCORING")
print("=" * 60)

# ============================================
# 1. REPARER LE CSV
# ============================================
print("\n📥 Telechargement du dataset German Credit...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
columns = [
    'status', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings', 'employment', 'installment_rate', 'personal_status', 'other_debtors',
    'residence_since', 'property', 'age', 'other_installments', 'housing',
    'num_credits', 'job', 'num_dependents', 'telephone', 'foreign_worker', 'target'
]

df = pd.read_csv(url, sep=' ', header=None, names=columns)
df['target'] = df['target'].map({1: 0, 2: 1})

print(f"   Shape: {df.shape}")
print(f"   NaN: {df.isnull().sum().sum()}")

os.makedirs('data', exist_ok=True)
df.to_csv('data/german_credit.csv', index=False)
print("   ✅ data/german_credit.csv repare")

# ============================================
# 2. GENERER LES MODELES
# ============================================
print("\n🤖 Generation des modeles...")

X = df.drop('target', axis=1)
y = df['target']

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs('models', exist_ok=True)

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
with open('models/logistic_regression.pkl', 'wb') as f:
    pickle.dump(lr, f)
print("   ✅ models/logistic_regression.pkl")

# Decision Tree
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(X_train, y_train)
with open('models/decision_tree.pkl', 'wb') as f:
    pickle.dump(dt, f)
print("   ✅ models/decision_tree.pkl")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
with open('models/random_forest.pkl', 'wb') as f:
    pickle.dump(rf, f)
print("   ✅ models/random_forest.pkl")

# Preprocesseurs
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("   ✅ models/scaler.pkl")

with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("   ✅ models/label_encoders.pkl")

# ============================================
# 3. CORRIGER evaluate.py
# ============================================
print("\n🔧 Correction de src/evaluate.py...")

evaluate_code = '''"""
Module d'evaluation pour le Credit Scoring Model.
Calcule Accuracy, Precision, Recall, F1-Score, ROC-AUC.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, X_test, y_test, model_name, use_scaled=False, scaler=None):
    """Evalue un modele et retourne les metriques."""
    if use_scaled and scaler is not None:
        X_test = scaler.transform(X_test)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob),
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    return metrics


def evaluate_all_models(models, X_test_scaled, X_test, y_test, scaler=None):
    """Evalue tous les modeles."""
    results = []
    
    for name, model in models.items():
        use_scaled = (name == 'Logistic Regression')
        metrics = evaluate_model(model, X_test, y_test, name, use_scaled, scaler)
        results.append(metrics)
        
        print(f"\\\\n📊 {name}:")
        print(f"   Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"   Precision: {metrics['Precision']:.4f}")
        print(f"   Recall:    {metrics['Recall']:.4f}")
        print(f"   F1-Score:  {metrics['F1-Score']:.4f}")
        print(f"   ROC-AUC:   {metrics['ROC-AUC']:.4f}")
    
    return pd.DataFrame([{k: v for k, v in r.items() if k not in ['y_pred', 'y_prob']} for r in results])


def plot_confusion_matrices(results_dict, y_test, save_path='results/confusion_matrices.png'):
    """Affiche les matrices de confusion."""
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, res) in enumerate(results_dict.items()):
        cm = confusion_matrix(y_test, res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Bon Credit', 'Mauvais Credit'],
                    yticklabels=['Bon Credit', 'Mauvais Credit'])
        axes[idx].set_title(f'{name}')
        axes[idx].set_ylabel('Reel')
        axes[idx].set_xlabel('Predic')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Matrices de confusion sauvegardees: {save_path}")
    plt.show()


def plot_roc_curves(results_dict, y_test, save_path='results/roc_curves.png'):
    """Affiche les courbes ROC."""
    from sklearn.metrics import roc_curve
    
    plt.figure(figsize=(8, 6))
    for name, res in results_dict.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {res['ROC-AUC']:.3f})", linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatoire')
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Courbes ROC')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Courbes ROC sauvegardees: {save_path}")
    plt.show()


if __name__ == '__main__':
    import pickle
    from preprocess import load_data, clean_data, encode_categorical, split_and_scale
    
    df = load_data()
    df = clean_data(df)
    df, encoders = encode_categorical(df)
    X_train_s, X_test_s, y_train, y_test, scaler, X_train, X_test = split_and_scale(df)
    
    models = {}
    for name in ['logistic_regression', 'decision_tree', 'random_forest']:
        with open(f'models/{name}.pkl', 'rb') as f:
            models[name.replace('_', ' ').title()] = pickle.load(f)
    
    results = evaluate_all_models(models, X_test_s, X_test, y_test, scaler)
    print("\\\\n📋 Tableau recapitulatif:")
    print(results.to_string(index=False))
'''

with open('src/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(evaluate_code)
print("   ✅ src/evaluate.py corrige")

# ============================================
# 4. VERIFICATION FINALE
# ============================================
print("\n" + "=" * 60)
print("✅ REPARATION TERMINEE!")
print("=" * 60)
print("\nVous pouvez maintenant executer:")
print("   python main.py")
print("   streamlit run app.py")