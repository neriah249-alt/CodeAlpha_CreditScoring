"""
Module d'évaluation pour le Credit Scoring Model.
Calcule Accuracy, Precision, Recall, F1-Score, ROC-AUC.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, X_test, y_test, model_name, use_scaled=False, scaler=None):
    """Évalue un modèle et retourne les métriques."""
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
    """Évalue tous les modèles."""
    results = []
    
    for name, model in models.items():
        use_scaled = (name == 'Logistic Regression')
        metrics = evaluate_model(model, X_test, y_test, name, use_scaled, scaler)
        results.append(metrics)
        
        print(f"\n📊 {name}:")
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
                    xticklabels=['Bon Crédit', 'Mauvais Crédit'],
                    yticklabels=['Bon Crédit', 'Mauvais Crédit'])
        axes[idx].set_title(f'{name}')
        axes[idx].set_ylabel('Réel')
        axes[idx].set_xlabel('Prédit')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Matrices de confusion sauvegardées: {save_path}")
    plt.show()


def plot_roc_curves(results_dict, y_test, save_path='results/roc_curves.png'):
    """Affiche les courbes ROC."""
    from sklearn.metrics import roc_curve
    
    plt.figure(figsize=(8, 6))
    for name, res in results_dict.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {res['ROC-AUC']:.3f})", linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire')
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Courbes ROC')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Courbes ROC sauvegardées: {save_path}")
    plt.show()


if __name__ == '__main__':
    import pickle
    from preprocess import load_data, clean_data, encode_categorical, split_and_scale
    
    df = load_data()
    df = clean_data(df)
    df, encoders = encode_categorical(df)
    X_train_s, X_test_s, y_train, y_test, scaler, X_train, X_test = split_and_scale(df)
    
    # Load models
    models = {}
    for name in ['logistic_regression', 'decision_tree', 'random_forest']:
        with open(f'models/{name}.pkl', 'rb') as f:
            models[name.replace('_', ' ').title()] = pickle.load(f)
    
    results = evaluate_all_models(models, X_test_s, X_test, y_test, scaler)
    print("\n📋 Tableau récapitulatif:")
    print(results.to_string(index=False))"""
Module d'évaluation pour le Credit Scoring Model.
Calcule Accuracy, Precision, Recall, F1-Score, ROC-AUC.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, X_test, y_test, model_name, use_scaled=False, scaler=None):
    """Évalue un modèle et retourne les métriques."""
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
    """Évalue tous les modèles."""
    results = []
    
    for name, model in models.items():
        use_scaled = (name == 'Logistic Regression')
        metrics = evaluate_model(model, X_test, y_test, name, use_scaled, scaler)
        results.append(metrics)
        
        print(f"\n📊 {name}:")
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
                    xticklabels=['Bon Crédit', 'Mauvais Crédit'],
                    yticklabels=['Bon Crédit', 'Mauvais Crédit'])
        axes[idx].set_title(f'{name}')
        axes[idx].set_ylabel('Réel')
        axes[idx].set_xlabel('Prédit')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Matrices de confusion sauvegardées: {save_path}")
    plt.show()


def plot_roc_curves(results_dict, y_test, save_path='results/roc_curves.png'):
    """Affiche les courbes ROC."""
    from sklearn.metrics import roc_curve
    
    plt.figure(figsize=(8, 6))
    for name, res in results_dict.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {res['ROC-AUC']:.3f})", linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire')
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Courbes ROC')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Courbes ROC sauvegardées: {save_path}")
    plt.show()


if __name__ == '__main__':
    import pickle
    from preprocess import load_data, clean_data, encode_categorical, split_and_scale
    
    df = load_data()
    df = clean_data(df)
    df, encoders = encode_categorical(df)
    X_train_s, X_test_s, y_train, y_test, scaler, X_train, X_test = split_and_scale(df)
    
    # Load models
    models = {}
    for name in ['logistic_regression', 'decision_tree', 'random_forest']:
        with open(f'models/{name}.pkl', 'rb') as f:
            models[name.replace('_', ' ').title()] = pickle.load(f)
    
    results = evaluate_all_models(models, X_test_s, X_test, y_test, scaler)
    print("\n📋 Tableau récapitulatif:")
    print(results.to_string(index=False))
 