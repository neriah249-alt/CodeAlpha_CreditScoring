"""
Pipeline principal du Credit Scoring Model.
Exécute le prétraitement, l'entraînement et l'évaluation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocess import load_data, clean_data, encode_categorical, split_and_scale
from src.train import train_all_models, save_models
from src.evaluate import evaluate_all_models, plot_confusion_matrices, plot_roc_curves
import pickle


def main():
    print("=" * 60)
    print("🏦 CREDIT SCORING MODEL - PIPELINE COMPLET")
    print("=" * 60)
    
    # 1. Chargement et prétraitement
    print("\n📥 Étape 1: Chargement des données...")
    df = load_data('data/german_credit.csv')
    df = clean_data(df)
    df, encoders = encode_categorical(df)
    
    print("\n🔧 Étape 2: Split et normalisation...")
    X_train_s, X_test_s, y_train, y_test, scaler, X_train, X_test = split_and_scale(df)
    
    # Sauvegarde des préprocesseurs
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    # 2. Entraînement
    print("\n🤖 Étape 3: Entraînement des modèles...")
    models = train_all_models(X_train_s, X_train, y_train)
    save_models(models)
    
    # 3. Évaluation
    print("\n📊 Étape 4: Évaluation des modèles...")
    results_df = evaluate_all_models(models, X_test_s, X_test, y_test, scaler)
    
    print("\n" + "=" * 60)
    print("📋 TABLEAU RÉCAPITULATIF")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    # 4. Visualisations
    print("\n📈 Étape 5: Génération des visualisations...")
    
    # Reconstruire results_dict pour les plots
    results_dict = {}
    for name, model in models.items():
        use_scaled = (name == 'Logistic Regression')
        X_eval = X_test_s if use_scaled else X_test
        y_pred = model.predict(X_eval)
        y_prob = model.predict_proba(X_eval)[:, 1]
        results_dict[name] = {
            'y_pred': y_pred,
            'y_prob': y_prob,
            'ROC-AUC': results_df[results_df['Model'] == name]['ROC-AUC'].values[0]
        }
    
    plot_confusion_matrices(results_dict, y_test)
    plot_roc_curves(results_dict, y_test)
    
    print("\n✅ Pipeline terminé avec succès!")
    print("📁 Modèles sauvegardés dans /models")
    print("📁 Visualisations sauvegardées dans /results")


if __name__ == '__main__':
    main()