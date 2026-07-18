"""
Pipeline principal du Credit Scoring Model.
Execute le pretraitement, l'entrainement et l'evaluation.
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
    print("CREDIT SCORING MODEL - PIPELINE COMPLET")
    print("=" * 60)
    
    print("\nEtape 1: Chargement des donnees...")
    df = load_data('data/german_credit.csv')
    df = clean_data(df)
    df, encoders = encode_categorical(df)
    
    print("\nEtape 2: Split et normalisation...")
    X_train_s, X_test_s, y_train, y_test, scaler, X_train, X_test = split_and_scale(df)
    
    os.makedirs('models', exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    print("\nEtape 3: Entrainement des modeles...")
    models = train_all_models(X_train_s, X_train, y_train)
    save_models(models)
    
    print("\nEtape 4: Evaluation des modeles...")
    results_df = evaluate_all_models(models, X_test_s, X_test, y_test, scaler)
    
    print("\n" + "=" * 60)
    print("TABLEAU RECAPITULATIF")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    print("\nEtape 5: Generation des visualisations...")
    os.makedirs('results', exist_ok=True)
    
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
    
    print("\nPipeline termine avec succes!")
    print("Modeles sauvegardes dans /models")
    print("Visualisations sauvegardees dans /results")


if __name__ == '__main__':
    main()