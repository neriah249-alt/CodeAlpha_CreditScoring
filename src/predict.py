"""
Module de prédiction pour le Credit Scoring Model.
Permet de prédire la solvabilité sur de nouvelles données.
"""

import pandas as pd
import numpy as np
import pickle


def load_model(model_path='models/random_forest.pkl'):
    """Charge un modèle sauvegardé."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_preprocessors(scaler_path='models/scaler.pkl', encoders_path='models/label_encoders.pkl'):
    """Charge les préprocesseurs."""
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
    return scaler, encoders


def predict_credit_risk(input_data, model, scaler, encoders, use_scaled=False):
    """
    Prédit le risque de crédit pour de nouvelles données.
    
    Args:
        input_data: dict ou DataFrame avec les features
        model: modèle entraîné
        scaler: StandardScaler
        encoders: dict de LabelEncoders
        use_scaled: True pour Logistic Regression
    
    Returns:
        dict avec prédiction et probabilité
    """
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()
    
    # Encode categorical features
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])
    
    # Scale if needed
    if use_scaled:
        df = scaler.transform(df)
    
    # Predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]
    
    return {
        'prediction': int(prediction),
        'risk_label': 'Mauvais Crédit' if prediction == 1 else 'Bon Crédit',
        'probability_good': float(probability[0]),
        'probability_bad': float(probability[1]),
        'confidence': float(max(probability))
    }


if __name__ == '__main__':
    # Exemple de prédiction
    model = load_model('models/random_forest.pkl')
    scaler, encoders = load_preprocessors()
    
    # Exemple de nouvelle demande de crédit
    new_client = {
        'status': 'A11',
        'duration': 24,
        'credit_history': 'A32',
        'purpose': 'A43',
        'credit_amount': 5000,
        'savings': 'A61',
        'employment': 'A73',
        'installment_rate': 3,
        'personal_status': 'A93',
        'other_debtors': 'A101',
        'residence_since': 3,
        'property': 'A121',
        'age': 35,
        'other_installments': 'A143',
        'housing': 'A152',
        'num_credits': 1,
        'job': 'A173',
        'num_dependents': 1,
        'telephone': 'A191',
        'foreign_worker': 'A201'
    }
    
    result = predict_credit_risk(new_client, model, scaler, encoders)
    print("\n🔮 Résultat de la prédiction:")
    print(f"   Risque: {result['risk_label']}")
    print(f"   Probabilité bon crédit: {result['probability_good']:.2%}")
    print(f"   Probabilité mauvais crédit: {result['probability_bad']:.2%}")
    print(f"   Confiance: {result['confidence']:.2%}")