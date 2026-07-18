"""
Module de prediction pour le Credit Scoring Model.
"""

import pandas as pd
import numpy as np
import pickle


def load_model(model_path='models/random_forest.pkl'):
    """Charge un modele sauvegarde."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_preprocessors(scaler_path='models/scaler.pkl', encoders_path='models/label_encoders.pkl'):
    """Charge les preprocesseurs."""
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
    return scaler, encoders


def predict_credit_risk(input_data, model, scaler, encoders, use_scaled=False):
    """
    Predic le risque de credit pour de nouvelles donnees.
    
    Args:
        input_data: dict ou DataFrame avec les 20 features exactes
        model: modele entraine
        scaler: StandardScaler
        encoders: dict de LabelEncoders
        use_scaled: True pour Logistic Regression
    
    Returns:
        dict avec prediction et probabilite
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
        'risk_label': 'Mauvais Credit' if prediction == 1 else 'Bon Credit',
        'probability_good': float(probability[0]),
        'probability_bad': float(probability[1]),
        'confidence': float(max(probability))
    }


if __name__ == '__main__':
    model = load_model('models/random_forest.pkl')
    scaler, encoders = load_preprocessors()
    
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
    print("\nResultat de la prediction:")
    print(f"   Risque: {result['risk_label']}")
    print(f"   Probabilite bon credit: {result['probability_good']:.2%}")
    print(f"   Probabilite mauvais credit: {result['probability_bad']:.2%}")
    print(f"   Confiance: {result['confidence']:.2%}")