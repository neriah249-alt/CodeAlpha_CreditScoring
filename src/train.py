"""
Module d'entraînement pour le Credit Scoring Model.
Entraîne Logistic Regression, Decision Tree et Random Forest.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle


def train_logistic_regression(X_train, y_train, **kwargs):
    """Entraîne une Régression Logistique."""
    model = LogisticRegression(max_iter=1000, random_state=42, **kwargs)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, **kwargs):
    """Entraîne un Arbre de Décision."""
    model = DecisionTreeClassifier(max_depth=10, random_state=42, **kwargs)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, **kwargs):
    """Entraîne une Forêt Aléatoire."""
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, **kwargs)
    model.fit(X_train, y_train)
    return model


def train_all_models(X_train_scaled, X_train, y_train):
    """Entraîne les 3 modèles et les retourne."""
    print("🔄 Entraînement des modèles...")
    
    models = {
        'Logistic Regression': train_logistic_regression(X_train_scaled, y_train),
        'Decision Tree': train_decision_tree(X_train, y_train),
        'Random Forest': train_random_forest(X_train, y_train)
    }
    
    print("✅ Tous les modèles entraînés!")
    return models


def save_models(models, filepath_prefix='models/'):
    """Sauvegarde les modèles entraînés."""
    for name, model in models.items():
        filename = name.lower().replace(' ', '_') + '.pkl'
        with open(f'{filepath_prefix}{filename}', 'wb') as f:
            pickle.dump(model, f)
    print("✅ Modèles sauvegardés!")


if __name__ == '__main__':
    from preprocess import load_data, clean_data, encode_categorical, split_and_scale
    
    df = load_data()
    df = clean_data(df)
    df, encoders = encode_categorical(df)
    X_train_s, X_test_s, y_train, y_test, scaler, X_train, X_test = split_and_scale(df)
    
    models = train_all_models(X_train_s, X_train, y_train)
    save_models(models)