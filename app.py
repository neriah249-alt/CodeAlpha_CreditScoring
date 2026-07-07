"""
Application Streamlit pour le Credit Scoring Model.
Interface interactive de prédiction de solvabilité.
"""

import streamlit as st
import pandas as pd
import pickle
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.predict import load_model, load_preprocessors, predict_credit_risk


# Configuration de la page
st.set_page_config(
    page_title="Credit Scoring - CodeAlpha",
    page_icon="🏦",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
    }
    .result-good {
        background-color: #d4edda;
        color: #155724;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
    }
    .result-bad {
        background-color: #f8d7da;
        color: #721c24;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# En-tête
st.markdown('<p class="main-header">🏦 Credit Scoring Model</p>', unsafe_allow_html=True)
st.markdown("### Prédiction de solvabilité bancaire - CodeAlpha Internship")

# Chargement des modèles
@st.cache_resource
def get_models():
    model = load_model('models/random_forest.pkl')
    scaler, encoders = load_preprocessors()
    return model, scaler, encoders

try:
    model, scaler, encoders = get_models()
    st.sidebar.success("✅ Modèles chargés")
except:
    st.sidebar.error("❌ Erreur de chargement des modèles")
    st.stop()

# Sidebar - Sélection du modèle
st.sidebar.header("⚙️ Configuration")
model_choice = st.sidebar.selectbox(
    "Choisir le modèle",
    ["Random Forest", "Logistic Regression", "Decision Tree"]
)

# Mapping des chemins de modèles
model_paths = {
    "Random Forest": "models/random_forest.pkl",
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl"
}

if model_choice != "Random Forest":
    model = load_model(model_paths[model_choice])

use_scaled = (model_choice == "Logistic Regression")

# Formulaire de saisie
st.markdown("---")
st.subheader("📝 Informations du client")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**💰 Situation financière**")
    status = st.selectbox("Statut du compte", ["A11", "A12", "A13", "A14"])
    credit_amount = st.number_input("Montant du crédit (€)", 250, 20000, 5000)
    duration = st.slider("Durée (mois)", 4, 72, 24)
    savings = st.selectbox("Épargne", ["A61", "A62", "A63", "A64", "A65"])
    installment_rate = st.slider("Taux d'annuité (%)", 1, 4, 2)

with col2:
    st.markdown("**👤 Informations personnelles**")
    age = st.slider("Âge", 19, 75, 35)
    personal_status = st.selectbox("Statut personnel", ["A91", "A92", "A93", "A94", "A95"])
    job = st.selectbox("Emploi", ["A171", "A172", "A173", "A174"])
    employment = st.selectbox("Durée emploi", ["A71", "A72", "A73", "A74", "A75"])
    num_dependents = st.selectbox("Personnes à charge", [1, 2])

with col3:
    st.markdown("**🏠 Situation résidentielle**")
    housing = st.selectbox("Logement", ["A151", "A152", "A153"])
    residence_since = st.slider("Années résidence", 1, 4, 2)
    property = st.selectbox("Propriété", ["A121", "A122", "A123", "A124"])
    telephone = st.selectbox("Téléphone", ["A191", "A192"])
    foreign_worker = st.selectbox("Travailleur étranger", ["A201", "A202"])

# Autres champs
st.markdown("**📋 Autres informations**")
col4, col5 = st.columns(2)
with col4:
    credit_history = st.selectbox("Historique crédit", ["A30", "A31", "A32", "A33", "A34"])
    purpose = st.selectbox("But du crédit", ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49", "A410"])
with col5:
    other_debtors = st.selectbox("Autres débiteurs", ["A101", "A102", "A103"])
    other_installments = st.selectbox("Autres crédits", ["A141", "A142", "A143"])
    num_credits = st.selectbox("Nombre de crédits", [1, 2, 3, 4])

# Bouton de prédiction
st.markdown("---")
if st.button("🔮 Prédire la solvabilité", type="primary", use_container_width=True):
    
    # Création du dictionnaire d'entrée
    input_data = {
        'status': status,
        'duration': duration,
        'credit_history': credit_history,
        'purpose': purpose,
        'credit_amount': credit_amount,
        'savings': savings,
        'employment': employment,
        'installment_rate': installment_rate,
        'personal_status': personal_status,
        'other_debtors': other_debtors,
        'residence_since': residence_since,
        'property': property,
        'age': age,
        'other_installments': other_installments,
        'housing': housing,
        'num_credits': num_credits,
        'job': job,
        'num_dependents': num_dependents,
        'telephone': telephone,
        'foreign_worker': foreign_worker
    }
    
    # Prédiction
    result = predict_credit_risk(input_data, model, scaler, encoders, use_scaled)
    
    # Affichage du résultat
    st.markdown("---")
    st.subheader("📊 Résultat de l'analyse")
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        if result['prediction'] == 0:
            st.markdown('<div class="result-good">✅ BON CRÉDIT</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-bad">⚠️ MAUVAIS CRÉDIT</div>', unsafe_allow_html=True)
    
    with col_res2:
        st.metric("Confiance", f"{result['confidence']:.1%}")
        st.progress(result['confidence'])
    
    # Détails des probabilités
    st.markdown("**Probabilités détaillées:**")
    prob_col1, prob_col2 = st.columns(2)
    with prob_col1:
        st.metric("Probabilité Bon Crédit", f"{result['probability_good']:.2%}")
    with prob_col2:
        st.metric("Probabilité Mauvais Crédit", f"{result['probability_bad']:.2%}")

# Pied de page
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <small>🏦 Credit Scoring Model - CodeAlpha Machine Learning Internship</small>
</div>
""", unsafe_allow_html=True)