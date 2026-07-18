"""
Application Streamlit pour le Credit Scoring Model.
Interface universelle avec devises multiples.
"""

import streamlit as st
import pandas as pd
import pickle
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.predict import load_model, load_preprocessors, predict_credit_risk


st.set_page_config(
    page_title="Credit Scoring - CodeAlpha",
    page_icon="🏦",
    layout="wide"
)

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
        border-left: 5px solid #28a745;
    }
    .result-bad {
        background-color: #f8d7da;
        color: #721c24;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        border-left: 5px solid #dc3545;
    }
    .result-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        border-left: 5px solid #ffc107;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: bold;
        color: #1f4e79;
        margin-top: 15px;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

EXCHANGE_RATES = {
    'EUR': 1.0,
    'USD': 1.09,
    'FCFA': 655.957,
    'GBP': 0.85,
    'NGN': 460.0,
    'ZAR': 20.5,
    'INR': 90.0,
    'CNY': 7.8,
}

CURRENCY_SYMBOLS = {
    'EUR': '€',
    'USD': '$',
    'FCFA': 'FCFA',
    'GBP': '£',
    'NGN': '₦',
    'ZAR': 'R',
    'INR': '₹',
    'CNY': '¥',
}

def to_local(euro_amount, rate):
    return int(euro_amount * rate)

st.markdown('<p class="main-header">🏦 Credit Scoring Model</p>', unsafe_allow_html=True)
st.markdown("### Prediction de solvabilite bancaire - CodeAlpha Internship")

st.sidebar.header("Configuration")

model_choice = st.sidebar.selectbox(
    "Choisir le modele",
    ["Random Forest", "Logistic Regression", "Decision Tree"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Devise")
currency = st.sidebar.selectbox(
    "Votre devise locale",
    list(CURRENCY_SYMBOLS.keys()),
    format_func=lambda x: f"{x} ({CURRENCY_SYMBOLS[x]})"
)

sym = CURRENCY_SYMBOLS[currency]
rate = EXCHANGE_RATES[currency]

@st.cache_resource
def get_models():
    model = load_model('models/random_forest.pkl')
    scaler, encoders = load_preprocessors()
    return model, scaler, encoders

try:
    model, scaler, encoders = get_models()
    st.sidebar.success("Modeles charges")
except Exception as e:
    st.sidebar.error(f"Erreur: {e}")
    st.stop()

model_paths = {
    "Random Forest": "models/random_forest.pkl",
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl"
}

if model_choice != "Random Forest":
    model = load_model(model_paths[model_choice])

use_scaled = (model_choice == "Logistic Regression")

st.markdown("---")
st.subheader("Informations du client")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<p class="section-title">Situation financiere</p>', unsafe_allow_html=True)
    
    status_options = {
        "Compte negatif (decouvert)": "A11",
        "Compte faible (moins de 200€)": "A12",
        "Compte positif (plus de 200€)": "A13",
        "Pas de compte bancaire": "A14"
    }
    status_label = st.selectbox("Situation bancaire", list(status_options.keys()))
    status = status_options[status_label]
    
    st.markdown('<p class="section-title">Montant du credit</p>', unsafe_allow_html=True)
    credit_amount_eur = st.number_input("Montant (€)", 250, 50000, 5000)
    credit_local = to_local(credit_amount_eur, rate)
    st.info(f"💱 En {currency}: **{credit_local:,} {sym}**")
    
    duration = st.slider("Duree de remboursement (mois)", 4, 72, 24)
    
    st.markdown('<p class="section-title">Revenu mensuel (info)</p>', unsafe_allow_html=True)
    salary_eur = st.number_input("Salaire net (€/mois)", 0, 50000, 2500)
    salary_local = to_local(salary_eur, rate)
    st.info(f"💱 En {currency}: **{salary_local:,} {sym}/mois**")
    
    savings_options = {
        "Aucune epargne": "A65",
        "Faible (moins de 130€)": "A61",
        "Moyenne (130-750€)": "A62",
        "Elevee (750-2000€)": "A63",
        "Tres elevee (plus de 2000€)": "A64"
    }
    savings_label = st.selectbox("Epargne mensuelle", list(savings_options.keys()))
    savings = savings_options[savings_label]
    
    installment_rate = st.slider("Taux d'interet mensuel (%)", 1, 4, 2)

with col2:
    st.markdown('<p class="section-title">Situation personnelle</p>', unsafe_allow_html=True)
    
    age = st.slider("Age du demandeur", 19, 75, 35)
    
    personal_options = {
        "Homme celibataire": "A93",
        "Homme marie / en couple": "A92",
        "Femme celibataire": "A95",
        "Femme mariee / en couple": "A92",
        "Divorce(e)": "A91"
    }
    personal_label = st.selectbox("Situation familiale", list(personal_options.keys()))
    personal_status = personal_options[personal_label]
    
    job_options = {
        "Sans emploi": "A171",
        "Employe non qualifie": "A172",
        "Employe qualifie": "A173",
        "Cadre / Manager / Profession liberale": "A174"
    }
    job_label = st.selectbox("Niveau professionnel", list(job_options.keys()))
    job = job_options[job_label]
    
    employment_options = {
        "Moins d'1 an (nouvel emploi)": "A71",
        "1 a 4 ans": "A72",
        "4 a 7 ans": "A73",
        "Plus de 7 ans (stable)": "A74",
        "Sans emploi": "A75"
    }
    employment_label = st.selectbox("Anciennete dans l'emploi", list(employment_options.keys()))
    employment = employment_options[employment_label]
    
    num_dependents = st.selectbox("Personnes a charge", [0, 1, 2, 3, 4, 5, 6])

with col3:
    st.markdown('<p class="section-title">Logement & Patrimoine</p>', unsafe_allow_html=True)
    
    housing_options = {
        "Proprietaire (maison ou appartement)": "A152",
        "Locataire": "A151",
        "Heberge gratuitement (famille, amis)": "A153"
    }
    housing_label = st.selectbox("Type de logement actuel", list(housing_options.keys()))
    housing = housing_options[housing_label]
    
    residence_since = st.slider("Annees dans ce logement", 1, 30, 2)
    
    property_options = {
        "Appartement en propriete": "A121",
        "Maison en propriete": "A122",
        "Location avec garantie / caution": "A123",
        "Pas de propriete immobiliere": "A124"
    }
    property_label = st.selectbox("Propriete immobiliere possedee", list(property_options.keys()))
    property = property_options[property_label]
    
    telephone_options = {
        "Avec telephone fixe (stable)": "A192",
        "Sans telephone fixe (mobile uniquement)": "A191"
    }
    telephone_label = st.selectbox("Telephone", list(telephone_options.keys()))
    telephone = telephone_options[telephone_label]
    
    foreign_options = {
        "Citoyen national / Resident permanent": "A201",
        "Travailleur etranger / Resident temporaire": "A202"
    }
    foreign_label = st.selectbox("Statut de residence", list(foreign_options.keys()))
    foreign_worker = foreign_options[foreign_label]

st.markdown('<p class="section-title">Historique et raison du pret</p>', unsafe_allow_html=True)
col4, col5 = st.columns(2)

with col4:
    history_options = {
        "Aucun credit anterieur (premier emprunt)": "A30",
        "Credits en cours, tous payes a temps": "A31",
        "Credits passes payes a temps (historique bon)": "A32",
        "Quelques retards de paiement": "A33",
        "Credits critiques / Impayes / Contentieux": "A34"
    }
    history_label = st.selectbox("Historique de credit", list(history_options.keys()))
    credit_history = history_options[history_label]
    
    purpose_options = {
        "Voiture neuve": "A40",
        "Voiture d'occasion": "A41",
        "Meubles / Equipement maison": "A42",
        "Electronique / TV / Telephone": "A43",
        "Voyages / Vacances": "A44",
        "Sante / Medical": "A45",
        "Education / Formation": "A46",
        "Business / Investissement": "A48",
        "Renovation / Travaux": "A49",
        "Autre": "A410"
    }
    purpose_label = st.selectbox("Raison du pret", list(purpose_options.keys()))
    purpose = purpose_options[purpose_label]

with col5:
    debtors_options = {
        "Aucun garant (seul demandeur)": "A101",
        "Co-demandeur (conjoint / partenaire)": "A102",
        "Garant externe (famille, ami, banque)": "A103"
    }
    debtors_label = st.selectbox("Garant(s) / Caution", list(debtors_options.keys()))
    other_debtors = debtors_options[debtors_label]
    
    installments_options = {
        "Aucun autre credit en cours": "A143",
        "Autres credits bancaires": "A141",
        "Credits magasins / a la consommation": "A142"
    }
    installments_label = st.selectbox("Autres credits en cours", list(installments_options.keys()))
    other_installments = installments_options[installments_label]
    
    num_credits = st.selectbox("Nombre de credits en cours (hors celui-ci)", [0, 1, 2, 3, 4, 5])

st.markdown("---")
if st.button("Analyser la solvabilite", type="primary", use_container_width=True):
    
    monthly_payment = credit_amount_eur / duration
    debt_ratio = (monthly_payment / salary_eur * 100) if salary_eur > 0 else 999
    
    input_data = {
        'status': status,
        'duration': duration,
        'credit_history': credit_history,
        'purpose': purpose,
        'credit_amount': credit_amount_eur,
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
        'foreign_worker': foreign_worker,
    }
    
    result = predict_credit_risk(input_data, model, scaler, encoders, use_scaled)
    
    # ============================================
    # ALERTES
    # ============================================
    alerts = []
    if debt_ratio > 50:
        alerts.append(f"Taux d'endettement tres eleve: {debt_ratio:.1f}%")
    elif debt_ratio > 33:
        alerts.append(f"Taux d'endettement eleve: {debt_ratio:.1f}%")
    
    if credit_amount_eur > salary_eur * 12:
        alerts.append("Montant du credit superieur au salaire annuel")
    
    if status == "A11":
        alerts.append("Compte bancaire a decouvert")
    
    if credit_history == "A34":
        alerts.append("Historique de credit critique")
    
    if savings == "A65":
        alerts.append("Aucune epargne de precaution")
    
    if job == "A171":
        alerts.append("Sans emploi")
    
    if employment == "A75":
        alerts.append("Sans emploi confirme")
    
    if alerts:
        st.markdown("### ⚡ Alertes detectees")
        for alert in alerts:
            st.warning(alert)
    
    # ============================================
    # PROBABILITÉS
    # ============================================
    st.markdown("---")
    st.subheader("Probabilites du modele")
    
    prob_col1, prob_col2, prob_col3 = st.columns(3)
    with prob_col1:
        st.metric("Bon Credit", f"{result['probability_good']:.2%}")
    with prob_col2:
        st.metric("Mauvais Credit", f"{result['probability_bad']:.2%}")
    with prob_col3:
        st.metric("Confiance", f"{result['confidence']:.1%}")
    
    # ============================================
    # DÉCISION FINALE ÉQUILIBRÉE
    # ============================================
    st.markdown("---")
    st.subheader("Decision finale")
    
    # Compter les alertes par niveau
    critical_alerts = [a for a in alerts if any(k in a for k in ["tres eleve", "critique"])]
    major_alerts = [a for a in alerts if any(k in a for k in ["decouvert", "Sans emploi"])]
    total_alerts = len(alerts)
    
    # LOGIQUE ÉQUILIBRÉE
    force_reject = (
        len(critical_alerts) >= 1 or           # 1 critique = refus
        (len(major_alerts) >= 2) or             # Découvert + sans emploi = refus
        (total_alerts >= 3) or                  # 3+ alertes = refus
        (result['probability_bad'] > 0.50)      # Proba > 50% = refus
    )
    
    force_approve = (
        total_alerts == 0 and                   # Aucune alerte
        result['probability_bad'] < 0.25 and    # Proba < 25%
        status in ["A13", "A14"] and            # Compte OK
        employment in ["A73", "A74"] and       # Emploi stable
        savings in ["A63", "A64"]              # Épargne élevée
    )
    
    # Décision
    if force_reject:
        is_approved = False
        final_decision = "REFUSE"
        override_reason = " (Profil a risque identifie par le systeme)"
    elif force_approve:
        is_approved = True
        final_decision = "APPROUVE"
        override_reason = " (Excellent profil confirme)"
    else:
        # Cas moyen : suivre le modèle
        is_approved = result['prediction'] == 0
        final_decision = "APPROUVE" if is_approved else "REFUSE"
        override_reason = ""
    
    # AFFICHAGE UNIQUE
    if is_approved:
        decision_text = "✅ CREDIT APPROUVE"
        conseil = f"Ce client presente un profil SUR.{override_reason} Recommandation : ACCORDER le credit."
        color = "#28a745"
        bg_color = "#d4edda"
        text_color = "#155724"
    else:
        decision_text = "❌ CREDIT REFUSE"
        conseil = f"Ce client presente un profil RISQUE.{override_reason} Recommandation : REFUSER le credit."
        color = "#dc3545"
        bg_color = "#f8d7da"
        text_color = "#721c24"
    
    st.markdown(f"""
    <div style="background-color: {bg_color}; 
                padding: 25px; 
                border-radius: 15px; 
                text-align: center; 
                border-left: 6px solid {color};
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h2 style="margin: 0; color: {text_color}; font-size: 2rem;">
            {decision_text}
        </h2>
        <p style="font-size: 1.3rem; margin-top: 15px; color: #333; font-weight: 500;">
            {conseil}
        </p>
        <p style="font-size: 1rem; color: #666; margin-top: 10px;">
            Confiance : <strong style="color: {text_color};">{result['confidence']:.1%}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ============================================
    # RECOMMANDATION
    # ============================================
    st.markdown("---")
    st.subheader("Recommandation du conseiller")
    
    if result['probability_bad'] > 0.7 or force_reject:
        reco = "Risque tres eleve. Refus recommande sans exception."
        icon = "🔴"
    elif result['probability_bad'] > 0.5:
        reco = "Risque eleve. Refus recommande. Exceptionnellement avec garant solide."
        icon = "🟠"
    elif result['probability_bad'] > 0.35:
        reco = "Risque modere. Demander garanties : co-demandeur ou caution."
        icon = "🟡"
    elif result['probability_bad'] > 0.2:
        reco = "Risque faible. Credit approuvable avec conditions standard."
        icon = "🟢"
    else:
        reco = "Excellent profil. Offrir taux preferentiel et conditions avantageuses."
        icon = "⭐"
    
    st.info(f"{icon} {reco}")
    
    # Détails du refus
    if not is_approved and alerts:
        st.markdown("---")
        st.markdown("### 📝 Motifs du refus")
        for i, alert in enumerate(alerts, 1):
            st.error(f"{i}. {alert}")
    
    # ============================================
    # RÉSUMÉ
    # ============================================
    st.markdown("---")
    st.subheader("Resume du profil")
    
    col_sum1, col_sum2, col_sum3 = st.columns(3)
    with col_sum1:
        st.metric("Salaire mensuel", f"{to_local(salary_eur, rate):,} {sym}")
    with col_sum2:
        st.metric("Montant demande", f"{to_local(credit_amount_eur, rate):,} {sym}")
    with col_sum3:
        st.metric("Taux d'endettement", f"{debt_ratio:.1f}%")
    
    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; font-size: 0.9rem; color: #555;">
        <strong>Profil :</strong> Age {age} ans | {personal_label} | {job_label} | 
        {employment_label} | {housing_label} | {history_label} | {purpose_label}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("""
    ⚖️ Avertissement : Cette analyse est fournie a titre indicatif par un modele d'IA. 
    La decision finale releve de la responsabilite de l'etablissement bancaire.
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 20px;">
    <p style="margin: 0;"><strong>🏦 Credit Scoring Model</strong></p>
    <p style="margin: 5px 0; font-size: 0.9rem;">CodeAlpha Machine Learning Internship</p>
    <p style="margin: 0; font-size: 0.8rem; color: #aaa;">
        EUR | USD | FCFA | GBP | NGN | ZAR | INR | CNY
    </p>
</div>
""", unsafe_allow_html=True)