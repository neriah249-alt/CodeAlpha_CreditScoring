# 🏦 Credit Scoring Model - CodeAlpha

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25-FF4B4B.svg)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2-orange.svg)](https://scikit-learn.org)

&gt; 🔮 **Prédiction de solvabilité bancaire** avec Machine Learning — Projet de stage CodeAlpha

---

## 🚀 Démo en ligne

🌐 **Tester l'application maintenant** : [https://codealphacreditscoring-9mvbktzdwwdhpvbfztar2u.streamlit.app/](https://codealphacreditscoring-9mvbktzdwwdhpvbfztar2u.streamlit.app/)

---

## 🎯 Objectif

Ce projet implémente un **modèle de scoring crédit** qui évalue la solvabilité d'un demandeur de prêt à partir de ses données financières et personnelles. Trois algorithmes de classification sont comparés et combinés avec des règles métier pour une décision fiable.

---

## 📊 Dataset

| Caractéristique | Valeur |
|----------------|--------|
| **Source** | UCI Machine Learning Repository |
| **Nom** | German Credit Dataset |
| **Instances** | 1 000 |
| **Features** | 20 (situation financière, personnelle, résidentielle) |
| **Cible** | Bon crédit (0) / Mauvais crédit (1) |

---

## 🗂️ Structure du projet
CodeAlpha_CreditScoring/
├── 📁 data/
│   └── german_credit.csv              # Dataset
├── 📁 models/
│   ├── random_forest.pkl              # Modèle principal
│   ├── logistic_regression.pkl        # Modèle de référence
│   ├── decision_tree.pkl              # Modèle interprétable
│   ├── scaler.pkl                     # Normalisation
│   └── label_encoders.pkl             # Encodage catégoriel
├── 📁 notebooks/
│   └── eda.ipynb                      # Analyse exploratoire
├── 📁 results/
│   ├── confusion_matrices.png         # Matrices de confusion
│   ├── roc_curves.png                 # Courbes ROC
│   └── model_evaluation.png           # Comparaison des modèles
├── 📁 src/
│   ├── preprocess.py                  # Prétraitement des données
│   ├── train.py                       # Entraînement des modèles
│   ├── evaluate.py                    # Évaluation et métriques
│   └── predict.py                     # Prédiction individuelle
├── 📄 main.py                         # Pipeline complet
├── 📄 app.py                          # Interface Streamlit
├── 📄 requirements.txt                # Dépendances
└── 📄 README.md                       # Documentation

---

## 🛠️ Technologies utilisées

| Domaine | Outils |
|---------|--------|
| **Langage** | Python 3.11 |
| **ML / Data** | Pandas, NumPy, Scikit-learn |
| **Visualisation** | Matplotlib, Seaborn |
| **Interface web** | Streamlit |
| **Notebook** | Jupyter |

---

## ⚡ Installation rapide

```bash
# 1. Cloner le repository
git clone https://github.com/neriah249-alt/CodeAlpha_CreditScoring.git
cd CodeAlpha_CreditScoring

# 2. Créer un environnement virtuel
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt


🎮 Utilisation
Option 1 : Pipeline complet (ligne de commande)
bash
python main.py

Sortie attendue :
plain
============================================================
CREDIT SCORING MODEL - PIPELINE COMPLET
============================================================
Etape 1: Chargement des donnees...
Shape initial: (1000, 21)
Valeurs manquantes: 0

Etape 3: Entrainement des modeles...
Tous les modeles entraines!

Etape 4: Evaluation des modeles...
Logistic Regression:
   Accuracy:  0.7650
   Precision: 0.6275
   Recall:    0.5333
   F1-Score:  0.5766
   ROC-AUC:   0.7905
...
Pipeline termine avec succes!

Option 2 : Interface web interactive
bash
streamlit run app.py
Puis ouvrir : http://localhost:8501

Fonctionnalités :
🌍 Devises multiples : EUR, USD, FCFA, GBP, NGN, ZAR, INR, CNY
📊 3 modèles sélectionnables (Random Forest, Logistic Regression, Decision Tree)
⚡ Alertes en temps réel sur le profil du demandeur
🎯 Décision finale avec recommandation du conseiller financier

Option 3 : Prédiction par code

from src.predict import load_model, load_preprocessors, predict_credit_risk

# Charger le modèle
model = load_model('models/random_forest.pkl')
scaler, encoders = load_preprocessors()

# Profil client
client = {
    'status': 'A13',              # Compte positif
    'duration': 24,
    'credit_history': 'A32',      # Bon historique
    'purpose': 'A42',
    'credit_amount': 5000,
    'savings': 'A63',             # Épargne élevée
    'employment': 'A74',          # Emploi stable
    'installment_rate': 2,
    'personal_status': 'A92',
    'other_debtors': 'A101',
    'residence_since': 4,
    'property': 'A122',
    'age': 45,
    'other_installments': 'A143',
    'housing': 'A152',
    'num_credits': 1,
    'job': 'A174',                # Cadre
    'num_dependents': 1,
    'telephone': 'A192',
    'foreign_worker': 'A201'
}

# Prédire
result = predict_credit_risk(client, model, scaler, encoders)
print(f"Décision : {result['risk_label']}")
print(f"Confiance : {result['confidence']:.1%}")

📈 Résultats des modèles
| Modèle              | Accuracy  | Precision | Recall    | F1-Score  | **ROC-AUC** |
| ------------------- | --------- | --------- | --------- | --------- | ----------- |
| Logistic Regression | 76.5%     | 62.8%     | 53.3%     | 57.7%     | 79.1%       |
| Decision Tree       | 70.5%     | 50.9%     | 48.3%     | 49.6%     | 61.7%       |
| **Random Forest** ⭐ | **76.0%** | **65.0%** | **43.3%** | **52.0%** | **79.6%**   |
⭐ Meilleur modèle : Random Forest (meilleur ROC-AUC et Precision)

🔍 Fonctionnalités clés
| Feature                 | Description                                                   |
| ----------------------- | ------------------------------------------------------------- |
| **Feature Engineering** | Encodage intelligent des variables catégorielles              |
| **Normalisation**       | StandardScaler pour la Régression Logistique                  |
| **Équilibrage**         | Règles métier pour éviter les faux positifs                   |
| **Visualisations**      | Matrices de confusion, courbes ROC, comparaison des métriques |
| **Multi-devises**       | Conversion automatique EUR ↔ locale                           |
| **Alertes temps réel**  | Détection des profils à risque                                |

🧪 Tests effectués
| Type de profil | Critères                                    | Résultat  |
| -------------- | ------------------------------------------- | --------- |
| ✅ **Sûr**      | Compte positif, Cadre, Stable, Épargne      | APPROUVÉ  |
| ❌ **Risqué**   | Découvert, Sans emploi, Historique critique | REFUSÉ    |
| ⚠️ **Moyen**   | Quelques alertes                            | À étudier |

🎓 Apprentissages
Ce projet m'a permis de maîtriser :
✅ Le prétraitement de données structurées
✅ L'entraînement et la comparaison de modèles de classification
✅ L'évaluation avec des métriques adaptées (Precision, Recall, F1, ROC-AUC)
✅ Le déploiement d'une application ML avec Streamlit
✅ L'intégration de règles métier pour des décisions fiables


📝 Auteur
Nom : OLAFA Maurica Nériah Mondjissiola
Stage : CodeAlpha Machine Learning Internship
LinkedIn : Mauricia Olafa
Date : 20 Juillet 2026
🙏 Remerciements
Merci à CodeAlpha pour cette opportunité de stage et l'accompagnement tout au long du projet.

<div align="center">
  <p>🎓 Projet réalisé dans le cadre du stage Machine Learning — CodeAlpha</p>
</div>
```

