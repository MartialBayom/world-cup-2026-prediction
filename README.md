# World Cup 2026 Match Prediction with Machine Learning

> *Prédire les résultats des matchs de la Coupe du Monde 2026 grâce au Machine Learning*

[![Streamlit App](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit)](https://huggingface.co/spaces/RemiInce/fifa-wc-predict)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2)](https://mlflow.org/)
[![AWS S3](https://img.shields.io/badge/AWS-S3-FF9900?logo=amazonaws)](https://aws.amazon.com/s3/)

---

## Objectif

**Avant un match, quelle équipe a le plus de chances de gagner ?**

Ce projet répond à cette question en entraînant des modèles de Machine Learning sur **49 215 matchs internationaux** depuis 1872, avec une application dédiée à la CDM 2026 (48 équipes, 104 matchs).

 **Application en ligne :** [huggingface.co/spaces/RemiInce/fifa-wc-predict](https://huggingface.co/spaces/RemiInce/fifa-wc-predict)

---

## Fonctionnalités

### Prédicteur de match
- Choisir 2 équipes parmi les 48 qualifiées pour la CDM 2026
- Voir les statistiques comparées (rang FIFA, buts marqués/encaissés)
- Obtenir la probabilité de victoire pour chaque équipe
- Choisir le modèle ML utilisé (GradientBoosting, XGBoost, Random Forest, Logistic Regression)

### Simulation du tournoi
- Simuler la CDM 2026 de A à Z (groupes → finale)
- Voir le parcours prédit de chaque équipe
- Résultat selon notre modèle : **l'Espagne remporte la CDM 2026**

---

## Résultats

| Modèle | AUC-ROC | Accuracy |
|---|---|---|
| **GradientBoosting**  | **0.776** | **73%** |
| Logistic Regression | 0.774 | 73% |
| Random Forest | 0.755 | - |
| XGBoost | 0.752 | - |

> **Baseline sans ML** (toujours prédire le favori) : 58.5%
> Notre modèle apporte **+14 points** de précision.

---

## Structure du projet

```
world-cup-2026-prediction/
├── app/
│   └── streamlit_app.py       # Application Streamlit
├── data/
│   ├── results.csv            # 49 215 matchs internationaux (1872–2024)
│   ├── goalscorers.csv        # Buteurs par match
│   ├── shootouts.csv          # Tirs au but
│   ├── former_names.csv       # Noms actualisés des pays
│   └── fifa_ranking-2022-2026.csv  # Classement FIFA
├── models/                    # Modèles entraînés (.pkl)
├── notebooks/
│   ├── 01_data_preparation.ipynb  # Nettoyage & Feature Engineering
│   ├── 03_eda.ipynb               # Analyse exploratoire
│   └── 04_modeling_gb.ipynb       # Entraînement & évaluation
├── .env.example               # Variables d'environnement (template)
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Features utilisées pour la prédiction

| Feature | Description |
|---|---|
| `rank_dif` | Écart de classement FIFA entre les deux équipes |
| `home_goals_avg` | Moyenne de buts marqués (5 derniers matchs) |
| `away_goals_avg` | Moyenne de buts encaissés (5 derniers matchs) |
| `home_form` | Forme récente de l'équipe 1 |
| `away_form` | Forme récente de l'équipe 2 |
| `elo_diff` | Différence de score Elo entre les deux équipes |
| `is_friendly` | Match amical ou officiel |

---

## Analyse exploratoire (EDA)

- **58.5%** des matchs sont remportés par le favori FIFA
- **47%** d'upsets quand l'écart de rang est faible (0–10) vs **1%** quand il est > 100
- Corrélation faible (r = 0.223) entre points FIFA et buts marqués → justifie l'Elo rating
- Toutes les 48 équipes CDM 2026 sont représentées dans le dataset

---

## Installation

```bash
# Cloner le repo
git clone https://github.com/MartialBayom/world-cup-2026-prediction.git
cd world-cup-2026-prediction

# Installer les dépendances
pip install -r requirements.txt

# Configurer les variables d'environnement
cp .env.example .env
# Remplir .env avec vos clés AWS

# Lancer l'application
streamlit run app/streamlit_app.py
```

---

## Infrastructure

```
Kaggle (données brutes)
    ↓
Amazon S3 (stockage)
    ↓
Feature Engineering (Elo rating, forme récente, stats FIFA)
    ↓
Machine Learning (4 modèles comparés)
    ↓
MLflow (tracking des expériences sur Hugging Face)
    ↓
Streamlit (déploiement public)
```

---

## What's next ?

- [ ] **Forme récente pondérée** — les matchs récents pèsent plus
- [ ] **Optuna AutoML** — optimisation automatique des hyperparamètres
- [ ] **API FIFA Live** — données en temps réel avant chaque match
- [ ] **Prédiction du score exact** — passer de classification à régression

---

## Équipe

| | Nom | Rôle |
|---|---|
|  **Martial BAYOM** | Data Science |
|  **Rémi INCENGIERI** | Data Science |

Projet réalisé dans le cadre du **certification Jedha AI School** (RNCP Niveau 6)

---

## Sources

| Dataset | Lien |
|---|---|
| International Football Results (1872–2024) | [Kaggle — martj42](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) |
| FIFA World Ranking 2022–2026 | [Kaggle — cashncarry](https://www.kaggle.com/datasets/cashncarry/fifaworldranking) |
