# Credit Card Fraud Detection
### End-to-end ML pipeline — ULB Dataset (284 807 transactions)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![SHAP](https://img.shields.io/badge/SHAP-interpretability-red)
![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF)

---

## Objectif

Détecter automatiquement les transactions frauduleuses dans un dataset
de 284 807 transactions bancaires réelles (2013, banques européennes).

**Défi principal :** classes extrêmement déséquilibrées — 1 fraude pour
577 transactions légitimes (0.173%). L'accuracy est une métrique inutile ici.

---

## Résultats

| Métrique | Valeur |
|---|---|
| Precision Fraude | **95.24%** |
| Recall Fraude | **81.63%** |
| F1 Fraude | **0.8791** |
| Average Precision | **0.8648** |
| Fausses alertes | **3 / 56 864** légitimes |
| Fraudes détectées | **80 / 98** réelles |

**Modèle final :** COBYLA Blending (Random Forest 72.6% + XGBoost 25.6%)  
**Seuil optimal :** 0.46 (optimisé par recherche sur F1-score)

---

## Pipeline

```
credit-card-fraud-detection/
├── notebooks/
│   ├── 01_eda.ipynb                  # Analyse exploratoire
│   ├── 02_preprocessing.ipynb        # Nettoyage & feature engineering
│   ├── 03_modeling.ipynb             # Modélisation & stacking
│   └── 04_evaluation_shap.ipynb      # Évaluation & interprétabilité
├── data/
│   └── README.md                     # Lien vers le dataset Kaggle
├── requirements.txt
└── README.md
```

---

## Approche

### 1. EDA — découvertes clés
- Distribution asymétrique des montants → `log1p(Amount)`
- Pic de fraude à **2h–3h du matin** (taux 10x supérieur à la journée)
- Features V14, V12, V10 très discriminantes (corrélation > 0.25)
- Fraudes avec médiane à 9€ — comportement de test de carte

### 2. Preprocessing
- `log1p(Amount)` + `StandardScaler` sur Amount et Hour
- Feature engineerée : `Hour = Time % 86400 // 3600`
- Split stratifié 80/20 — ratio fraude conservé dans chaque split
- SMOTE : 394 → 227 451 fraudes synthétiques sur le train uniquement

### 3. Modélisation — 7 approches comparées

| Modèle | Average Precision | F1 Fraude |
|---|---|---|
| Logistic Regression (baseline) | 0.7128 | 0.1046 |
| Random Forest | 0.8606 | 0.8391 |
| XGBoost | 0.8647 | 0.8137 |
| LightGBM | 0.1230 | 0.1931 |
| Stacking v1 (LR+RF+XGB+LGBM) | 0.8518 | 0.8619 |
| Stacking v2 (LR+RF+XGB) | 0.8575 | 0.8649 |
| **COBYLA Blending** | **0.8648** | **0.8791** |

### 4. COBYLA Blending
Au lieu d'un méta-modèle générique, COBYLA (Constrained Optimization BY
Linear Approximations) optimise directement l'Average Precision pour
trouver les poids optimaux de chaque modèle :

```
Poids optimaux :
  Random Forest        72.6%
  XGBoost              25.6%
  Logistic Regression   0.9%
  LightGBM              0.9%
```

### 5. Interprétabilité — SHAP values

**Features les plus importantes (XGBoost) :**
V14 · V4 · V12 · V11 · V10 · V8 · V7 · Amount_scaled

**Analyse transaction individuelle :**
Pour la transaction #840 (probabilité fraude : 98.54%), V14 contribue
à +5.41 à elle seule — signal dominant confirmé par le summary plot global.

**Analyse des 18 fraudes ratées :**
- V14 moyen : -2.16 (vs -8.33 pour les fraudes détectées)
- Amount quasi normal : -0.005 (vs -0.270 pour les détectées)
- Profil : fraudes sophistiquées imitant délibérément le comportement légitime

---

## Installation

```bash
git clone https://github.com/kgueye001/credit-card-fraud-detection
cd credit-card-fraud-detection
pip install -r requirements.txt
```

Dataset disponible sur
[Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
Télécharger `creditcard.csv` dans le dossier `data/`.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
imbalanced-learn
shap
scipy
```

---

## Leçons clés

1. **L'accuracy est trompeuse** sur données déséquilibrées —
   un modèle "99.83% accurate" peut être complètement inutile
2. **Optimiser directement la métrique métier** (COBYLA → AP)
   bat un méta-modèle générique
3. **Ajouter un mauvais modèle dégrade l'ensemble** —
   LightGBM sans tuning réduit l'AP du stacking
4. **SHAP permet l'audit individuel** de chaque décision —
   indispensable en contexte réglementaire bancaire
5. **Les fraudes ratées ont un profil distinct** —
   elles nécessitent des features supplémentaires
   (historique client, géolocalisation, device fingerprinting)

---


