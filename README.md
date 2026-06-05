# Credit Card Fraud Detection
### End-to-end ML pipeline — ULB Dataset (284 807 transactions)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![SHAP](https://img.shields.io/badge/SHAP-interpretability-red)
![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF)
![Docker](https://img.shields.io/badge/Docker-containerized-2496ED)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![CloudRun](https://img.shields.io/badge/Google_Cloud_Run-deployed-4285F4)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E)

---

## 🚀 Demo en ligne

| | URL |
|---|---|
| **Interface Gradio** | [kgueye001-fraud-detection.hf.space](https://kgueye001-fraud-detection.hf.space) |
| **API REST** | [fraud-api-245513771842.europe-west1.run.app/docs](https://fraud-api-245513771842.europe-west1.run.app/docs) |

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
├── deployment/
│   ├── models/                       # Modèles sérialisés (.pkl)
│   ├── app.py                        # API FastAPI
│   ├── model.py                      # Logique de prédiction
│   ├── gradio_app.py                 # Interface Gradio
│   ├── Dockerfile                    # Containerisation
│   └── requirements.txt              # Dépendances production
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

## Déploiement

### Stack technique

| Composant | Technologie | Rôle |
|---|---|---|
| Sérialisation | `joblib` | Sauvegarde des 4 modèles + scalers |
| API REST | `FastAPI` + `uvicorn` | Endpoint `/predict` avec validation Pydantic |
| Containerisation | `Docker` | Image `python:3.10-slim` + `libgomp1` |
| Registry | Google Container Registry | Stockage de l'image Docker |
| Cloud | Google Cloud Run | Déploiement serverless — europe-west1 |
| Interface | `Gradio` | Interface visuelle publique |
| Hosting | Hugging Face Spaces | Démo accessible sans installation |

### Architecture

```
Utilisateur
    │
    ▼
Gradio (Hugging Face Spaces)
    │  requête HTTP POST
    ▼
FastAPI (Google Cloud Run)
    │
    ▼
FraudDetector.predict()
    ├── preprocess() — log1p + StandardScaler
    ├── LR.predict_proba()   × 0.9%
    ├── RF.predict_proba()   × 72.6%
    ├── XGB.predict_proba()  × 25.6%
    ├── LGBM.predict_proba() × 0.9%
    └── COBYLA weighted blend → {"prediction": "FRAUD", "fraud_probability": 0.68}
```

### Tester l'API

```bash
curl -X POST "https://fraud-api-245513771842.europe-west1.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 0.0,
    "time": 406.0,
    "v_features": [-2.312, 1.952, -1.610, 3.998, -0.522,
                   -1.427, -2.537, 1.392, -2.770, -2.772,
                   3.202, -2.900, -0.595, -4.289, 0.390,
                   -1.141, -2.830, -0.017, 0.417, 0.127,
                   0.517, -0.035, -0.465, 0.320, 0.045,
                   0.178, 0.261, -0.143]
  }'
```

Réponse :
```json
{
  "fraud_probability": 0.6804,
  "prediction": "FRAUD",
  "threshold": 0.46,
  "model": "COBYLA Blending (RF 72.6% + XGB 25.6%)"
}
```

---

## Installation locale

```bash
git clone https://github.com/gueye001/credit-card-fraud-detection
cd credit-card-fraud-detection
pip install -r requirements.txt
```

Dataset disponible sur
[Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
Télécharger `creditcard.csv` dans le dossier `data/`.

### Lancer l'API en local avec Docker

```bash
cd deployment
docker build -t fraud-api .
docker run -p 8080:8080 fraud-api
```

API disponible sur `http://localhost:8080/docs`

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
