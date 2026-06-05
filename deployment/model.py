import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"

FEATURE_NAMES = (
    [f"V{i}" for i in range(1, 29)]
    + ["Amount_scaled", "Hour_scaled"]
)

class FraudDetector:
    def __init__(self):
        self.lr              = joblib.load(MODEL_DIR / "lr.pkl")
        self.rf              = joblib.load(MODEL_DIR / "rf.pkl")
        self.xgb             = joblib.load(MODEL_DIR / "xgb.pkl")
        self.lgbm            = joblib.load(MODEL_DIR / "lgbm.pkl")
        self.weights         = joblib.load(MODEL_DIR / "cobyla_weights.pkl")
        self.scaler_amount   = joblib.load(MODEL_DIR / "scaler_amount.pkl")
        self.scaler_hour     = joblib.load(MODEL_DIR / "scaler_hour.pkl")
        self.threshold       = 0.46

    def preprocess(self, amount: float, time: float,
                   v_features: list) -> pd.DataFrame:
        hour        = (time % 86400) // 3600
        amount_log  = np.log1p(amount)
        amount_sc   = self.scaler_amount.transform([[amount_log]])[0][0]
        hour_sc     = self.scaler_hour.transform([[hour]])[0][0]

        features = v_features + [amount_sc, hour_sc]
        return pd.DataFrame([features], columns=FEATURE_NAMES)

    def predict(self, amount: float, time: float,
                v_features: list) -> dict:
        X = self.preprocess(amount, time, v_features)

        probas = np.column_stack([
            self.lr.predict_proba(X)[:, 1],
            self.rf.predict_proba(X)[:, 1],
            self.xgb.predict_proba(X)[:, 1],
            self.lgbm.predict_proba(X)[:, 1],
        ])

        proba  = float(np.dot(probas, self.weights)[0])
        label  = int(proba >= self.threshold)

        return {
            "fraud_probability" : round(proba, 4),
            "prediction"        : "FRAUD" if label == 1 else "LEGITIMATE",
            "threshold"         : self.threshold,
            "model"             : "COBYLA Blending (RF 72.6% + XGB 25.6%)"
        }