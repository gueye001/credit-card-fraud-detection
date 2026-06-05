from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from model import FraudDetector

app = FastAPI(
    title="Fraud Detection API",
    description="COBYLA Blending — RF 72.6% + XGBoost 25.6%",
    version="1.0.0"
)

detector = FraudDetector()


class Transaction(BaseModel):
    amount: float = Field(..., ge=0, description="Montant en euros")
    time: float   = Field(..., ge=0, description="Secondes depuis début dataset")
    v_features: List[float] = Field(..., min_items=28, max_items=28,
                                     description="Features V1 à V28")

class PredictionResponse(BaseModel):
    fraud_probability : float
    prediction        : str
    threshold         : float
    model             : str


@app.get("/")
def root():
    return {
        "service" : "Fraud Detection API",
        "version" : "1.0.0",
        "status"  : "online"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    try:
        result = detector.predict(
            amount     = transaction.amount,
            time       = transaction.time,
            v_features = transaction.v_features
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
def model_info():
    return {
        "model"      : "COBYLA Blending",
        "weights"    : {
            "Random Forest"        : "72.6%",
            "XGBoost"              : "25.6%",
            "Logistic Regression"  : "0.9%",
            "LightGBM"             : "0.9%"
        },
        "threshold"   : 0.46,
        "metrics"     : {
            "precision"           : 0.9524,
            "recall"              : 0.8163,
            "f1"                  : 0.8791,
            "average_precision"   : 0.8648
        }
    }
