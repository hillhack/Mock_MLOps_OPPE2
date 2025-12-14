from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

from google.cloud import storage

app = FastAPI()

MODEL_PATH = "model.joblib"
BUCKET_NAME = "fraud-models-bucket"
MODEL_BLOB = "model.joblib"

# ---------- Load model at startup ----------
@app.on_event("startup")
def load_model():
    global model

    if not os.path.exists(MODEL_PATH):
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_BLOB)
        blob.download_to_filename(MODEL_PATH)

    model = joblib.load(MODEL_PATH)


# ---------- Request schema ----------
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


# ---------- Prediction endpoint ----------
@app.post("/predict")
def predict(txn: Transaction):
    X = np.array([list(txn.dict().values())])
    prob = model.predict_proba(X)[0][1]
    pred = int(prob >= 0.5)

    return {
        "prediction": pred,
        "probability": prob
    }
