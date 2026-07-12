"""
Inference for the disease-specific clinical risk models (heart, diabetes).

Loads the sklearn pipelines trained by ``train_clinical.py`` on real clinical
datasets, plus the UI form specs so the app can render inputs that always
match the model's features.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

_MODEL_FILES = {"heart": "heart_model.pkl", "diabetes": "diabetes_model.pkl"}


@lru_cache(maxsize=1)
def clinical_specs() -> dict:
    """UI form specifications for both risk calculators."""
    return json.loads((MODELS / "clinical_specs.json").read_text())


@lru_cache(maxsize=1)
def clinical_metrics() -> dict:
    return json.loads((MODELS / "clinical_metrics.json").read_text())


@lru_cache(maxsize=2)
def _model(which: str):
    return joblib.load(MODELS / _MODEL_FILES[which])


def predict_clinical_risk(which: str, values: dict) -> dict:
    """Risk probability from one of the clinical models.

    Parameters
    ----------
    which : "heart" or "diabetes".
    values : feature -> value mapping. Missing/None features are allowed —
        the pipeline's median imputer fills them (that's how the training
        data's hidden-missing values were handled too).

    Returns
    -------
    {"risk_probability": float, "risk_label": "High"|"Moderate"|"Low"}
    """
    if which not in _MODEL_FILES:
        raise ValueError(f"Unknown clinical model: {which}")
    features = clinical_metrics()[which]["features"]
    row = pd.DataFrame([[values.get(f) for f in features]], columns=features)
    proba = float(_model(which).predict_proba(row)[0][1])
    label = "High" if proba >= 0.6 else ("Moderate" if proba >= 0.35 else "Low")
    return {"risk_probability": round(proba, 4), "risk_label": label}


if __name__ == "__main__":
    # Smoke test with typical high/low-risk profiles.
    high_heart = {"age": 63, "sex": 1, "cp": 4, "trestbps": 160, "chol": 300,
                  "fbs": 1, "restecg": 2, "thalach": 108, "exang": 1,
                  "oldpeak": 3.0, "slope": 2, "ca": 2, "thal": 7}
    low_heart = {"age": 35, "sex": 0, "cp": 3, "trestbps": 115, "chol": 180,
                 "fbs": 0, "restecg": 0, "thalach": 185, "exang": 0,
                 "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3}
    print("heart high-risk profile ->", predict_clinical_risk("heart", high_heart))
    print("heart low-risk profile  ->", predict_clinical_risk("heart", low_heart))

    high_dia = {"pregnancies": 8, "glucose": 190, "blood_pressure": 90,
                "skin_thickness": 40, "insulin": 300, "bmi": 38.0,
                "diabetes_pedigree": 1.2, "age": 55}
    low_dia = {"pregnancies": 0, "glucose": 90, "blood_pressure": 70,
               "skin_thickness": 20, "insulin": 80, "bmi": 22.0,
               "diabetes_pedigree": 0.2, "age": 25}
    print("diabetes high-risk profile ->", predict_clinical_risk("diabetes", high_dia))
    print("diabetes low-risk profile  ->", predict_clinical_risk("diabetes", low_dia))
    print("missing-values tolerated   ->", predict_clinical_risk("diabetes", {"glucose": 150, "age": 45}))
