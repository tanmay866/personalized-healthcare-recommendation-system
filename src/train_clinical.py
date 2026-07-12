"""
Train disease-specific risk models on REAL clinical datasets.

1. Heart-disease risk — UCI Cleveland Heart Disease (303 real patients).
   Target: presence of coronary artery disease (num > 0), the standard
   binarization used in the literature.
   Known data issues handled: 6 missing values in ``ca``/``thal``.

2. Diabetes risk — Pima Indians Diabetes (768 real patients).
   Known data issue handled: zeros in glucose/blood_pressure/skin_thickness/
   insulin/bmi are physiologically impossible -> treated as missing and
   median-imputed inside the pipeline (so inference handles gaps too).

Both use sklearn Pipelines (imputer + scaler + classifier) so the saved model
is self-contained. Selection by cross-validated ROC-AUC — the standard metric
for clinical risk, robust to class imbalance.

Run:  python src/train_clinical.py
Outputs: models/heart_model.pkl, models/diabetes_model.pkl,
         models/clinical_metrics.json, models/clinical_specs.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import joblib

from preprocess import ROOT, RAW

MODELS = ROOT / "models"

PIMA_ZERO_AS_MISSING = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]


def load_heart() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(RAW / "heart_cleveland.csv")
    y = (df["num"] > 0).astype(int)  # standard binarization: any CAD vs none
    X = df.drop(columns=["num"])
    return X, y


def load_pima() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(RAW / "pima_diabetes.csv")
    # Zeros in these columns are hidden missing values (impossible readings).
    df[PIMA_ZERO_AS_MISSING] = df[PIMA_ZERO_AS_MISSING].replace(0, np.nan)
    y = df["outcome"].astype(int)
    X = df.drop(columns=["outcome"])
    return X, y


def build_candidates() -> dict:
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.1, random_state=42
        ),
    }


def train_one(name: str, X: pd.DataFrame, y: pd.Series) -> dict:
    features = list(X.columns)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    print(f"\n=== {name} ===  ({len(X)} patients, positive rate {y.mean():.2f})")
    print(f"{'Model':<20}{'CV AUC':>9}{'std':>7}")
    print("-" * 36)
    results = {}
    for mname, clf in build_candidates().items():
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )
        scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
        results[mname] = (pipe, scores.mean(), scores.std())
        print(f"{mname:<20}{scores.mean():>9.4f}{scores.std():>7.4f}")

    best_name = max(results, key=lambda k: results[k][1])
    best_pipe = results[best_name][0]
    best_pipe.fit(X_tr, y_tr)

    proba = best_pipe.predict_proba(X_te)[:, 1]
    pred = best_pipe.predict(X_te)
    auc = roc_auc_score(y_te, proba)
    acc = accuracy_score(y_te, pred)
    baseline = max(y.mean(), 1 - y.mean())
    print(f"Best: {best_name} | test AUC {auc:.4f} | test acc {acc:.4f} | majority baseline {baseline:.4f}")

    return {
        "pipeline": best_pipe,
        "metrics": {
            "best_model": best_name,
            "cv_auc": round(float(results[best_name][1]), 4),
            "test_auc": round(float(auc), 4),
            "test_accuracy": round(float(acc), 4),
            "majority_baseline": round(float(baseline), 4),
            "n_patients": int(len(X)),
            "positive_rate": round(float(y.mean()), 4),
            "features": features,
        },
    }


# UI form specifications: label, kind, range, default, help — the app renders
# its input forms from this so the models and UI can't drift apart.
SPECS = {
    "heart": {
        "title": "Heart Disease Risk (UCI Cleveland, 303 real patients)",
        "fields": {
            "age": {"label": "Age", "kind": "int", "min": 20, "max": 90, "default": 50},
            "sex": {"label": "Sex", "kind": "select", "options": {"Female": 0, "Male": 1}},
            "cp": {"label": "Chest pain type", "kind": "select", "options": {
                "Typical angina": 1, "Atypical angina": 2, "Non-anginal pain": 3, "Asymptomatic": 4}},
            "trestbps": {"label": "Resting blood pressure (mm Hg)", "kind": "int", "min": 90, "max": 210, "default": 130},
            "chol": {"label": "Serum cholesterol (mg/dl)", "kind": "int", "min": 100, "max": 600, "default": 240},
            "fbs": {"label": "Fasting blood sugar > 120 mg/dl", "kind": "select", "options": {"No": 0, "Yes": 1}},
            "restecg": {"label": "Resting ECG", "kind": "select", "options": {
                "Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}},
            "thalach": {"label": "Max heart rate achieved", "kind": "int", "min": 60, "max": 220, "default": 150},
            "exang": {"label": "Exercise-induced angina", "kind": "select", "options": {"No": 0, "Yes": 1}},
            "oldpeak": {"label": "ST depression (exercise vs rest)", "kind": "float", "min": 0.0, "max": 7.0, "default": 1.0},
            "slope": {"label": "Slope of peak exercise ST", "kind": "select", "options": {
                "Upsloping": 1, "Flat": 2, "Downsloping": 3}},
            "ca": {"label": "Major vessels colored by fluoroscopy", "kind": "select", "options": {
                "0": 0, "1": 1, "2": 2, "3": 3}},
            "thal": {"label": "Thalassemia test", "kind": "select", "options": {
                "Normal": 3, "Fixed defect": 6, "Reversible defect": 7}},
        },
    },
    "diabetes": {
        "title": "Diabetes Risk (Pima Indians, 768 real patients)",
        "fields": {
            "pregnancies": {"label": "Number of pregnancies", "kind": "int", "min": 0, "max": 20, "default": 1},
            "glucose": {"label": "Plasma glucose (mg/dl)", "kind": "int", "min": 40, "max": 250, "default": 110},
            "blood_pressure": {"label": "Diastolic blood pressure (mm Hg)", "kind": "int", "min": 30, "max": 130, "default": 72},
            "skin_thickness": {"label": "Triceps skin fold (mm)", "kind": "int", "min": 5, "max": 100, "default": 25},
            "insulin": {"label": "2-hour serum insulin (mu U/ml)", "kind": "int", "min": 10, "max": 900, "default": 100},
            "bmi": {"label": "Body mass index", "kind": "float", "min": 15.0, "max": 70.0, "default": 28.0},
            "diabetes_pedigree": {"label": "Diabetes pedigree (family history score)", "kind": "float", "min": 0.05, "max": 2.5, "default": 0.4},
            "age": {"label": "Age", "kind": "int", "min": 18, "max": 90, "default": 35},
        },
    },
}


def main() -> None:
    heart = train_one("Heart disease (Cleveland)", *load_heart())
    pima = train_one("Diabetes (Pima)", *load_pima())

    joblib.dump(heart["pipeline"], MODELS / "heart_model.pkl")
    joblib.dump(pima["pipeline"], MODELS / "diabetes_model.pkl")
    (MODELS / "clinical_metrics.json").write_text(
        json.dumps({"heart": heart["metrics"], "diabetes": pima["metrics"]}, indent=2)
    )
    (MODELS / "clinical_specs.json").write_text(json.dumps(SPECS, indent=2))
    print(f"\nSaved heart + diabetes pipelines, metrics and UI specs to {MODELS}")


if __name__ == "__main__":
    main()
