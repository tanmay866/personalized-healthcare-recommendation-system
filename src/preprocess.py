"""
Preprocessing utilities for the Personalized Healthcare Recommendation System.

Two datasets are handled independently because they power two separate models:

1. ``disease_symptoms.csv``  -> symptom (132 binary flags) -> disease (41 classes)
   Used by the disease-prediction model.

2. ``patient_profile.csv``   -> symptoms + vitals -> personalized risk level
   Used by the risk-assessment model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Project paths (resolve relative to this file so scripts work from any cwd).
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


# --------------------------------------------------------------------------- #
# Dataset 1: disease  <-  symptoms
# --------------------------------------------------------------------------- #
def load_disease_symptoms() -> tuple[pd.DataFrame, pd.Series]:
    """Load and clean the symptom -> disease dataset.

    Returns
    -------
    X : DataFrame of 0/1 symptom flags (one column per symptom).
    y : Series of disease labels (stripped of stray whitespace).
    """
    df = pd.read_csv(RAW / "disease_symptoms.csv")

    # The source file has a trailing empty column (caused by a stray comma).
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.dropna(axis=1, how="all")

    target = "prognosis"
    y = df[target].astype(str).str.strip()
    X = df.drop(columns=[target])

    # Ensure every symptom column is a clean 0/1 integer flag.
    X = X.fillna(0).astype(int)

    return X, y


# --------------------------------------------------------------------------- #
# Dataset 2: personalized risk  <-  symptoms + vitals
# --------------------------------------------------------------------------- #
# Columns that are pre-scaled duplicates of raw features -> drop to avoid leakage
# and redundancy. We keep the human-readable raw values instead.
_LEAKY_COLS = ["age_scaled", "bp_scaled", "chol_scaled"]

_BINARY_SYMPTOMS = ["fever", "cough", "fatigue", "difficulty_breathing"]


def load_patient_profile(
    target: str = "risk_level", include_disease: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    """Load and clean the patient-profile dataset for the risk model.

    Parameters
    ----------
    target : which column to predict. ``"risk_level"`` (Low/Medium/High) by
        default; ``"outcome_variable"`` (Positive/Negative) also supported.
    include_disease : if True, one-hot encode the disease name as a feature.
        Off by default: 116 disease categories over only ~300 rows would
        overfit badly. The risk model predicts from symptoms + vitals, which
        generalizes and matches the app's "assess my risk" flow where the
        disease may not yet be known.

    Returns
    -------
    X : DataFrame of model-ready numeric features.
    y : Series of target labels.
    """
    df = pd.read_csv(RAW / "patient_profile.csv")

    # Drop exact duplicate rows (the raw file contains ~49).
    df = df.drop_duplicates().reset_index(drop=True)

    # Drop redundant pre-scaled columns.
    df = df.drop(columns=[c for c in _LEAKY_COLS if c in df.columns])

    y = df[target].astype(str).str.strip()

    # Never let the two candidate targets leak into the features.
    drop_from_X = {"risk_level", "outcome_variable"}
    X = df.drop(columns=[c for c in drop_from_X if c in df.columns])

    # Encode Yes/No symptom flags -> 1/0.
    for col in _BINARY_SYMPTOMS:
        if col in X.columns:
            X[col] = (X[col].astype(str).str.strip().str.lower() == "yes").astype(int)

    # Encode gender -> 0/1.
    if "gender" in X.columns:
        X["gender"] = (X["gender"].astype(str).str.strip().str.lower() == "male").astype(int)

    # blood_pressure / cholesterol_level are already ordinal ints (0/1/2).
    for col in ["blood_pressure", "cholesterol_level", "age"]:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    # The disease name is dropped by default (see docstring). Optionally one-hot
    # encode it when the caller explicitly wants it as a feature.
    if "disease" in X.columns:
        if include_disease:
            X = pd.get_dummies(X, columns=["disease"], prefix="dis")
        else:
            X = X.drop(columns=["disease"])

    # Cast any remaining boolean dummy columns to int for model friendliness.
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

    return X, y


if __name__ == "__main__":
    Xd, yd = load_disease_symptoms()
    print(f"[disease_symptoms] X={Xd.shape}  diseases={yd.nunique()}")

    Xr, yr = load_patient_profile("risk_level")
    print(f"[patient_profile] X={Xr.shape}  target classes={yr.nunique()} -> {sorted(yr.unique())}")
