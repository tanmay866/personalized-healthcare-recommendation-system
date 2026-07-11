"""
Inference layer for the Streamlit app.

Loads the trained models + knowledge base once and exposes two clean functions:

* ``predict_disease(symptoms)`` -> disease, confidence, top-k, recommendations
* ``predict_risk(profile)``     -> outcome label + probability

All artifacts are loaded lazily and cached so the app stays snappy.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
KB_PATH = ROOT / "data" / "processed" / "knowledge_base.json"
SENTIMENT_PATH = ROOT / "data" / "processed" / "drug_sentiment.csv"

# Maps our 41 model disease labels to drugs.com condition names present in the
# sentiment table. Diseases without a reliable match are simply absent — the
# sentiment feature degrades gracefully for them.
DISEASE_TO_CONDITIONS: dict[str, list[str]] = {
    "Acne": ["Acne"],
    "AIDS": ["HIV Infection"],
    "Allergy": ["Allergies", "Allergic Reactions", "Allergic Rhinitis"],
    "Arthritis": ["Rheumatoid Arthritis", "Osteoarthritis"],
    "Bronchial Asthma": ["Asthma", "Asthma, Maintenance", "Asthma, acute"],
    "Common Cold": ["Cold Symptoms"],
    "Diabetes": ["Diabetes, Type 2", "Diabetes, Type 1"],
    "Dimorphic hemmorhoids(piles)": ["Hemorrhoids"],
    "Fungal infection": [
        "Tinea Corporis", "Tinea Cruris", "Tinea Pedis", "Cutaneous Candidiasis",
    ],
    "GERD": ["GERD"],
    "Gastroenteritis": ["Gastroenteritis"],
    "Heart attack": ["Heart Attack"],
    "Hepatitis B": ["Hepatitis B"],
    "Hepatitis C": ["Hepatitis C"],
    "Hypertension": ["High Blood Pressure"],
    "Hyperthyroidism": ["Hyperthyroidism"],
    "Hypothyroidism": ["Underactive Thyroid", "Hypothyroidism, After Thyroid Removal"],
    "Malaria": ["Malaria Prevention"],
    "Migraine": ["Migraine", "Migraine Prevention"],
    "Osteoarthristis": ["Osteoarthritis"],
    "Peptic ulcer diseae": ["Duodenal Ulce", "GERD"],
    "Pneumonia": ["Pneumonia"],
    "Psoriasis": ["Psoriasis", "Plaque Psoriasis"],
    "Tuberculosis": ["Tuberculosis", "Tuberculosis, Latent"],
    "Urinary tract infection": ["Urinary Tract Infection"],
    "Paralysis (brain hemorrhage)": ["Ischemic Stroke", "Ischemic Stroke, Prophylaxis"],
}


# --------------------------------------------------------------------------- #
# Lazy-loaded, cached artifacts
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _disease_artifacts():
    model = joblib.load(MODELS / "disease_model.pkl")
    le = joblib.load(MODELS / "disease_label_encoder.pkl")
    symptoms = json.loads((MODELS / "symptoms.json").read_text())
    kb = json.loads(KB_PATH.read_text())
    return model, le, symptoms, kb


@lru_cache(maxsize=1)
def _risk_artifacts():
    model = joblib.load(MODELS / "risk_model.pkl")
    features = json.loads((MODELS / "risk_features.json").read_text())
    return model, features


def list_symptoms() -> list[str]:
    """Return the ordered list of symptom names the disease model expects."""
    _, _, symptoms, _ = _disease_artifacts()
    return symptoms


def humanize(symptom: str) -> str:
    """Turn ``continuous_sneezing`` into ``Continuous Sneezing`` for display."""
    return symptom.replace("_", " ").strip().title()


# --------------------------------------------------------------------------- #
# Disease prediction + recommendations
# --------------------------------------------------------------------------- #
def predict_disease(selected_symptoms: list[str], top_k: int = 3) -> dict:
    """Predict disease from a list of symptom names and attach recommendations.

    Parameters
    ----------
    selected_symptoms : symptom names (must match the model's symptom vocabulary).
    top_k : how many alternative diagnoses to return with probabilities.

    Returns
    -------
    dict with keys: ``disease``, ``confidence``, ``top_k``, ``recommendation``.
    """
    model, le, symptoms, kb = _disease_artifacts()

    # Build the binary feature vector in the exact training column order.
    selected = set(selected_symptoms)
    x = np.array([[1 if s in selected else 0 for s in symptoms]], dtype=int)

    proba = model.predict_proba(x)[0]
    order = np.argsort(proba)[::-1]

    top = [
        {"disease": le.classes_[i], "probability": round(float(proba[i]), 4)}
        for i in order[:top_k]
    ]
    disease = top[0]["disease"]

    return {
        "disease": disease,
        "confidence": top[0]["probability"],
        "top_k": top,
        "recommendation": kb.get(disease, {}),
    }


def get_recommendation(disease: str) -> dict:
    """Return the knowledge-base entry for a disease (or empty dict)."""
    _, _, _, kb = _disease_artifacts()
    return kb.get(disease, {})


# --------------------------------------------------------------------------- #
# Personalized risk / outcome screening
# --------------------------------------------------------------------------- #
def predict_risk(profile: dict) -> dict:
    """Predict diagnosis-outcome likelihood from a patient profile.

    Parameters
    ----------
    profile : dict with keys matching the risk model features:
        fever, cough, fatigue, difficulty_breathing (0/1),
        age (int), gender (0=female,1=male),
        blood_pressure (0/1/2), cholesterol_level (0/1/2).

    Returns
    -------
    dict with ``outcome`` (Positive/Negative) and ``probability`` of positive.
    """
    model, features = _risk_artifacts()

    x = pd.DataFrame([[profile.get(f, 0) for f in features]], columns=features)
    x = x.astype(float).to_numpy()

    proba = model.predict_proba(x)[0]
    classes = list(model.classes_)
    pred = classes[int(np.argmax(proba))]
    pos_idx = classes.index("Positive") if "Positive" in classes else int(np.argmax(proba))

    return {
        "outcome": pred,
        "probability": round(float(proba[pos_idx]), 4),
        "classes": {c: round(float(p), 4) for c, p in zip(classes, proba)},
    }


# --------------------------------------------------------------------------- #
# Content-based filtering: related diseases by symptom-profile similarity
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _similarity_map() -> dict:
    p = MODELS / "disease_similarity.json"
    return json.loads(p.read_text()) if p.exists() else {}


def related_diseases(disease: str, top_n: int = 3) -> list[dict]:
    """Content-based filtering: diseases with the most similar symptom profiles.

    Similarity = cosine similarity between mean symptom vectors, precomputed at
    training time. Returns [{disease, similarity}, ...].
    """
    return _similarity_map().get(disease, [])[:top_n]


# --------------------------------------------------------------------------- #
# Drug-review sentiment (NLP feature)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _sentiment_table() -> pd.DataFrame | None:
    if not SENTIMENT_PATH.exists():
        return None
    return pd.read_csv(SENTIMENT_PATH)


def get_drug_sentiment(disease: str, top_n: int = 8) -> pd.DataFrame | None:
    """Top real-world drugs for a disease ranked by patient-review sentiment.

    Returns a DataFrame (drugName, condition, n_reviews, avg_rating,
    sentiment_score) or None when the disease has no mapped condition or the
    sentiment table hasn't been built.
    """
    table = _sentiment_table()
    conditions = DISEASE_TO_CONDITIONS.get(disease)
    if table is None or not conditions:
        return None

    rows = table[table["condition"].isin(conditions)].copy()
    if rows.empty:
        return None

    # Hybrid ranking: blend the NLP sentiment of the review text (crowd /
    # collaborative signal) with the explicit star rating (content signal).
    rows["hybrid_score"] = (
        0.6 * rows["sentiment_score"] + 0.4 * (rows["avg_rating"] / 10.0)
    ).round(4)

    # A drug can appear under multiple mapped conditions — keep its best row,
    # then rank by the hybrid score with review count as a stabilizer.
    rows = rows.sort_values("hybrid_score", ascending=False)
    rows = rows.drop_duplicates(subset="drugName", keep="first")
    rows = rows.sort_values(
        ["hybrid_score", "n_reviews"], ascending=[False, False]
    ).head(top_n)
    return rows.reset_index(drop=True)


def sentiment_conditions_for(disease: str) -> list[str]:
    """The drugs.com condition names a disease maps to (for display)."""
    return DISEASE_TO_CONDITIONS.get(disease, [])


def list_sentiment_conditions() -> list[str]:
    """All condition names available in the sentiment table (for the explorer)."""
    table = _sentiment_table()
    if table is None:
        return []
    return sorted(table["condition"].unique().tolist())


def condition_sentiment(condition: str, top_n: int = 15) -> pd.DataFrame | None:
    """Top drugs for an arbitrary drugs.com condition (sentiment explorer)."""
    table = _sentiment_table()
    if table is None:
        return None
    rows = table[table["condition"] == condition].copy()
    if rows.empty:
        return None
    return (
        rows.sort_values(["sentiment_score", "n_reviews"], ascending=[False, False])
        .head(top_n)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    # Smoke test with a couple of hand-picked symptoms.
    syms = list_symptoms()
    demo = [s for s in syms if s in {"itching", "skin_rash", "nodal_skin_eruptions"}]
    print("Symptoms used:", demo)
    print("Disease:", predict_disease(demo)["disease"])
    print("Top-3:", predict_disease(demo)["top_k"])
    print()
    print(
        "Risk demo:",
        predict_risk(
            {
                "fever": 1,
                "cough": 1,
                "fatigue": 1,
                "difficulty_breathing": 1,
                "age": 55,
                "gender": 1,
                "blood_pressure": 2,
                "cholesterol_level": 2,
            }
        ),
    )
    print()
    sent = get_drug_sentiment("Migraine", top_n=5)
    print("Sentiment demo (Migraine):")
    print(sent.to_string(index=False) if sent is not None else "(no sentiment data)")
