"""
Train the symptom -> disease prediction model.

Compares several classifiers with stratified cross-validation, selects the best,
evaluates it on a held-out test set, and saves the fitted model together with the
ordered symptom list and label classes needed for inference in the app.

Run:  python src/train_disease.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from preprocess import load_disease_symptoms, ROOT

MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def build_candidates() -> dict:
    """Return the set of candidate classifiers to compare."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "NaiveBayes": MultinomialNB(),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.3,
            subsample=0.9,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        ),
    }


def main() -> None:
    X, y = load_disease_symptoms()
    symptoms = list(X.columns)

    # XGBoost needs integer-encoded labels; keep an encoder for all models so the
    # comparison is apples-to-apples and we can map predictions back to names.
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    print(f"{'Model':<14}{'CV mean':>10}{'CV std':>10}")
    print("-" * 34)
    for name, model in build_candidates().items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        results[name] = scores.mean()
        print(f"{name:<14}{scores.mean():>10.4f}{scores.std():>10.4f}")

    best_name = max(results, key=results.get)
    print(f"\nBest model: {best_name}  (CV acc={results[best_name]:.4f})")

    # Refit the winner on the full training split and evaluate on the test set.
    best = build_candidates()[best_name]
    best.fit(X_train, y_train)
    y_pred = best.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Held-out test accuracy: {test_acc:.4f}\n")
    print(
        classification_report(
            y_test, y_pred, target_names=le.classes_, zero_division=0
        )
    )

    # Persist everything inference needs.
    joblib.dump(best, MODELS / "disease_model.pkl")
    joblib.dump(le, MODELS / "disease_label_encoder.pkl")
    (MODELS / "symptoms.json").write_text(json.dumps(symptoms, indent=2))
    (MODELS / "disease_metrics.json").write_text(
        json.dumps(
            {
                "best_model": best_name,
                "cv_accuracy": round(float(results[best_name]), 4),
                "test_accuracy": round(float(test_acc), 4),
                "n_symptoms": len(symptoms),
                "n_diseases": int(len(le.classes_)),
                "all_cv_scores": {k: round(float(v), 4) for k, v in results.items()},
            },
            indent=2,
        )
    )
    print(f"Saved model + artifacts to {MODELS}")


if __name__ == "__main__":
    main()
