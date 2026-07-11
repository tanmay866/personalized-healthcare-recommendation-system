"""
Train the personalized health-risk / diagnosis-outcome model.

Predicts whether a patient's diagnostic outcome is likely Positive or Negative
from their symptoms and vitals (age, gender, blood pressure, cholesterol). This
is the "personalization / screening" half of the system and complements the
symptom -> disease model.

Note on target choice: the dataset's ``risk_level`` column (Low/Medium/High) is
not learnable from these features (models sit at the majority-class baseline),
whereas ``outcome_variable`` carries real signal (~71% CV vs ~52% baseline). We
therefore predict ``outcome_variable`` and surface it as a "risk screening".

Run:  python src/train_risk.py
"""

from __future__ import annotations

import json

import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report

from preprocess import load_patient_profile, ROOT

MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def build_candidates() -> dict:
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    }


TARGET = "outcome_variable"


def main() -> None:
    X, y = load_patient_profile(target=TARGET)
    features = list(X.columns)

    X_arr = X.astype(float).to_numpy()
    y_arr = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    print(f"{'Model':<20}{'CV mean':>10}{'CV std':>10}")
    print("-" * 40)
    for name, model in build_candidates().items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        results[name] = scores.mean()
        print(f"{name:<20}{scores.mean():>10.4f}{scores.std():>10.4f}")

    best_name = max(results, key=results.get)
    best = build_candidates()[best_name]
    best.fit(X_train, y_train)
    y_pred = best.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"\nBest model: {best_name}  (CV acc={results[best_name]:.4f})")
    print(f"Held-out test accuracy: {test_acc:.4f}\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    from collections import Counter

    baseline = max(Counter(y_arr).values()) / len(y_arr)

    joblib.dump(best, MODELS / "risk_model.pkl")
    (MODELS / "risk_features.json").write_text(json.dumps(features, indent=2))
    (MODELS / "risk_metrics.json").write_text(
        json.dumps(
            {
                "target": TARGET,
                "best_model": best_name,
                "cv_accuracy": round(float(results[best_name]), 4),
                "test_accuracy": round(float(test_acc), 4),
                "majority_baseline": round(float(baseline), 4),
                "features": features,
                "classes": sorted(y.unique().tolist()),
            },
            indent=2,
        )
    )
    print(f"Saved risk model + artifacts to {MODELS}")


if __name__ == "__main__":
    main()
