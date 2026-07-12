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
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report, log_loss
from sklearn.neural_network import MLPClassifier

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
        "NeuralNet (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            random_state=42,
        ),
    }


def tune_best(name: str, X_train, y_train, cv) -> tuple:
    """Grid-search the winning model family for a final accuracy squeeze."""
    grids = {
        "RandomForest": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                "n_estimators": [200, 400],
                "max_depth": [6, 8, 12, None],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2, 3, 4],
            },
        ),
    }
    if name not in grids:
        return None, None
    est, grid = grids[name]
    gs = GridSearchCV(est, grid, cv=cv, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


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

    # Hyperparameter tuning on the winning family.
    tuned, best_params = tune_best(best_name, X_train, y_train, cv)
    if tuned is not None:
        tuned_cv = cross_val_score(tuned, X_train, y_train, cv=cv).mean()
        print(f"\nTuned {best_name}: CV {tuned_cv:.4f} (was {results[best_name]:.4f})  params={best_params}")
        if tuned_cv >= results[best_name]:
            best = tuned
            results[best_name] = tuned_cv
        else:
            best = build_candidates()[best_name]
    else:
        best = build_candidates()[best_name]

    best.fit(X_train, y_train)

    # ------------------------------------------------------------------ #
    # Probability calibration. The app surfaces raw probabilities (risk
    # gauge), so they should be trustworthy: a predicted 70% should come
    # true ~70% of the time. We compare the uncalibrated model against
    # sigmoid (Platt) and isotonic calibration by Brier score / log-loss
    # and keep the winner.
    # ------------------------------------------------------------------ #
    pos_label = "Positive"

    def _probs(model):
        idx = list(model.classes_).index(pos_label)
        return model.predict_proba(X_test)[:, idx]

    y_test_bin = (y_test == pos_label).astype(int)
    candidates_cal = {"uncalibrated": best}
    for method in ("sigmoid", "isotonic"):
        cal = CalibratedClassifierCV(clone(best), method=method, cv=5)
        cal.fit(X_train, y_train)
        candidates_cal[method] = cal

    print(f"\n{'Calibration':<14}{'Brier':>8}{'LogLoss':>9}{'Acc':>7}")
    print("-" * 38)
    scores = {}
    for name, model in candidates_cal.items():
        p = _probs(model)
        scores[name] = brier_score_loss(y_test_bin, p)
        print(
            f"{name:<14}{scores[name]:>8.4f}"
            f"{log_loss(y_test_bin, p):>9.4f}"
            f"{accuracy_score(y_test, model.predict(X_test)):>7.3f}"
        )

    # Adopt a calibrated variant only if it clearly improves probability
    # quality (Brier -0.005 or better) without materially hurting accuracy
    # (>1pt drop). On a small test set, tiny Brier differences are noise —
    # trading 5pts of accuracy for 0.001 Brier would be a bad deal.
    base_brier = scores["uncalibrated"]
    base_acc = accuracy_score(y_test, candidates_cal["uncalibrated"].predict(X_test))
    cal_choice = "uncalibrated"
    for name in ("isotonic", "sigmoid"):
        acc = accuracy_score(y_test, candidates_cal[name].predict(X_test))
        if scores[name] <= base_brier - 0.005 and acc >= base_acc - 0.01:
            cal_choice = name
            break
    best = candidates_cal[cal_choice]
    print(f"\nCalibration selected: {cal_choice}")

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
                "calibration": cal_choice,
                "test_brier_score": round(float(scores[cal_choice]), 4),
                "test_brier_uncalibrated": round(float(scores["uncalibrated"]), 4),
                "features": features,
                "classes": sorted(y.unique().tolist()),
            },
            indent=2,
        )
    )
    print(f"Saved risk model + artifacts to {MODELS}")


if __name__ == "__main__":
    main()
