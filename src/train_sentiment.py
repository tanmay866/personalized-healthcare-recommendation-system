"""
Train the NLP sentiment model on real patient drug reviews (UCI Drug Review
dataset, drugs.com) and precompute per-drug sentiment scores.

This powers the "medicine sentiment" feature: recommended medicines are ranked
by how satisfied real patients were, and users can explore drug sentiment for
a condition.

Pipeline
--------
1. Load raw reviews (161K train split of drugsCom).
2. Label sentiment from the star rating: >=7 positive, <=4 negative
   (5-6 dropped as ambiguous) — the standard convention for this dataset.
3. TF-IDF (uni+bi-grams) -> LogisticRegression classifier.
4. Evaluate on the official test split.
5. Aggregate per (drug, condition): review count, mean rating, and mean
   model-predicted positive probability -> data/processed/drug_sentiment.csv
6. Save a small review sample so the EDA notebook runs without the full data.

Data source (not committed — 112 MB):
  https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip
Download and unzip, then point --data-dir at the folder with the two .tsv files.

Run:  python src/train_sentiment.py --data-dir /path/to/tsvs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"


def label_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a binary sentiment label from the 1-10 star rating."""
    df = df.dropna(subset=["review", "rating"]).copy()
    df = df[df["rating"] != 0]
    df["sentiment"] = (df["rating"] >= 7).astype(int)
    # Drop ambiguous middle ratings (5-6) from *training* data only.
    return df[(df["rating"] <= 4) | (df["rating"] >= 7)]


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50_000,
                    ngram_range=(1, 2),
                    stop_words="english",
                    min_df=3,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=1000, C=4.0, n_jobs=-1, random_state=42),
            ),
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Folder containing drugsComTrain_raw.tsv / drugsComTest_raw.tsv")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)

    train_raw = pd.read_csv(data_dir / "drugsComTrain_raw.tsv", sep="\t")
    test_raw = pd.read_csv(data_dir / "drugsComTest_raw.tsv", sep="\t")

    train = label_sentiment(train_raw)
    test = label_sentiment(test_raw)
    print(f"Train reviews: {len(train):,}  |  Test reviews: {len(test):,}")
    print(f"Positive share (train): {train['sentiment'].mean():.3f}")

    pipe = build_pipeline()
    pipe.fit(train["review"], train["sentiment"])

    pred = pipe.predict(test["review"])
    acc = accuracy_score(test["sentiment"], pred)
    f1 = f1_score(test["sentiment"], pred)
    print(f"\nTest accuracy: {acc:.4f}   F1: {f1:.4f}\n")
    print(classification_report(test["sentiment"], pred, target_names=["negative", "positive"]))

    # ------------------------------------------------------------------ #
    # Per-drug sentiment table (uses ALL reviews incl. neutral, scored by
    # the model so every review contributes a probability).
    # ------------------------------------------------------------------ #
    all_reviews = pd.concat([train_raw, test_raw], ignore_index=True)
    all_reviews = all_reviews.dropna(subset=["review", "drugName", "condition"])
    # Filter noise conditions like "3</span> users found this comment helpful."
    all_reviews = all_reviews[~all_reviews["condition"].str.contains("</span>", na=False)]

    print(f"\nScoring {len(all_reviews):,} reviews for per-drug aggregation...")
    all_reviews["pos_proba"] = pipe.predict_proba(all_reviews["review"])[:, 1]

    agg = (
        all_reviews.groupby(["drugName", "condition"])
        .agg(
            n_reviews=("review", "size"),
            avg_rating=("rating", "mean"),
            sentiment_score=("pos_proba", "mean"),
            total_useful=("usefulCount", "sum"),
        )
        .reset_index()
    )
    # Keep drug/condition pairs with enough reviews to be meaningful.
    agg = agg[agg["n_reviews"] >= 5].copy()
    agg["avg_rating"] = agg["avg_rating"].round(2)
    agg["sentiment_score"] = agg["sentiment_score"].round(4)
    agg = agg.sort_values(["condition", "sentiment_score"], ascending=[True, False])
    agg.to_csv(PROCESSED / "drug_sentiment.csv", index=False)
    print(f"Wrote {len(agg):,} (drug, condition) sentiment rows -> drug_sentiment.csv")

    # Small sample so notebooks/demos run without the 112MB download.
    sample = all_reviews.sample(n=5000, random_state=42)[
        ["drugName", "condition", "review", "rating", "usefulCount"]
    ]
    sample.to_csv(PROCESSED / "drug_reviews_sample.csv", index=False)

    joblib.dump(pipe, MODELS / "sentiment_model.pkl")
    (MODELS / "sentiment_metrics.json").write_text(
        json.dumps(
            {
                "model": "TF-IDF (1-2 grams, 50k feats) + LogisticRegression",
                "train_reviews": int(len(train)),
                "test_reviews": int(len(test)),
                "test_accuracy": round(float(acc), 4),
                "test_f1": round(float(f1), 4),
                "labeling": "rating >=7 positive, <=4 negative, 5-6 dropped",
                "n_drug_condition_pairs": int(len(agg)),
            },
            indent=2,
        )
    )
    print(f"Saved sentiment model + metrics to {MODELS}")


if __name__ == "__main__":
    main()
