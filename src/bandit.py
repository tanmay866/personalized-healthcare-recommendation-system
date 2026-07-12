"""
Reinforcement learning from user feedback: a Thompson-sampling bandit that
adapts the medicine ranking as 👍/👎 feedback accumulates.

Formulation
-----------
Each (disease, medicine) pair is a bandit arm with a Beta posterior over its
"helpfulness" rate:

    prior:      Beta(1 + h·C,  1 + (1-h)·C)      h = hybrid score (sentiment
                                                  + rating), C = prior strength
    posterior:  Beta(alpha0 + 👍,  beta0 + 👎)

The prior anchors each arm at its offline hybrid score (so the cold-start
ranking equals the sentiment ranking), and user feedback moves the posterior —
a few downvotes will sink a drug, sustained upvotes lift one. **Thompson
sampling** draws one sample per arm and ranks by the draws, which naturally
balances exploring uncertain arms against exploiting known-good ones.

The UI displays the ranking by posterior mean (stable between reruns) and the
sampling draw is exposed for serving decisions / tests.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

import db

# How many pseudo-observations the offline hybrid score is worth. Higher =
# more feedback needed to move a ranking; lower = faster adaptation.
PRIOR_STRENGTH = 8.0


def record_feedback(username: str, disease: str, drug: str, vote: int) -> None:
    """Store one 👍 (+1) / 👎 (-1) vote."""
    if vote not in (-1, 1):
        raise ValueError("vote must be +1 or -1")
    db.insert_feedback(
        username, disease, drug, vote, datetime.now(timezone.utc).isoformat()
    )


def rank_medicines(
    disease: str, drugs: pd.DataFrame, seed: int | None = None
) -> pd.DataFrame:
    """Re-rank a hybrid-scored medicine table using the feedback posteriors.

    Parameters
    ----------
    disease : disease the medicines treat (feedback is scoped per disease).
    drugs : DataFrame with ``drugName`` and ``hybrid_score`` columns
        (as returned by ``recommend.get_drug_sentiment``).
    seed : RNG seed for reproducible Thompson draws (tests).

    Returns
    -------
    Copy of ``drugs`` with ``ups``, ``downs``, ``posterior_mean`` and
    ``thompson_sample`` columns, sorted by posterior mean (descending).
    """
    rng = np.random.default_rng(seed)
    counts = db.feedback_counts(disease)

    ups, downs, means, samples = [], [], [], []
    for _, row in drugs.iterrows():
        u, d = counts.get(row["drugName"], (0, 0))
        h = float(row.get("hybrid_score", 0.5))
        alpha = 1.0 + h * PRIOR_STRENGTH + u
        beta = 1.0 + (1.0 - h) * PRIOR_STRENGTH + d
        ups.append(u)
        downs.append(d)
        means.append(alpha / (alpha + beta))
        samples.append(float(rng.beta(alpha, beta)))

    out = drugs.copy()
    out["ups"] = ups
    out["downs"] = downs
    out["posterior_mean"] = np.round(means, 4)
    out["thompson_sample"] = np.round(samples, 4)
    return out.sort_values("posterior_mean", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    # Smoke test: feedback should move the ranking.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from recommend import get_drug_sentiment

    disease = "Migraine"
    drugs = get_drug_sentiment(disease, top_n=5)
    print("Cold-start ranking (= hybrid ranking):")
    print(rank_medicines(disease, drugs, seed=0)[
        ["drugName", "hybrid_score", "ups", "downs", "posterior_mean"]
    ].to_string(index=False))

    top_drug = drugs.iloc[0]["drugName"]
    for _ in range(6):
        record_feedback("smoketest", disease, top_drug, -1)
    print(f"\nAfter 6 downvotes on {top_drug}:")
    print(rank_medicines(disease, drugs, seed=0)[
        ["drugName", "hybrid_score", "ups", "downs", "posterior_mean"]
    ].to_string(index=False))
