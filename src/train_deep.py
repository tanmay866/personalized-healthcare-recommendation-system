"""
Deep-learning comparison model (TensorFlow / Keras).

Implements the architecture suggested in the project brief — an Embedding
layer followed by Dense(128) -> Dense(64) — adapted properly for this task:
each symptom is a token, a patient is the *set* of their symptom tokens, and
the model learns a 32-dim embedding per symptom which is average-pooled and
classified into one of 41 diseases:

    symptom IDs -> Embedding(133, 32) -> GlobalAveragePooling
                -> Dense(128, relu) -> Dense(64, relu) -> softmax(41)

Design note: TensorFlow is deliberately **not** in requirements.txt — the
deployed app uses the (equally accurate, 40x smaller) RandomForest. This
script exists as an honest deep-learning comparison; install TF to run it:

    pip install tensorflow
    python src/train_deep.py

Outputs: models/deep_metrics.json (metrics only — the model itself is not
persisted, to keep the repo lean).
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from preprocess import load_disease_symptoms, ROOT

MODELS = ROOT / "models"
MAX_SYMPTOMS = 20  # longest observed symptom set is well under this
EMBED_DIM = 32
EPOCHS = 25


def to_token_sequences(X) -> np.ndarray:
    """Multi-hot symptom matrix -> padded sequences of symptom token IDs.

    Token 0 is reserved for padding; symptom i becomes token i+1.
    """
    seqs = np.zeros((len(X), MAX_SYMPTOMS), dtype=np.int32)
    rows, cols = np.nonzero(X.values)
    for r in np.unique(rows):
        tokens = cols[rows == r] + 1
        seqs[r, : min(len(tokens), MAX_SYMPTOMS)] = tokens[:MAX_SYMPTOMS]
    return seqs


def main() -> None:
    import tensorflow as tf
    from tensorflow.keras import layers, models

    tf.random.set_seed(42)
    np.random.seed(42)

    X, y = load_disease_symptoms()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    seqs = to_token_sequences(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        seqs, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    n_tokens = X.shape[1] + 1  # +1 for padding token
    model = models.Sequential(
        [
            layers.Input(shape=(MAX_SYMPTOMS,)),
            layers.Embedding(input_dim=n_tokens, output_dim=EMBED_DIM, mask_zero=True),
            layers.GlobalAveragePooling1D(),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(len(le.classes_), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    history = model.fit(
        X_tr, y_tr,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=64,
        verbose=2,
    )

    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    test_acc = accuracy_score(y_te, y_pred)
    print(f"\nTest accuracy: {test_acc:.4f}  ({len(y_te)} samples, {len(le.classes_)} classes)")

    n_params = int(model.count_params())
    (MODELS / "deep_metrics.json").write_text(
        json.dumps(
            {
                "framework": f"TensorFlow {tf.__version__}",
                "architecture": "Embedding(133,32) -> GlobalAvgPool -> Dense(128) -> Dense(64) -> softmax(41)",
                "epochs": EPOCHS,
                "parameters": n_params,
                "test_accuracy": round(float(test_acc), 4),
                "final_val_accuracy": round(float(history.history["val_accuracy"][-1]), 4),
                "note": "Comparison model only — the deployed app uses RandomForest "
                "(equal accuracy, no TF dependency).",
            },
            indent=2,
        )
    )
    print(f"Saved metrics to {MODELS / 'deep_metrics.json'}")


if __name__ == "__main__":
    main()
