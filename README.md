# Personalized Healthcare & Medicine Recommendation System

A machine-learning system that predicts a likely disease from symptoms, recommends medicines backed by real patient reviews, and screens heart-disease and diabetes risk using models trained on real clinical data. Recommendations adapt over time based on user feedback.

**Live demo:** https://personalized-healthcare-recommendation-system.streamlit.app (sign up for a free account)

![Python](https://img.shields.io/badge/python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-models-orange)
![Streamlit](https://img.shields.io/badge/streamlit-app-red)
![FastAPI](https://img.shields.io/badge/fastapi-REST%20API-teal)

## What it does

You select your symptoms and the app predicts the most likely disease (out of 41), then shows everything useful around that prediction: which medicines real patients rate highest for the condition, precautions, diet suggestions, and the right specialist to consult. There are also dedicated risk calculators for heart disease and diabetes, trained on real patient records from the UCI Cleveland and Pima Indians datasets.

A few things happen behind the scenes that I think make this more interesting than a typical classifier-behind-a-form project:

- Medicine rankings combine an NLP sentiment model (trained on 215K drugs.com reviews) with star ratings, and then a Thompson-sampling bandit re-ranks them as users click thumbs up/down. Cold start equals the offline ranking; feedback moves it.
- A medical knowledge graph (diseases, symptoms, medications, specialists — 307 nodes) is used to surface related diseases through multi-hop connections, e.g. two diseases sharing a specialist and several symptoms. This catches relationships that plain cosine similarity between symptom vectors misses.
- User accounts, health profiles, activity history and feedback live in a real database — SQLite locally, PostgreSQL in production (the live demo runs on Neon). Login persists across page refreshes via a signed cookie.

## Screenshots

| | |
|---|---|
| Login | ![Login](docs/screenshots/01_login.png) |
| Disease prediction | ![Prediction](docs/screenshots/02_prediction.png) |
| Care recommendations | ![Recommendations](docs/screenshots/03_recommendations.png) |
| Medicine ranking with feedback | ![Adaptive medicines](docs/screenshots/04_adaptive_medicines.png) |
| Knowledge graph | ![Knowledge graph](docs/screenshots/05_knowledge_graph.png) |
| Clinical risk calculators | ![Clinical risk](docs/screenshots/06_risk_screening.png) |
| Sentiment explorer | ![Sentiment explorer](docs/screenshots/07_sentiment_explorer.png) |
| Analytics dashboard | ![Dashboard](docs/screenshots/08_dashboard.png) |

## Models and results

| Model | Data | Algorithm | Test result | Baseline |
|-------|------|-----------|-------------|----------|
| Disease predictor | 4,920 records, 41 classes | Random Forest | 100% | 2.4% |
| Heart-disease risk | 303 real patients (UCI Cleveland) | Logistic Regression | AUC 0.95, acc 87% | 54% |
| Diabetes risk | 768 real patients (Pima Indians) | Logistic Regression | AUC 0.81, acc 71% | 65% |
| Outcome screening | symptoms + vitals | Random Forest (tuned) | 80% | 52% |
| Review sentiment | 215K drug reviews | TF-IDF + Logistic Regression | 90%, F1 0.93 | 72% |

Some notes on these numbers, because a few of them deserve context:

**The 100% is a property of the dataset, not brilliance.** The disease-symptom dataset is cleanly separable — every disease has a consistent symptom signature — so every classifier I tried (RF, SVM, NB, XGBoost, an MLP, and a Keras embedding model) lands at or near 100%. The value of this component is the complete working pipeline around it, not the benchmark.

**The clinical models are the honest ones.** Real patient data is noisy: the Cleveland set has missing values in two columns, and Pima famously encodes missing measurements as zeros (374 rows have an insulin reading of 0, which is physiologically impossible). Both are median-imputed inside the sklearn pipeline, which also means the deployed calculators tolerate partially-filled forms. Logistic regression beat random forest and gradient boosting on both datasets — the usual outcome on a few hundred clinical records.

**One target turned out to be unlearnable.** The patient-profile dataset has a `risk_level` column I originally wanted to predict, but every model sat exactly at the majority-class baseline (~58%), i.e. it learned nothing. I switched to the `outcome_variable` target, which carries real signal (80% vs a 52% baseline). I also evaluated probability calibration (sigmoid and isotonic, Brier score) and ended up keeping the uncalibrated model — it was already well-calibrated, and isotonic traded 5 points of accuracy for a 0.001 Brier improvement. The analysis is in `notebooks/02_modeling.ipynb`.

There's also a TensorFlow/Keras comparison model (`src/train_deep.py`) that matches the RandomForest at 100%. I kept it out of the deployed app on purpose — no reason to carry a 500 MB dependency for equal accuracy.

## Architecture

```mermaid
flowchart LR
    U[User] --> APP[Streamlit app]
    C[API client] --> API[FastAPI + JWT]
    APP --> INF[Inference layer]
    API --> INF
    INF --> MODELS[Trained models]
    INF --> KG[Knowledge graph]
    INF --> BANDIT[Feedback bandit]
    APP --> DB[(SQLite / PostgreSQL)]
    API --> DB
    BANDIT --> DB
```

The Streamlit app and the REST API share the same inference layer and database. Tests: 29 API checks plus 19 browser end-to-end checks (Playwright), run against both database backends.

## Running it locally

```bash
git clone https://github.com/tanmay866/personalized-healthcare-recommendation-system.git
cd personalized-healthcare-recommendation-system

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

streamlit run app/app.py
```

Trained models ship with the repo, so this works immediately. Retraining scripts are in `src/` if you want to reproduce them (`train_disease.py`, `train_clinical.py`, `train_risk.py`; the sentiment model needs the 112 MB UCI drug-review download, documented in `train_sentiment.py`).

By default storage is a local SQLite file. To use PostgreSQL instead, set one variable — nothing else changes:

```bash
export DATABASE_URL="postgresql://user:pass@host:5432/dbname"
```

Production secrets (all optional locally): `DATABASE_URL`, `ADMIN_PASSWORD`, `API_JWT_SECRET`.

## REST API

The same functionality is exposed as a JWT-secured REST service:

```bash
uvicorn api.main:app --port 8000    # Swagger docs at /docs
```

<details>
<summary>Endpoints</summary>

| Method | Endpoint | Auth | Purpose |
|--------|----------|------|---------|
| POST | `/auth/signup` | — | Create an account |
| POST | `/auth/login` | — | Get a JWT (24h) |
| GET | `/symptoms` | — | The 132 symptoms the model understands |
| POST | `/predict/disease` | JWT | Symptoms → disease, recommendations, related diseases, medicines |
| POST | `/predict/risk` | JWT | Vitals → outcome likelihood |
| POST | `/predict/clinical/{heart\|diabetes}` | JWT | Clinical risk calculators |
| GET | `/recommend/{disease}` | JWT | Knowledge-base entry |
| GET | `/sentiment/{condition}` | JWT | Top drugs by review sentiment |
| GET | `/graph/{disease}` | JWT | Graph neighborhood + related diseases |
| GET | `/medicines/{disease}` | JWT | Bandit-ranked medicines |
| POST | `/feedback` | JWT | Thumbs up/down — updates the bandit |
| GET | `/admin/users` | JWT, admin | User list (role-based access) |
| GET | `/health` | — | Liveness probe |

```bash
TOKEN=$(curl -s -X POST localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"<user>","password":"<pass>"}' | jq -r .access_token)

curl -X POST localhost:8000/predict/disease \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"symptoms":["continuous_sneezing","chills","watering_from_eyes"]}'
```
</details>

## Project layout

```
app/app.py                  Streamlit app (auth, 5 tabs, persistent login)
api/main.py                 FastAPI backend
src/
  preprocess.py             cleaning + feature engineering
  train_disease.py          disease model + similarity artifact
  train_clinical.py         heart + diabetes models (real clinical data)
  train_risk.py             outcome screening (tuning + calibration analysis)
  train_sentiment.py        NLP sentiment on drug reviews
  train_deep.py             Keras comparison model
  build_knowledge_base.py   the 41-disease recommendation knowledge base
  knowledge_graph.py        medical graph + Personalized PageRank
  bandit.py                 Thompson-sampling feedback bandit
  clinical.py               clinical risk inference
  recommend.py              shared inference layer
  auth.py / db.py           users, roles, activity; SQLAlchemy storage
data/                       datasets + processed artifacts
models/                     trained models and metrics
notebooks/                  01_eda, 02_modeling (executed, with output)
```

## Data sources

- [Disease–symptom dataset](https://github.com/anujdutt9/Disease-Prediction-from-Symptoms) (4,920 records)
- [UCI Heart Disease (Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease)
- [Pima Indians Diabetes](https://archive.ics.uci.edu/dataset/34/diabetes)
- [UCI Drug Reviews, drugs.com](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com) (215K reviews)

## Possible next steps

Research-grade diagnosis data (DDXPlus) for the disease model, contextual bandits with off-policy evaluation once there's real traffic, and collaborative filtering when per-user history accumulates. Database backups and monitoring for the hosted instance.

## Author

Built by **Tanmay Patel** — data analysis, model training, recommendation engine, web app, API, database and deployment.

GitHub: [@tanmay866](https://github.com/tanmay866)

---

*This is an educational project, not a medical device. Nothing it outputs is medical advice — medication references are drug classes, not prescriptions. Consult a qualified doctor for real health concerns.*
