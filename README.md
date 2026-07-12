# Personalized Healthcare & Medicine Recommendation System

A machine-learning web application that predicts a likely disease from your symptoms, recommends medicines backed by real patient reviews, and screens heart-disease and diabetes risk using models trained on real clinical data. Recommendations improve over time from user feedback.

**Live demo:** https://personalized-healthcare-recommendation-system.streamlit.app — sign up free and try it.

![Python](https://img.shields.io/badge/python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-models-orange)
![Streamlit](https://img.shields.io/badge/streamlit-app-red)
![FastAPI](https://img.shields.io/badge/fastapi-REST%20API-teal)
![PostgreSQL](https://img.shields.io/badge/postgresql-neon-blue)

---

## Contents

- [Overview](#overview)
- [Features](#features)
- [Screenshots](#screenshots)
- [How it works](#how-it-works)
- [Models and results](#models-and-results)
- [Getting started](#getting-started)
- [REST API](#rest-api)
- [Testing](#testing)
- [Data sources](#data-sources)
- [Roadmap](#roadmap)
- [Author](#author)

---

## Overview

The flow is simple from the user's side:

1. **Select your symptoms** → the app predicts the most likely disease (out of 41) with confidence scores
2. **Get a full care plan** → medicine suggestions, precautions, diet, lifestyle advice, and the right specialist to consult
3. **See what worked for others** → real medicines ranked by sentiment analysis of 215K patient reviews
4. **Vote on recommendations** → 👍/👎 feedback re-ranks medicines for everyone (reinforcement learning)
5. **Check your risk** → heart-disease and diabetes calculators trained on real patient records

Behind that sits a full product: user accounts with roles, a hosted PostgreSQL database, a JWT-secured REST API, and 48 automated tests.

---

## Features

| Feature | What it does |
|---------|--------------|
| Disease prediction | 132 symptoms → 41 diseases (Random Forest, top-3 with confidence) |
| Care recommendations | Curated per-disease guidance: medications, precautions, diet, specialist |
| Evidence-ranked medicines | NLP sentiment over 215K real drug reviews, blended with star ratings |
| Adaptive ranking (RL) | Thompson-sampling bandit re-ranks medicines from user feedback |
| Knowledge graph | 307-node medical graph; PageRank finds related diseases via multi-hop paths |
| Clinical risk calculators | Heart (UCI Cleveland) and diabetes (Pima) — real patient data |
| Sentiment explorer | Browse drug sentiment per condition; analyze any review text live |
| Analytics dashboard | Usage trends, popular diseases, model metrics; admin sees all activity |
| User management | Signup/login, salted-hash passwords, roles, profiles, persistent sessions |
| REST API | FastAPI + JWT, 13 endpoints, Swagger docs |

---

## Screenshots

| | |
|---|---|
| **Login** ![Login](docs/screenshots/01_login.png) | **Disease prediction** ![Prediction](docs/screenshots/02_prediction.png) |
| **Care recommendations** ![Recommendations](docs/screenshots/03_recommendations.png) | **Medicine ranking + feedback** ![Adaptive medicines](docs/screenshots/04_adaptive_medicines.png) |
| **Knowledge graph** ![Knowledge graph](docs/screenshots/05_knowledge_graph.png) | **Clinical risk calculators** ![Clinical risk](docs/screenshots/06_risk_screening.png) |
| **Sentiment explorer** ![Sentiment explorer](docs/screenshots/07_sentiment_explorer.png) | **Analytics dashboard** ![Dashboard](docs/screenshots/08_dashboard.png) |

---

## How it works

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

The web app and the REST API share one inference layer and one database.

- **Prediction** — symptoms become a 132-dim binary vector; a Random Forest classifies it into one of 41 diseases
- **Recommendation** — a curated knowledge base supplies care guidance; the NLP sentiment model ranks real medicines; the bandit adjusts that ranking from feedback
- **Discovery** — cosine similarity plus knowledge-graph PageRank suggest related diseases
- **Storage** — SQLAlchemy Core; SQLite locally, PostgreSQL in production, switched by one `DATABASE_URL` variable

---

## Models and results

| Model | Data | Algorithm | Test result | Baseline |
|-------|------|-----------|-------------|----------|
| Disease predictor | 4,920 records, 41 classes | Random Forest | 100% | 2.4% |
| Heart-disease risk | 303 real patients (UCI Cleveland) | Logistic Regression | AUC 0.95, acc 87% | 54% |
| Diabetes risk | 768 real patients (Pima Indians) | Logistic Regression | AUC 0.81, acc 71% | 65% |
| Outcome screening | symptoms + vitals | Random Forest (tuned) | 80% | 52% |
| Review sentiment | 215K drug reviews | TF-IDF + Logistic Regression | 90%, F1 0.93 | 72% |

A few of these numbers deserve honest context:

**The 100% is a property of the dataset, not brilliance.** The disease-symptom dataset is cleanly separable, so every classifier I tried (RF, SVM, NB, XGBoost, an MLP, a Keras model) lands at or near 100%. The value is the complete working pipeline, not the benchmark.

**The clinical models are the honest ones.** Real patient data is messy — Pima famously encodes missing measurements as zeros (374 rows show an impossible insulin of 0). These are treated as missing and imputed inside the sklearn pipeline, so the deployed calculators tolerate partly-filled forms too. Logistic regression beat the tree ensembles on both clinical datasets, which is the usual outcome on a few hundred records.

**One target was unlearnable.** I originally tried to predict a `risk_level` column, but every model sat at the majority baseline — no signal. I switched to the outcome target (80% vs 52% baseline) and documented the dead end. I also evaluated probability calibration and kept the uncalibrated model: isotonic traded 5 points of accuracy for a 0.001 Brier improvement. Details in `notebooks/02_modeling.ipynb`.

---

## Getting started

```bash
git clone https://github.com/tanmay866/personalized-healthcare-recommendation-system.git
cd personalized-healthcare-recommendation-system

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

streamlit run app/app.py        # → http://localhost:8501
```

Trained models ship with the repo, so this works immediately — sign up inside the app and explore.

**Optional:**

- **Retrain models** — scripts in `src/`: `train_disease.py`, `train_clinical.py`, `train_risk.py` (the sentiment model needs a 112 MB download, documented in `train_sentiment.py`)
- **Use PostgreSQL** — `export DATABASE_URL="postgresql://user:pass@host:5432/db"` — nothing else changes; the live demo runs on hosted Neon PostgreSQL
- **Production secrets** — `DATABASE_URL`, `ADMIN_PASSWORD`, `API_JWT_SECRET`

<details>
<summary><b>Project structure</b></summary>

```
app/app.py                  Streamlit app (auth, 5 tabs, persistent login)
api/main.py                 FastAPI backend (JWT)
src/
  preprocess.py             cleaning + feature engineering
  train_disease.py          disease model + similarity artifact
  train_clinical.py         heart + diabetes models (real clinical data)
  train_risk.py             outcome screening (tuning + calibration analysis)
  train_sentiment.py        NLP sentiment on drug reviews
  train_deep.py             Keras comparison model
  build_knowledge_base.py   41-disease recommendation knowledge base
  knowledge_graph.py        medical graph + Personalized PageRank
  bandit.py                 Thompson-sampling feedback bandit
  clinical.py               clinical risk inference
  recommend.py              shared inference layer
  auth.py / db.py           users, roles, activity; SQLAlchemy storage
data/                       datasets + processed artifacts
models/                     trained models and metrics
notebooks/                  01_eda, 02_modeling (executed, with output)
```
</details>

---

## REST API

Everything is also available as a JWT-secured REST service:

```bash
uvicorn api.main:app --port 8000    # Swagger docs at /docs
```

<details>
<summary><b>Endpoints and example</b></summary>

| Method | Endpoint | Auth | Purpose |
|--------|----------|------|---------|
| POST | `/auth/signup` | — | Create an account |
| POST | `/auth/login` | — | Get a JWT (24h) |
| GET | `/symptoms` | — | The 132 symptoms the model understands |
| POST | `/predict/disease` | JWT | Symptoms → disease, recommendations, medicines |
| POST | `/predict/risk` | JWT | Vitals → outcome likelihood |
| POST | `/predict/clinical/{heart\|diabetes}` | JWT | Clinical risk calculators |
| GET | `/recommend/{disease}` | JWT | Knowledge-base entry |
| GET | `/sentiment/{condition}` | JWT | Top drugs by review sentiment |
| GET | `/graph/{disease}` | JWT | Graph neighborhood + related diseases |
| GET | `/medicines/{disease}` | JWT | Bandit-ranked medicines |
| POST | `/feedback` | JWT | 👍/👎 — updates the bandit |
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

---

## Testing

- **29 API checks** — auth flows, JWT validation, role enforcement, all ML endpoints, input validation, bandit re-ranking after feedback
- **19 browser end-to-end checks** (Playwright) — login through prediction, feedback, dashboards, logout
- Both suites pass on SQLite **and** PostgreSQL backends

---

## Data sources

- [Disease–symptom dataset](https://github.com/anujdutt9/Disease-Prediction-from-Symptoms) — 4,920 records
- [UCI Heart Disease (Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease) — 303 patients
- [Pima Indians Diabetes](https://archive.ics.uci.edu/dataset/34/diabetes) — 768 patients
- [UCI Drug Reviews (drugs.com)](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com) — 215K reviews

## Roadmap

Research-grade diagnosis data (DDXPlus) for the disease model, contextual bandits once there's real traffic, collaborative filtering when per-user history accumulates, and backups/monitoring for the hosted database.

## Author

Built by **Tanmay Patel** — data analysis, model training, recommendation engine, web app, API, database and deployment.

GitHub: [@tanmay866](https://github.com/tanmay866)

---

*This is an educational project, not a medical device. Nothing it outputs is medical advice — consult a qualified doctor for real health concerns.*
