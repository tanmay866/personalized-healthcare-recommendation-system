<div align="center">

# 🩺 Personalized Healthcare & Medicine Recommendation System

**An end-to-end machine-learning platform** — predicts diseases from symptoms, recommends medicines with real-world evidence, screens health risk with models trained on real clinical data, and learns from user feedback.

[![Live Demo](https://img.shields.io/badge/🔴_Live_Demo-Try_it_now-FF4B4B?style=for-the-badge)](https://personalized-healthcare-recommendation-system.streamlit.app)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_+_JWT-009688?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Neon-4169E1?logo=postgresql&logoColor=white)

*5 ML models · 3 real-world datasets · NLP on 215K reviews · knowledge graph · reinforcement learning · 48 automated tests*

</div>

---

## 🎯 What it does

Enter your symptoms → the system tells you the **likely disease**, what **medicines real patients rate highest** for it, which **precautions and diet** to follow, and which **specialist** to consult. Separately, calculators trained on **real clinical data** estimate your heart-disease and diabetes risk. Every recommendation improves as users give 👍/👎 feedback.

| Capability | How |
|---|---|
| 🔬 **Disease prediction** | 132 symptoms → 41 diseases, RandomForest, top-3 with confidence |
| 💊 **Care recommendations** | Curated knowledge base: medicines, precautions, diet, lifestyle, specialist |
| 🏆 **Evidence-ranked medicines** | NLP sentiment over **215K real drug reviews** + star ratings (hybrid score) |
| 🎰 **Learns from feedback (RL)** | Thompson-sampling bandit re-ranks medicines as users vote 👍/👎 |
| 🕸 **Knowledge-graph discovery** | 307-node medical graph; Personalized PageRank finds related diseases via multi-hop paths |
| ❤️🩸 **Clinical risk calculators** | Trained on **real patients**: UCI Cleveland heart (AUC 0.95), Pima diabetes (AUC 0.81) |
| 💬 **Sentiment explorer** | Per-condition drug rankings + live review analyzer (try your own text) |
| 📈 **Analytics dashboard** | Usage trends, disease popularity, model metrics; admin sees all activity |
| 🔐 **Real user management** | Salted-hash auth, roles, health profiles, persistent login, SQLite→PostgreSQL via one env var |
| 🔌 **REST API** | FastAPI + JWT, 13 endpoints, auto-generated Swagger docs |

---

## 📸 Screenshots

| | |
|---|---|
| **Login & landing** ![Login](docs/screenshots/01_login.png) | **Disease prediction** ![Prediction](docs/screenshots/02_prediction.png) |
| **Care recommendations** ![Recommendations](docs/screenshots/03_recommendations.png) | **Adaptive medicines (RL feedback)** ![Adaptive medicines](docs/screenshots/04_adaptive_medicines.png) |
| **Knowledge graph** ![Knowledge graph](docs/screenshots/05_knowledge_graph.png) | **Clinical risk calculators (real patient data)** ![Clinical risk calculators](docs/screenshots/06_risk_screening.png) |
| **Sentiment explorer (NLP)** ![Sentiment explorer](docs/screenshots/07_sentiment_explorer.png) | **Analytics dashboard** ![Dashboard](docs/screenshots/08_dashboard.png) |

---

## 🏗️ Architecture

```mermaid
flowchart LR
    U[👤 User] --> APP[Streamlit App<br/>auth · 5 tabs · cookies]
    U2[🔧 API Client] --> API[FastAPI + JWT<br/>13 REST endpoints]
    APP --> INF[Inference Layer]
    API --> INF
    INF --> M1[Disease Predictor<br/>RandomForest · 100%]
    INF --> M2[Clinical Risk Models<br/>heart AUC 0.95 · diabetes AUC 0.81]
    INF --> M3[NLP Sentiment<br/>TF-IDF + LogReg · 90%]
    INF --> KG[Knowledge Graph<br/>307 nodes · PageRank]
    INF --> RL[RL Bandit<br/>Thompson sampling]
    INF --> KB[(Knowledge Base<br/>41 diseases)]
    APP --> DB[(SQLite / PostgreSQL<br/>users · activity · feedback)]
    API --> DB
    RL --> DB
```

---

## 🧠 The models — and the reasoning behind them

### 1 · Symptom → Disease Predictor
4,920 records · 132 binary symptom features · 41 balanced classes. Compared RandomForest, SVM, Naive Bayes, XGBoost and an MLP neural network (5-fold stratified CV) — **100% test accuracy** (RandomForest deployed). A TensorFlow/Keras model (`Embedding → Dense(128) → Dense(64)`) matches it; the app ships RandomForest to avoid a 500 MB dependency for zero gain.

> ⚖️ *This dataset is cleanly separable, so near-perfect accuracy is a property of the data — the engineering value is the complete deployed system, not the benchmark.*

### 2 · Clinical Risk Calculators — real patient data
- **❤️ Heart disease** — UCI Cleveland, **303 real patients**: Logistic Regression, **test AUC 0.95 · 87% accuracy** (54% baseline)
- **🩸 Diabetes** — Pima Indians, **768 real patients**: Logistic Regression, **test AUC 0.81 · 71% accuracy** (65% baseline)
- Real-data messiness handled inside the sklearn Pipeline: Cleveland's missing `ca`/`thal` values and Pima's zeros-as-hidden-missing (374 impossible insulin readings) are median-imputed — inference tolerates partial inputs too
- **Logistic Regression beat RandomForest and GradientBoosting on both** — the classic small-clinical-data result

### 3 · General Outcome Screening
Symptoms + vitals → positive/negative outcome likelihood: tuned RandomForest, **80% test accuracy vs 52% baseline**. The original `risk_level` target proved statistically unlearnable (models stuck at the majority baseline) — verified and documented, then pivoted to the learnable target. Calibration was evaluated (Brier/reliability curves in `02_modeling.ipynb`); the raw model was already well-calibrated, so calibrated variants that traded accuracy for nothing were rejected.

### 4 · Drug-Review Sentiment (NLP)
**215K real reviews** from drugs.com → TF-IDF (uni+bigrams, 50K features) + Logistic Regression: **90% test accuracy, 0.93 F1** on ~49K held-out reviews. Every review is scored and aggregated per (drug, condition) to rank real medicines by patient satisfaction.

### 5 · Recommendation intelligence
- **Content-based:** cosine similarity between disease symptom profiles
- **Graph-based:** Personalized PageRank over a 307-node medical knowledge graph — finds relations through shared specialists/medications that vector similarity can't see
- **Reinforcement learning:** each (disease, medicine) pair is a bandit arm with a Beta posterior anchored at its offline hybrid score; 👍/👎 votes update the posterior and re-rank via Thompson sampling

> 🧭 **Design principle:** every model is matched to a dataset that can actually support it, every metric is reported against its baseline, and negative results (unlearnable targets, rejected calibration) are documented rather than hidden.

---

## 📊 Results at a glance

| Model | Data | Algorithm | Result | Baseline |
|-------|------|-----------|--------|----------|
| Disease predictor | 4,920 records · 41 classes | Random Forest | **100%** | 2.4% |
| ❤️ Heart risk | 303 real patients | Logistic Regression | **AUC 0.95 · 87%** | 54% |
| 🩸 Diabetes risk | 768 real patients | Logistic Regression | **AUC 0.81 · 71%** | 65% |
| Outcome screening | symptoms + vitals | Random Forest (tuned) | **80%** | 52% |
| Review sentiment | 215K drug reviews | TF-IDF + LogReg | **90% · F1 0.93** | 72% |
| Deep learning (comparison) | 4,920 records | Keras Embedding+Dense | **100%** | 2.4% |

**Quality assurance:** 29 API tests + 19 browser end-to-end tests (Playwright) covering login → prediction → feedback → dashboards → logout, verified on both SQLite and PostgreSQL backends.

---

## 🚀 Run it locally

```bash
git clone https://github.com/tanmay866/personalized-healthcare-recommendation-system.git
cd personalized-healthcare-recommendation-system

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

streamlit run app/app.py          # → http://localhost:8501
```

Sign up for an account in the app. Trained models ship with the repo; retrain any of them with the scripts in `src/` (`train_disease.py`, `train_clinical.py`, `train_risk.py`, `train_sentiment.py`).

<details>
<summary><b>🔌 REST API (FastAPI + JWT)</b></summary>

```bash
uvicorn api.main:app --port 8000    # interactive docs at /docs
```

| Method | Endpoint | Auth | Purpose |
|--------|----------|------|---------|
| POST | `/auth/signup` | — | Create an account |
| POST | `/auth/login` | — | Get a JWT access token (24h) |
| GET | `/symptoms` | — | The 132 symptoms the model understands |
| POST | `/predict/disease` | JWT | Symptoms → disease + recommendations + related + medicines |
| POST | `/predict/risk` | JWT | Vitals → outcome likelihood |
| POST | `/predict/clinical/{heart\|diabetes}` | JWT | Risk from real-clinical-data models |
| GET | `/recommend/{disease}` | JWT | Knowledge-base entry |
| GET | `/sentiment/{condition}` | JWT | Top drugs by review sentiment |
| GET | `/graph/{disease}` | JWT | Knowledge-graph neighborhood + related diseases |
| GET | `/medicines/{disease}` | JWT | Bandit-ranked medicine recommendations |
| POST | `/feedback` | JWT | 👍/👎 feedback — trains the RL bandit |
| GET | `/admin/users` | JWT · Admin | Role-guarded user list |
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

<details>
<summary><b>🗄️ Database: SQLite → PostgreSQL with one variable</b></summary>

Storage uses SQLAlchemy Core, so the same code runs on both backends:

- **SQLite** (default, zero setup) — `data/app.db`, created automatically
- **PostgreSQL** — `export DATABASE_URL="postgresql://user:pass@host:5432/db"` — no code changes

The live demo runs on a hosted **Neon PostgreSQL**, so accounts, history and feedback persist across restarts. On Streamlit Cloud, set `DATABASE_URL` in the app Secrets (auto-bridged to the environment).

Production secrets: `DATABASE_URL` · `ADMIN_PASSWORD` · `API_JWT_SECRET`.
</details>

<details>
<summary><b>🗂️ Project structure</b></summary>

```
├── app/app.py                  # Streamlit app (5 tabs, auth, persistent login)
├── api/main.py                 # FastAPI REST backend (JWT)
├── src/
│   ├── preprocess.py           # data cleaning & feature engineering
│   ├── train_disease.py        # disease model + similarity artifact
│   ├── train_clinical.py       # heart + diabetes (real clinical data)
│   ├── train_risk.py           # outcome screening (+ tuning, calibration)
│   ├── train_sentiment.py      # NLP sentiment on 215K drug reviews
│   ├── train_deep.py           # TensorFlow/Keras comparison
│   ├── build_knowledge_base.py # 41-disease recommendation KB
│   ├── knowledge_graph.py      # medical graph + PageRank
│   ├── bandit.py               # RL: Thompson-sampling feedback bandit
│   ├── clinical.py             # clinical risk inference + UI specs
│   ├── recommend.py            # inference layer
│   ├── auth.py                 # users, roles, profiles, activity
│   └── db.py                   # SQLite/PostgreSQL via SQLAlchemy Core
├── data/                       # datasets (raw + processed KB/sentiment)
├── models/                     # trained models, metrics, artifacts
├── notebooks/                  # 01_eda · 02_modeling (executed, with charts)
└── docs/screenshots/
```
</details>

---

## 🛠️ Tech stack

`Python 3.11` · `pandas` · `scikit-learn` · `XGBoost` · `TensorFlow/Keras` · `NetworkX` · `Streamlit` · `Plotly` · `FastAPI` · `PyJWT` · `SQLAlchemy` · `PostgreSQL (Neon)` · `Playwright` (testing)

**Datasets:** [Disease–Symptom](https://github.com/anujdutt9/Disease-Prediction-from-Symptoms) · [UCI Cleveland Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) · [Pima Indians Diabetes](https://archive.ics.uci.edu/dataset/34/diabetes) · [UCI Drug Reviews (drugs.com)](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com)

---

## 🔮 Roadmap

- Research-grade diagnosis data (DDXPlus, 1.3M synthetic-clinical patients) for the symptom→disease model
- Contextual bandits + off-policy evaluation on live traffic
- Collaborative filtering (user–item SVD) once per-user interaction history accumulates
- Database backups/monitoring and PyTorch model variants

---

## 👨‍💻 Author

**Tanmay Patel** — designed and built end-to-end: data analysis, model training, recommendation engine, web app, REST API, database and deployment.

[![GitHub](https://img.shields.io/badge/GitHub-tanmay866-181717?logo=github)](https://github.com/tanmay866)

© 2026 Tanmay Patel · All rights reserved

---

## ⚠️ Disclaimer

This project is for **educational and demonstration purposes only**. It is **not** a medical device and must not be used for real diagnosis or treatment. Medication references are general drug classes, not prescriptions. Always consult a qualified healthcare professional.
