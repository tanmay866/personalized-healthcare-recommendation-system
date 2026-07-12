# 🩺 Personalized Healthcare & Medicine Recommendation System

**🔴 Live demo: [personalized-healthcare-recommendation-system.streamlit.app](https://personalized-healthcare-recommendation-system.streamlit.app)** — log in with `admin` / `admin123` or sign up.

A machine-learning system that predicts a likely **disease from a patient's symptoms**, recommends **medicines, precautions, diet, lifestyle changes, and the right specialist**, and provides a **personalized health-risk screening** from patient vitals.

> Built as a Data Science / ML internship project for **Zidio Development**.

---

## ✨ Features

| # | Feature | What it does |
|---|---------|--------------|
| 1 | **User Management** | Signup/login with salted-hash passwords in a real database — **SQLite by default, PostgreSQL via one `DATABASE_URL` env var** (SQLAlchemy Core) — Admin/User roles, per-user health profiles that prefill the risk form. |
| 2 | **Disease Prediction** | Enter your symptoms → ML model predicts the most likely disease (with confidence & top-3 alternatives). |
| 3 | **Care Recommendations** | For the predicted disease: medicines, precautions, diet, workout/lifestyle tips, and which specialist to consult. |
| 4 | **Content-Based Filtering** | "Related diseases" via cosine similarity between disease symptom profiles. |
| 4b | **Knowledge-Graph Recommendations** | A 307-node medical graph (diseases ↔ symptoms ↔ medications ↔ specialists); **Personalized PageRank** finds related diseases through multi-hop paths, with an interactive network visualization. |
| 5 | **Adaptive Medicine Ranking (RL)** | Real medicines start at a hybrid score (60% NLP review sentiment + 40% star rating, 200K+ reviews) and **adapt to 👍/👎 user feedback via a Thompson-sampling bandit** — Beta posteriors per (disease, medicine) arm. |
| 6 | **Health Risk Screening** | Symptoms + vitals (age, gender, blood pressure, cholesterol) → likelihood of a positive diagnosis. |
| 7 | **Sentiment Explorer (NLP)** | Per-condition drug sentiment rankings + a live review analyzer powered by the trained NLP model. |
| 8 | **Analytics Dashboard** | Usage trends, disease popularity rankings, model performance comparison, dataset insights; admins see all user activity. |

---

## 📸 Screenshots

| | |
|---|---|
| **Login & landing** ![Login](docs/screenshots/01_login.png) | **Disease prediction** ![Prediction](docs/screenshots/02_prediction.png) |
| **Care recommendations** ![Recommendations](docs/screenshots/03_recommendations.png) | **Adaptive medicines (RL feedback)** ![Adaptive medicines](docs/screenshots/04_adaptive_medicines.png) |
| **Knowledge graph** ![Knowledge graph](docs/screenshots/05_knowledge_graph.png) | **Risk screening** ![Risk screening](docs/screenshots/06_risk_screening.png) |
| **Sentiment explorer (NLP)** ![Sentiment explorer](docs/screenshots/07_sentiment_explorer.png) | **Analytics dashboard** ![Dashboard](docs/screenshots/08_dashboard.png) |

---

## 🧠 How it works — three models

This project deliberately uses **two complementary models**, each backed by a suitable dataset:

### Model 1 — Symptom → Disease Predictor
- **Dataset:** 4,920 records · 132 symptoms · 41 diseases (perfectly balanced).
- **Approach:** symptoms encoded as a 132-dim binary vector → multi-class classification.
- **Models compared:** Random Forest, SVM, Naive Bayes, XGBoost, and a **neural network** (MLP with 128→64 hidden layers) — 5-fold stratified CV.
- **Deep learning comparison:** a **TensorFlow/Keras** model (symptom-token `Embedding(32) → GlobalAvgPool → Dense(128) → Dense(64) → softmax`, `src/train_deep.py`) also reaches 100% — the deployed app keeps RandomForest (equal accuracy, no 500MB TF dependency).
- **Result:** **100% test accuracy** (Random Forest selected).
- Also produces the **content-based filtering** artifact: cosine similarity between per-disease mean symptom vectors → "related diseases".
  - *Note:* this dataset is cleanly separable — every disease maps to a consistent symptom signature — so near-perfect accuracy is expected and is a property of the data, not overfitting. The engineering value here is the **complete, deployed end-to-end system**, not beating a hard benchmark.

### Model 2 — Personalized Risk / Outcome Screening
- **Dataset:** patient-profile data (symptoms + age, gender, blood pressure, cholesterol).
- **Target:** `outcome_variable` (Positive / Negative diagnosis).
  - We tested predicting `risk_level` (Low/Med/High) but it carries **no learnable signal** (models sit at the majority-class baseline), so we transparently pivoted to `outcome_variable`, which does.
- **Models compared:** Random Forest, Gradient Boosting, Logistic Regression, MLP — plus a **GridSearchCV tuning pass** on the winner.
- **Result:** **80% test accuracy** vs a **52% majority-class baseline** (tuned Random Forest) — a genuine, honest improvement.
- **Probability calibration:** the app surfaces raw probabilities, so we evaluated sigmoid and isotonic calibration against the raw model (Brier score / reliability curves — see `02_modeling.ipynb`). The raw model was already well calibrated (Brier 0.14); calibrated variants traded accuracy for negligible gains, so we kept it — with the analysis documented.

### Model 3 — Drug-Review Sentiment (NLP)
- **Dataset:** UCI Drug Review dataset (drugs.com) — **215K patient reviews**, 3,400+ drugs, 800+ conditions. *(Not committed — 112 MB; `src/train_sentiment.py` documents the download. A 5K sample ships in `data/processed/` for exploration.)*
- **Approach:** ratings → binary sentiment labels (≥7 positive, ≤4 negative); **TF-IDF (uni+bi-grams, 50K features) → Logistic Regression**.
- **Result:** **90% test accuracy, 0.93 F1** on ~49K held-out reviews.
- **Usage:** every review is scored by the model, then aggregated per (drug, condition) into `data/processed/drug_sentiment.csv`. The app uses this to rank **real medicines by patient satisfaction** for 26 of the 41 predictable diseases, and offers a sentiment explorer + live review analyzer.

> **Honesty note:** rather than force one weak dataset to do everything, each model is matched to a dataset that can actually support it. Recognizing and documenting this trade-off is part of the project.

---

## 🗂️ Project structure

```
personalized-healthcare-recommendation-system/
├── app/
│   └── app.py                  # Streamlit web app (3 tabs)
├── data/
│   ├── raw/                    # source datasets
│   │   ├── disease_symptoms.csv
│   │   └── patient_profile.csv
│   └── processed/
│       ├── knowledge_base.csv  # 41 diseases → medicines/diet/precautions/…
│       └── knowledge_base.json
├── models/                     # trained models + metrics + artifacts
├── notebooks/
│   └── 01_eda.ipynb            # exploratory data analysis
├── api/
│   └── main.py                 # FastAPI REST backend (JWT auth)
├── src/
│   ├── preprocess.py           # data cleaning & feature engineering
│   ├── train_disease.py        # disease model + similarity artifact
│   ├── train_risk.py           # risk model (+ GridSearch tuning)
│   ├── train_sentiment.py      # NLP sentiment model on drug reviews
│   ├── train_deep.py           # TensorFlow/Keras comparison model
│   ├── build_knowledge_base.py # authors the recommendation KB
│   ├── auth.py                 # users, roles, profiles, activity log
│   ├── db.py                   # DB layer (SQLite/PostgreSQL via SQLAlchemy)
│   ├── knowledge_graph.py      # medical knowledge graph + PageRank recs
│   ├── bandit.py               # RL: Thompson-sampling feedback bandit
│   └── recommend.py            # inference layer used by the app
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

```bash
# 1. Clone and enter the project
cd personalized-healthcare-recommendation-system

# 2. Create a virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. (Optional) Reproduce the models from scratch
python src/build_knowledge_base.py
python src/train_disease.py
python src/train_risk.py

# 4. Launch the app
streamlit run app/app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

**Login:** create your own account via *Sign up*, or use the demo admin: `admin` / `admin123` (admins additionally see all-user activity in the Analytics Dashboard).

---

## 🔌 REST API (FastAPI + JWT)

The system is also exposed as a standalone REST service with JWT authentication:

```bash
uvicorn api.main:app --port 8000
# interactive docs: http://localhost:8000/docs
```

| Method | Endpoint | Auth | Purpose |
|--------|----------|------|---------|
| POST | `/auth/signup` | — | Create an account |
| POST | `/auth/login` | — | Get a JWT access token (24h) |
| GET | `/symptoms` | — | The 132 symptoms the model understands |
| POST | `/predict/disease` | JWT | Symptoms → disease + recommendations + related diseases + top real medicines |
| POST | `/predict/risk` | JWT | Vitals → outcome likelihood |
| GET | `/recommend/{disease}` | JWT | Knowledge-base entry for a disease |
| GET | `/sentiment/{condition}` | JWT | Top drugs for a condition by review sentiment |
| GET | `/graph/{disease}` | JWT | Knowledge-graph neighborhood + graph-walk related diseases |
| GET | `/medicines/{disease}` | JWT | Adaptive (bandit-ranked) medicine recommendations |
| POST | `/feedback` | JWT | 👍/👎 medicine feedback — trains the RL bandit |
| GET | `/admin/users` | JWT (Admin) | List users — role-based access control |
| GET | `/health` | — | Liveness probe |

Example:

```bash
TOKEN=$(curl -s -X POST localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' | jq -r .access_token)

curl -X POST localhost:8000/predict/disease \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"symptoms":["continuous_sneezing","chills","watering_from_eyes"]}'
```

Set `API_JWT_SECRET` in production (a dev fallback is used otherwise).

### Database backends

Storage uses SQLAlchemy Core, so the same code runs on:

- **SQLite** (default, zero setup): `data/app.db`, created automatically.
- **PostgreSQL** (hosted or local): set one variable — no code changes.

```bash
export DATABASE_URL="postgresql://user:pass@host:5432/dbname"
streamlit run app/app.py          # or uvicorn api.main:app
```

Free hosted options: [Neon](https://neon.tech) or [Supabase](https://supabase.com) — create a project, copy its connection string. On Streamlit Cloud, add `DATABASE_URL` in the app's **Secrets** instead (the app bridges `st.secrets` → env automatically). The full E2E suite passes identically on both backends.

---

## 🛠️ Tech stack

- **Python** · pandas · NumPy
- **scikit-learn** · **XGBoost** (modeling) · **TF-IDF NLP** (sentiment)
- **Streamlit** (web app) · **Plotly** (charts)
- **joblib** (model persistence)

---

## 📊 Results summary

| Model | Task | Best algorithm | Accuracy | Baseline |
|-------|------|----------------|----------|----------|
| Disease predictor | 41-class symptom → disease | Random Forest | **100%** | 2.4% (random) |
| Risk screener | Positive/Negative outcome | Random Forest | **~77%** | 52% (majority) |
| Review sentiment (NLP) | Positive/Negative review | TF-IDF + Logistic Regression | **90%** (F1 0.93) | 72% (majority) |
| Deep learning (comparison) | 41-class symptom → disease | Keras Embedding + Dense | **100%** | 2.4% (random) |

---

## 🔮 Future enhancements

- **Managed cloud database in production** — PostgreSQL support is built in (set `DATABASE_URL`, verified end-to-end); what remains is provisioning a managed instance with backups/monitoring for a real deployment.
- **Larger, real-world clinical datasets** — the current datasets are educational; production would need clinically validated data and re-evaluation.
- **Contextual bandits on live traffic** — the Thompson-sampling bandit adapts to feedback today; real deployments would add user-context features and off-policy evaluation.
- **Collaborative filtering (user–item SVD)** — needs per-user interaction history at scale; the current crowd signal comes from 200K+ drug reviews.
- **PyTorch model variants** and larger architectures on richer data.

---

## ⚠️ Disclaimer

This project is for **educational and demonstration purposes only**. It is **not** a medical device and must **not** be used for real diagnosis or treatment. Medication references are general drug *classes*, not prescriptions. Always consult a qualified healthcare professional.
