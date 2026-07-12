# 🩺 Personalized Healthcare & Medicine Recommendation System

A machine-learning system that predicts a likely **disease from a patient's symptoms**, recommends **medicines, precautions, diet, lifestyle changes, and the right specialist**, and provides a **personalized health-risk screening** from patient vitals.

> Built as a Data Science / ML internship project for **Zidio Development**.

---

## ✨ Features

| # | Feature | What it does |
|---|---------|--------------|
| 1 | **User Management** | Signup/login with salted-hash passwords stored in a **SQLite database**, Admin/User roles, and per-user health profiles that prefill the risk form. |
| 2 | **Disease Prediction** | Enter your symptoms → ML model predicts the most likely disease (with confidence & top-3 alternatives). |
| 3 | **Care Recommendations** | For the predicted disease: medicines, precautions, diet, workout/lifestyle tips, and which specialist to consult. |
| 4 | **Content-Based Filtering** | "Related diseases" via cosine similarity between disease symptom profiles. |
| 5 | **Hybrid Medicine Ranking** | Real medicines ranked by a hybrid score: 60% NLP review sentiment (crowd signal) + 40% star rating (content signal), from 200K+ drugs.com reviews. |
| 6 | **Health Risk Screening** | Symptoms + vitals (age, gender, blood pressure, cholesterol) → likelihood of a positive diagnosis. |
| 7 | **Sentiment Explorer (NLP)** | Per-condition drug sentiment rankings + a live review analyzer powered by the trained NLP model. |
| 8 | **Analytics Dashboard** | Usage trends, disease popularity rankings, model performance comparison, dataset insights; admins see all user activity. |

---

## 🧠 How it works — three models

This project deliberately uses **two complementary models**, each backed by a suitable dataset:

### Model 1 — Symptom → Disease Predictor
- **Dataset:** 4,920 records · 132 symptoms · 41 diseases (perfectly balanced).
- **Approach:** symptoms encoded as a 132-dim binary vector → multi-class classification.
- **Models compared:** Random Forest, SVM, Naive Bayes, XGBoost, and a **neural network** (MLP with 128→64 hidden layers) — 5-fold stratified CV.
- **Result:** **100% test accuracy** (Random Forest selected).
- Also produces the **content-based filtering** artifact: cosine similarity between per-disease mean symptom vectors → "related diseases".
  - *Note:* this dataset is cleanly separable — every disease maps to a consistent symptom signature — so near-perfect accuracy is expected and is a property of the data, not overfitting. The engineering value here is the **complete, deployed end-to-end system**, not beating a hard benchmark.

### Model 2 — Personalized Risk / Outcome Screening
- **Dataset:** patient-profile data (symptoms + age, gender, blood pressure, cholesterol).
- **Target:** `outcome_variable` (Positive / Negative diagnosis).
  - We tested predicting `risk_level` (Low/Med/High) but it carries **no learnable signal** (models sit at the majority-class baseline), so we transparently pivoted to `outcome_variable`, which does.
- **Models compared:** Random Forest, Gradient Boosting, Logistic Regression, MLP — plus a **GridSearchCV tuning pass** on the winner.
- **Result:** **80% test accuracy** vs a **52% majority-class baseline** (tuned Random Forest) — a genuine, honest improvement.

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
│   ├── build_knowledge_base.py # authors the recommendation KB
│   ├── auth.py                 # users, roles, profiles, activity log
│   ├── db.py                   # SQLite storage layer (users + activity)
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

---

## 🔮 Future enhancements

- ~~REST API backend with JWT-token authentication~~ ✅ **Done** — see the REST API section above. Remaining: a hosted PostgreSQL database (current storage is SQLite through a plain-SQL layer, so the swap is a connection change, not a rewrite).
- Larger, real-world clinical datasets and probability calibration.
- **Graph-based recommendations** (medical knowledge graphs) and **reinforcement learning** from user feedback.
- TensorFlow/PyTorch deep-learning models (an sklearn MLP neural network is already included in the comparison).

---

## ⚠️ Disclaimer

This project is for **educational and demonstration purposes only**. It is **not** a medical device and must **not** be used for real diagnosis or treatment. Medication references are general drug *classes*, not prescriptions. Always consult a qualified healthcare professional.
