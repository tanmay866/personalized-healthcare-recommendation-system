# 🩺 Personalized Healthcare & Medicine Recommendation System

A machine-learning system that predicts a likely **disease from a patient's symptoms**, recommends **medicines, precautions, diet, lifestyle changes, and the right specialist**, and provides a **personalized health-risk screening** from patient vitals.

> Built as a Data Science / ML internship project for **Zidio Development**.

---

## ✨ Features

| # | Feature | What it does |
|---|---------|--------------|
| 1 | **Disease Prediction** | Enter your symptoms → ML model predicts the most likely disease (with confidence & top-3 alternatives). |
| 2 | **Care Recommendations** | For the predicted disease, get medicines, precautions, diet, workout/lifestyle tips, and which specialist to consult. |
| 3 | **Health Risk Screening** | Enter symptoms + vitals (age, gender, blood pressure, cholesterol) → model estimates the likelihood of a positive diagnosis. |
| 4 | **Interactive Dashboard** | A clean Streamlit web app with charts (confidence bars, a risk gauge) and model performance metrics. |

---

## 🧠 How it works — two models

This project deliberately uses **two complementary models**, each backed by a suitable dataset:

### Model 1 — Symptom → Disease Predictor
- **Dataset:** 4,920 records · 132 symptoms · 41 diseases (perfectly balanced).
- **Approach:** symptoms encoded as a 132-dim binary vector → multi-class classification.
- **Models compared:** Random Forest, SVM, Naive Bayes, XGBoost (5-fold stratified CV).
- **Result:** **100% test accuracy** (Random Forest selected).
  - *Note:* this dataset is cleanly separable — every disease maps to a consistent symptom signature — so near-perfect accuracy is expected and is a property of the data, not overfitting. The engineering value here is the **complete, deployed end-to-end system**, not beating a hard benchmark.

### Model 2 — Personalized Risk / Outcome Screening
- **Dataset:** patient-profile data (symptoms + age, gender, blood pressure, cholesterol).
- **Target:** `outcome_variable` (Positive / Negative diagnosis).
  - We tested predicting `risk_level` (Low/Med/High) but it carries **no learnable signal** (models sit at the majority-class baseline), so we transparently pivoted to `outcome_variable`, which does.
- **Models compared:** Random Forest, Gradient Boosting, Logistic Regression.
- **Result:** **~77% test accuracy** vs a **52% majority-class baseline** (Random Forest) — a genuine, honest improvement.

> **Honesty note:** rather than force one weak dataset to do everything, each model is matched to a dataset that can actually support it. Recognizing and documenting this trade-off is part of the project.

---

## 🗂️ Project structure

```
personalized-healthcare-recommendation-system/
├── app/
│   └── app.py                  # Streamlit web app (2 tabs)
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
├── src/
│   ├── preprocess.py           # data cleaning & feature engineering
│   ├── train_disease.py        # trains + evaluates disease model
│   ├── train_risk.py           # trains + evaluates risk model
│   ├── build_knowledge_base.py # authors the recommendation KB
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

---

## 🛠️ Tech stack

- **Python** · pandas · NumPy
- **scikit-learn** · **XGBoost** (modeling)
- **Streamlit** (web app) · **Plotly** (charts)
- **joblib** (model persistence)

---

## 📊 Results summary

| Model | Task | Best algorithm | Accuracy | Baseline |
|-------|------|----------------|----------|----------|
| Disease predictor | 41-class symptom → disease | Random Forest | **100%** | 2.4% (random) |
| Risk screener | Positive/Negative outcome | Random Forest | **~77%** | 52% (majority) |

---

## 🔮 Future enhancements

- REST API backend (Flask / FastAPI) with **JWT authentication** and role-based access.
- **NLP sentiment analysis** on real drug-review data to rank recommended medicines.
- Larger, real-world clinical datasets and model calibration.
- Persistent user profiles and history tracking.
- Deep-learning / graph-based recommendation modules.

---

## ⚠️ Disclaimer

This project is for **educational and demonstration purposes only**. It is **not** a medical device and must **not** be used for real diagnosis or treatment. Medication references are general drug *classes*, not prescriptions. Always consult a qualified healthcare professional.
