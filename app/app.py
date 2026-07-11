"""
Personalized Healthcare & Medicine Recommendation System — Streamlit app.

Two tools in one interface:
  1. Disease Prediction  — enter symptoms -> predicted disease + medicine /
     precaution / diet / workout / specialist recommendations.
  2. Health Risk Screening — enter symptoms + vitals -> likelihood of a
     positive diagnosis (personalized outcome model).

Run:  streamlit run app/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

# Make src/ importable regardless of where Streamlit is launched from.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recommend import (  # noqa: E402
    condition_sentiment,
    get_drug_sentiment,
    get_recommendation,
    humanize,
    list_sentiment_conditions,
    list_symptoms,
    predict_disease,
    predict_risk,
    sentiment_conditions_for,
)

MODELS = ROOT / "models"

st.set_page_config(
    page_title="Healthcare Recommendation System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      .main-title { font-size: 2.3rem; font-weight: 800; margin-bottom: 0; }
      .subtitle { color: #6b7280; margin-top: 0.2rem; font-size: 1rem; }
      .rec-card {
          background: rgba(37, 99, 235, 0.06);
          border-left: 4px solid #2563eb;
          border-radius: 8px; padding: 0.9rem 1.1rem; margin-bottom: 0.8rem;
      }
      .rec-card h4 { margin: 0 0 0.4rem 0; font-size: 1rem; }
      .pill {
          display:inline-block; background:#2563eb; color:#fff;
          padding:2px 10px; border-radius:999px; font-size:0.8rem; margin:2px;
      }
      .disclaimer {
          background: rgba(220, 38, 38, 0.08); border: 1px solid rgba(220,38,38,0.3);
          border-radius: 8px; padding: 0.7rem 1rem; font-size: 0.85rem; color:#b91c1c;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def _metrics() -> dict:
    out = {}
    for name in ["disease_metrics.json", "risk_metrics.json", "sentiment_metrics.json"]:
        p = MODELS / name
        if p.exists():
            out[name] = json.loads(p.read_text())
    return out


def _rec_card(title: str, items, is_list: bool = True) -> None:
    if not items:
        return
    if is_list and isinstance(items, str):
        items = [i.strip() for i in items.split(";") if i.strip()]
    if is_list:
        body = "".join(f'<span class="pill">{i}</span>' for i in items)
    else:
        body = f"<p style='margin:0'>{items}</p>"
    st.markdown(
        f'<div class="rec-card"><h4>{title}</h4>{body}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("🩺 About")
    st.write(
        "A two-model ML system: **symptom → disease** prediction with care "
        "recommendations, plus a **personalized risk screening** from vitals."
    )
    m = _metrics()
    if "disease_metrics.json" in m:
        dm = m["disease_metrics.json"]
        st.metric("Disease model accuracy", f"{dm['test_accuracy']*100:.0f}%")
        st.caption(f"{dm['best_model']} · {dm['n_diseases']} diseases · {dm['n_symptoms']} symptoms")
    if "risk_metrics.json" in m:
        rm = m["risk_metrics.json"]
        st.metric("Risk model accuracy", f"{rm['test_accuracy']*100:.0f}%")
        st.caption(f"{rm['best_model']} · baseline {rm['majority_baseline']*100:.0f}%")
    if "sentiment_metrics.json" in m:
        sm = m["sentiment_metrics.json"]
        st.metric("Sentiment model accuracy", f"{sm['test_accuracy']*100:.0f}%")
        st.caption(f"NLP · {sm['train_reviews']:,} reviews · F1 {sm['test_f1']:.2f}")
    st.divider()
    st.markdown(
        '<div class="disclaimer">⚠️ <b>Disclaimer:</b> Educational demo only. '
        "Not a substitute for professional medical advice. Always consult a "
        "qualified doctor.</div>",
        unsafe_allow_html=True,
    )


st.markdown('<p class="main-title">Personalized Healthcare & Medicine Recommendation System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Machine-learning powered disease prediction, care recommendations, and risk screening.</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(
    [
        "🔬  Disease Prediction & Recommendations",
        "📊  Health Risk Screening",
        "💬  Medicine Sentiment Explorer",
    ]
)

# --------------------------------------------------------------------------- #
# Tab 1 — disease prediction
# --------------------------------------------------------------------------- #
with tab1:
    st.subheader("Select your symptoms")
    all_symptoms = list_symptoms()
    labels = {humanize(s): s for s in all_symptoms}

    chosen_labels = st.multiselect(
        "Start typing to search symptoms:",
        options=sorted(labels.keys()),
        help="Pick every symptom you're experiencing for the most accurate prediction.",
    )
    chosen = [labels[l] for l in chosen_labels]

    if st.button("🔍 Predict Disease", type="primary", width='stretch'):
        if len(chosen) == 0:
            st.warning("Please select at least one symptom.")
        else:
            result = predict_disease(chosen, top_k=3)
            disease = result["disease"]
            conf = result["confidence"]

            c1, c2 = st.columns([1, 1])
            with c1:
                st.success(f"### 🧬 {disease}")
                st.metric("Model confidence", f"{conf*100:.1f}%")
                rec = result["recommendation"]
                if rec.get("description"):
                    st.info(rec["description"])
                if rec.get("specialist"):
                    st.markdown(f"**👨‍⚕️ Consult:** {rec['specialist']}")
            with c2:
                top = result["top_k"]
                fig = go.Figure(
                    go.Bar(
                        x=[t["probability"] * 100 for t in top][::-1],
                        y=[t["disease"] for t in top][::-1],
                        orientation="h",
                        marker_color="#2563eb",
                        text=[f"{t['probability']*100:.1f}%" for t in top][::-1],
                        textposition="auto",
                    )
                )
                fig.update_layout(
                    title="Top predictions",
                    height=250,
                    margin=dict(l=10, r=10, t=40, b=10),
                    xaxis_title="Probability (%)",
                )
                st.plotly_chart(fig, width='stretch')

            st.divider()
            st.subheader("💡 Personalized Recommendations")
            rec = result["recommendation"]
            rc1, rc2 = st.columns(2)
            with rc1:
                _rec_card("💊 Medications", rec.get("medications"))
                _rec_card("🥗 Diet", rec.get("diet"))
                _rec_card("🏃 Workout / Lifestyle", rec.get("workout"), is_list=False)
            with rc2:
                _rec_card("🛡️ Precautions", rec.get("precautions"))

            # --- NLP feature: real medicines ranked by patient-review sentiment
            sent = get_drug_sentiment(disease, top_n=8)
            if sent is not None:
                st.divider()
                st.subheader("💬 Top-Rated Real Medicines (by patient sentiment)")
                st.caption(
                    f"Ranked by an NLP sentiment model trained on 200K+ real patient "
                    f"reviews (drugs.com), for: {', '.join(sentiment_conditions_for(disease))}."
                )
                sc1, sc2 = st.columns([1, 1])
                with sc1:
                    show = sent.rename(
                        columns={
                            "drugName": "Medicine",
                            "condition": "Condition",
                            "n_reviews": "Reviews",
                            "avg_rating": "Avg rating (/10)",
                            "sentiment_score": "Sentiment",
                        }
                    )[["Medicine", "Condition", "Reviews", "Avg rating (/10)", "Sentiment"]]
                    st.dataframe(show, width='stretch', hide_index=True)
                with sc2:
                    sfig = go.Figure(
                        go.Bar(
                            x=(sent["sentiment_score"] * 100)[::-1],
                            y=sent["drugName"][::-1],
                            orientation="h",
                            marker_color="#10b981",
                            text=[f"{v*100:.0f}%" for v in sent["sentiment_score"]][::-1],
                            textposition="auto",
                        )
                    )
                    sfig.update_layout(
                        title="Patient sentiment (% positive)",
                        height=320,
                        margin=dict(l=10, r=10, t=40, b=10),
                        xaxis_title="Positive sentiment (%)",
                    )
                    st.plotly_chart(sfig, width='stretch')

            st.markdown(
                '<div class="disclaimer">⚠️ These recommendations are general and '
                "educational. Medication names are drug classes, not prescriptions. "
                "Please consult a licensed physician before taking any action.</div>",
                unsafe_allow_html=True,
            )

# --------------------------------------------------------------------------- #
# Tab 2 — risk screening
# --------------------------------------------------------------------------- #
with tab2:
    st.subheader("Enter your health profile")
    st.caption(
        "This model estimates the **likelihood of a positive diagnosis** from your "
        "symptoms and vitals — a quick personalized screening."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=35)
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
    with col2:
        fever = st.checkbox("Fever")
        cough = st.checkbox("Cough")
        fatigue = st.checkbox("Fatigue")
        breathing = st.checkbox("Difficulty breathing")
    with col3:
        bp = st.select_slider("Blood pressure", ["Low", "Normal", "High"], value="Normal")
        chol = st.select_slider("Cholesterol level", ["Low", "Normal", "High"], value="Normal")

    level_map = {"Low": 0, "Normal": 1, "High": 2}
    if st.button("📈 Assess Risk", type="primary", width='stretch'):
        profile = {
            "fever": int(fever),
            "cough": int(cough),
            "fatigue": int(fatigue),
            "difficulty_breathing": int(breathing),
            "age": age,
            "gender": 1 if gender == "Male" else 0,
            "blood_pressure": level_map[bp],
            "cholesterol_level": level_map[chol],
        }
        res = predict_risk(profile)
        prob = res["probability"] * 100

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob,
                number={"suffix": "%"},
                title={"text": "Likelihood of positive diagnosis"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2563eb"},
                    "steps": [
                        {"range": [0, 40], "color": "#dcfce7"},
                        {"range": [40, 70], "color": "#fef9c3"},
                        {"range": [70, 100], "color": "#fee2e2"},
                    ],
                },
            )
        )
        gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(gauge, width='stretch')

        if res["outcome"] == "Positive":
            st.error(f"⚠️ Elevated risk — model predicts a **Positive** outcome ({prob:.1f}%). Consider consulting a doctor.")
        else:
            st.success(f"✅ Lower risk — model predicts a **Negative** outcome (positive likelihood {prob:.1f}%).")

        st.markdown(
            '<div class="disclaimer">⚠️ This screening is a statistical estimate on a '
            "small dataset, not a diagnosis. Consult a doctor for any health concern.</div>",
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------- #
# Tab 3 — medicine sentiment explorer (NLP)
# --------------------------------------------------------------------------- #
with tab3:
    st.subheader("Explore real patient sentiment about medicines")
    st.caption(
        "Powered by a TF-IDF + Logistic Regression sentiment model trained on "
        "200K+ patient reviews from drugs.com (90% test accuracy)."
    )

    conditions = list_sentiment_conditions()
    if not conditions:
        st.info("Sentiment data not available. Run `python src/train_sentiment.py` first.")
    else:
        default_ix = conditions.index("Migraine") if "Migraine" in conditions else 0
        cond = st.selectbox("Choose a condition:", conditions, index=default_ix)
        rows = condition_sentiment(cond, top_n=15)
        if rows is None or rows.empty:
            st.warning("No drugs with enough reviews for this condition.")
        else:
            e1, e2 = st.columns([1, 1])
            with e1:
                show = rows.rename(
                    columns={
                        "drugName": "Medicine",
                        "n_reviews": "Reviews",
                        "avg_rating": "Avg rating (/10)",
                        "sentiment_score": "Sentiment",
                        "total_useful": "Helpful votes",
                    }
                )[["Medicine", "Reviews", "Avg rating (/10)", "Sentiment", "Helpful votes"]]
                st.dataframe(show, width='stretch', hide_index=True, height=430)
            with e2:
                efig = go.Figure(
                    go.Scatter(
                        x=rows["n_reviews"],
                        y=rows["sentiment_score"] * 100,
                        mode="markers+text",
                        text=rows["drugName"],
                        textposition="top center",
                        textfont=dict(size=9),
                        marker=dict(
                            size=rows["avg_rating"] * 2.2,
                            color=rows["sentiment_score"] * 100,
                            colorscale="RdYlGn",
                            cmin=0,
                            cmax=100,
                            showscale=True,
                            colorbar=dict(title="Sent. %"),
                        ),
                    )
                )
                efig.update_layout(
                    title=f"Drugs for {cond}: sentiment vs. review volume",
                    xaxis_title="Number of reviews",
                    yaxis_title="Positive sentiment (%)",
                    height=430,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(efig, width='stretch')

    st.divider()
    st.subheader("🧪 Try the sentiment model live")
    st.caption("Type any medicine review and see how the NLP model scores it.")
    demo_text = st.text_area(
        "Review text:",
        placeholder="e.g. This medication worked wonders for my headaches, no side effects at all!",
        height=100,
    )
    if st.button("Analyze sentiment", width='stretch'):
        if not demo_text.strip():
            st.warning("Please enter a review first.")
        else:
            import joblib

            @st.cache_resource
            def _sentiment_model():
                return joblib.load(MODELS / "sentiment_model.pkl")

            model = _sentiment_model()
            proba = float(model.predict_proba([demo_text])[0][1])
            if proba >= 0.5:
                st.success(f"😊 **Positive** — {proba*100:.1f}% positive sentiment")
            else:
                st.error(f"😞 **Negative** — {proba*100:.1f}% positive sentiment")
            st.progress(proba)
