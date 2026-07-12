"""
Personalized Healthcare & Medicine Recommendation System — Streamlit app.

Features
--------
* User management: signup/login (salted-hash passwords), Admin/User roles,
  per-user health profile.
* Disease Prediction: symptoms -> predicted disease + care recommendations
  (medicines, precautions, diet, workout, specialist) + related diseases
  (content-based filtering) + real medicines ranked by a hybrid
  sentiment/rating score (NLP on 200K+ patient reviews).
* Health Risk Screening: symptoms + vitals -> likelihood of positive diagnosis.
* Medicine Sentiment Explorer: per-condition drug rankings + live NLP demo.
* Analytics Dashboard: usage trends, disease popularity, model performance,
  dataset insights; admins additionally see user activity.

Run:  streamlit run app/app.py
Default admin: admin / admin123
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Fix: recent pyarrow builds segfault on macOS when their bundled mimalloc
# allocator initializes from Streamlit's script-runner threads. Arrow reads
# this env var lazily at first memory-pool use, which happens after this line.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Streamlit Cloud provides secrets via st.secrets — bridge DATABASE_URL to the
# environment before the first db call (the engine is created lazily), so a
# hosted PostgreSQL can be configured without code changes.
try:
    if "DATABASE_URL" in st.secrets:
        os.environ.setdefault("DATABASE_URL", st.secrets["DATABASE_URL"])
except Exception:
    pass  # no secrets file — default SQLite

# Make src/ importable regardless of where Streamlit is launched from.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auth import (  # noqa: E402
    get_events,
    get_profile,
    list_users,
    log_event,
    register_user,
    update_profile,
    verify_user,
)
from bandit import rank_medicines, record_feedback  # noqa: E402
from clinical import clinical_metrics, clinical_specs, predict_clinical_risk  # noqa: E402
from knowledge_graph import ego_graph_data, graph_related_diseases  # noqa: E402
from recommend import (  # noqa: E402
    condition_sentiment,
    get_drug_sentiment,
    humanize,
    list_sentiment_conditions,
    list_symptoms,
    predict_disease,
    predict_risk,
    related_diseases,
    sentiment_conditions_for,
)

MODELS = ROOT / "models"
ACCENT = "#2563eb"

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
      .main-title { font-size: 2.2rem; font-weight: 800; margin-bottom: 0;
                    background: linear-gradient(90deg, #1d4ed8, #0ea5e9);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
      .subtitle { color: #64748b; margin-top: 0.15rem; font-size: 1rem; }
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
      .pill-ghost {
          display:inline-block; border:1px solid #2563eb; color:#2563eb;
          padding:2px 10px; border-radius:999px; font-size:0.8rem; margin:2px;
      }
      .disclaimer {
          background: rgba(220, 38, 38, 0.08); border: 1px solid rgba(220,38,38,0.3);
          border-radius: 8px; padding: 0.7rem 1rem; font-size: 0.85rem; color:#b91c1c;
      }
      .role-badge {
          display:inline-block; background:#0ea5e9; color:#fff; font-size:0.72rem;
          padding:1px 8px; border-radius:999px; vertical-align:middle; margin-left:6px;
      }
      .footer { color:#94a3b8; font-size:0.8rem; text-align:center;
                margin-top:2.5rem; border-top:1px solid #e2e8f0; padding-top:0.8rem; }
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


@st.cache_data
def _dataset_insights() -> dict:
    """Lightweight stats about the training data for the dashboard."""
    df = pd.read_csv(ROOT / "data" / "raw" / "disease_symptoms.csv")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].dropna(axis=1, how="all")
    sym = df.drop(columns=["prognosis"]).sum().sort_values(ascending=False)
    return {
        "n_records": len(df),
        "n_diseases": df["prognosis"].nunique(),
        "n_symptoms": df.shape[1] - 1,
        "top_symptoms": sym.head(15),
    }


def _rec_card(title: str, items, is_list: bool = True) -> None:
    if not items:
        return
    if is_list and isinstance(items, str):
        items = [i.strip() for i in items.split(";") if i.strip()]
    if is_list:
        body = "".join(f'<span class="pill">{i}</span>' for i in items)
    else:
        body = f"<p style='margin:0'>{items}</p>"
    st.markdown(f'<div class="rec-card"><h4>{title}</h4>{body}</div>', unsafe_allow_html=True)


_KIND_COLORS = {
    "disease": "#2563eb",
    "symptom": "#f59e0b",
    "medication": "#10b981",
    "specialist": "#8b5cf6",
}


def _ego_graph_fig(disease: str):
    """Interactive plotly network of the disease's knowledge-graph neighborhood."""
    data = ego_graph_data(disease)
    if not data:
        return None
    import networkx as nx

    g = nx.Graph()
    for e in data["edges"]:
        g.add_edge(e["source"], e["target"], weight=e["weight"])
    pos = nx.spring_layout(g, seed=42, k=1.4)

    edge_x, edge_y = [], []
    for e in data["edges"]:
        x0, y0 = pos[e["source"]]
        x1, y1 = pos[e["target"]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=edge_x, y=edge_y, mode="lines",
                   line=dict(width=1, color="#cbd5e1"), hoverinfo="none")
    )
    kinds = {n["id"]: n["kind"] for n in data["nodes"]}
    for kind, color in _KIND_COLORS.items():
        xs, ys, names = [], [], []
        for node_id, k in kinds.items():
            if k == kind and node_id in pos:
                xs.append(pos[node_id][0]); ys.append(pos[node_id][1])
                names.append(node_id)
        if xs:
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="markers+text", name=kind,
                    text=[n if len(n) < 22 else n[:20] + "…" for n in names],
                    hovertext=names, hoverinfo="text",
                    textposition="top center", textfont=dict(size=9),
                    marker=dict(
                        size=26 if kind == "disease" else 14,
                        color=color, line=dict(width=1, color="#fff"),
                    ),
                )
            )
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=30, b=10),
        showlegend=True, legend=dict(orientation="h", y=-0.05),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        title=f"Knowledge graph around {disease}",
    )
    return fig


def _bar(x, y, title, color=ACCENT, height=300, xaxis="", horizontal=True):
    fig = go.Figure(
        go.Bar(
            x=x, y=y, orientation="h" if horizontal else "v", marker_color=color,
        )
    )
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40, b=10), xaxis_title=xaxis)
    return fig


# =========================================================================== #
# Authentication gate
# =========================================================================== #
if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.markdown('<p class="main-title">🩺 Personalized Healthcare & Medicine Recommendation System</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">ML-powered disease prediction, medicine recommendations, risk screening & review-sentiment analytics.</p>', unsafe_allow_html=True)
    st.write("")

    left, right = st.columns([1.1, 1])
    with left:
        login_tab, signup_tab = st.tabs(["🔐 Log in", "✨ Sign up"])
        with login_tab:
            with st.form("login", border=True):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Log in", type="primary", width="stretch"):
                    user = verify_user(u, p)
                    if user:
                        st.session_state.user = user
                        log_event(user["username"], "login", {})
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
            st.caption("Demo admin account: `admin` / `admin123`")
        with signup_tab:
            with st.form("signup", border=True):
                name = st.text_input("Full name")
                u2 = st.text_input("Choose a username")
                p2 = st.text_input("Choose a password", type="password")
                if st.form_submit_button("Create account", type="primary", width="stretch"):
                    ok, msg = register_user(u2, name, p2)
                    (st.success if ok else st.error)(msg)
    with right:
        m = _metrics()
        st.markdown("#### What's inside")
        cols = st.columns(3)
        if "disease_metrics.json" in m:
            cols[0].metric("Disease model", f"{m['disease_metrics.json']['test_accuracy']*100:.0f}%", "41 diseases")
        if "risk_metrics.json" in m:
            cols[1].metric("Risk screening", f"{m['risk_metrics.json']['test_accuracy']*100:.0f}%", "vs 52% baseline")
        if "sentiment_metrics.json" in m:
            cols[2].metric("NLP sentiment", f"{m['sentiment_metrics.json']['test_accuracy']*100:.0f}%", "215K reviews")
        st.markdown(
            """
            - 🔬 **Disease prediction** from 132 symptoms
            - 💊 **Medicine, diet & precaution** recommendations
            - 🔗 **Related diseases** (content-based filtering)
            - 💬 **Real-medicine rankings** from patient review sentiment (NLP)
            - 📊 **Personal risk screening** from your vitals
            - 📈 **Analytics dashboard** with usage trends
            """
        )
        st.markdown(
            '<div class="disclaimer">⚠️ Educational demo — not medical advice. '
            "Always consult a qualified doctor.</div>",
            unsafe_allow_html=True,
        )
    st.stop()

user = st.session_state.user

# =========================================================================== #
# Sidebar (logged in)
# =========================================================================== #
with st.sidebar:
    st.markdown(
        f"### 👋 {user['name']}<span class='role-badge'>{user['role']}</span>",
        unsafe_allow_html=True,
    )
    if st.button("Log out", width="stretch"):
        st.session_state.user = None
        st.rerun()

    st.divider()
    with st.expander("🧾 My health profile", expanded=False):
        prof = get_profile(user["username"])
        page = st.number_input("Age", 1, 120, int(prof.get("age", 30)))
        pgender = st.radio(
            "Gender", ["Female", "Male"],
            index=1 if prof.get("gender") == "Male" else 0, horizontal=True,
        )
        pbp = st.select_slider("Blood pressure", ["Low", "Normal", "High"], value=prof.get("bp", "Normal"))
        pchol = st.select_slider("Cholesterol", ["Low", "Normal", "High"], value=prof.get("chol", "Normal"))
        if st.button("Save profile", width="stretch"):
            update_profile(user["username"], {"age": int(page), "gender": pgender, "bp": pbp, "chol": pchol})
            st.success("Profile saved — the Risk tab now uses it.")

    st.divider()
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
        "Not a substitute for professional medical advice.</div>",
        unsafe_allow_html=True,
    )

st.markdown('<p class="main-title">Personalized Healthcare & Medicine Recommendation System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Machine-learning powered disease prediction, care recommendations, risk screening and review analytics.</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔬  Disease Prediction",
        "📊  Health Risk Screening",
        "💬  Medicine Sentiment",
        "📈  Analytics Dashboard",
    ]
)

# =========================================================================== #
# Tab 1 — disease prediction & recommendations
# =========================================================================== #
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

    if st.button("🔍 Predict Disease", type="primary", width="stretch"):
        if len(chosen) == 0:
            st.warning("Please select at least one symptom.")
            st.session_state.pop("prediction", None)
        else:
            # Persist in session state so the results survive the reruns
            # triggered by the feedback (👍/👎) buttons below.
            result = predict_disease(chosen, top_k=3)
            st.session_state.prediction = result
            log_event(
                user["username"], "disease_prediction",
                {"disease": result["disease"], "confidence": result["confidence"],
                 "n_symptoms": len(chosen)},
            )

    result = st.session_state.get("prediction")
    if result:
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
                related = related_diseases(disease, top_n=3)
                if related:
                    pills = "".join(
                        f'<span class="pill-ghost">{r["disease"]} · {r["similarity"]*100:.0f}%</span>'
                        for r in related
                    )
                    st.markdown(
                        f'<div class="rec-card"><h4>🔗 Related diseases (similar symptom profiles)</h4>{pills}</div>',
                        unsafe_allow_html=True,
                    )
            with c2:
                top = result["top_k"]
                fig = go.Figure(
                    go.Bar(
                        x=[t["probability"] * 100 for t in top][::-1],
                        y=[t["disease"] for t in top][::-1],
                        orientation="h",
                        marker_color=ACCENT,
                        text=[f"{t['probability']*100:.1f}%" for t in top][::-1],
                        textposition="auto",
                    )
                )
                fig.update_layout(
                    title="Top predictions", height=250,
                    margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Probability (%)",
                )
                st.plotly_chart(fig, width="stretch")

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

            # Adaptive medicine ranking: hybrid score prior + RL from feedback.
            sent = get_drug_sentiment(disease, top_n=8)
            if sent is not None:
                ranked = rank_medicines(disease, sent)
                st.divider()
                st.subheader("💬 Top-Rated Real Medicines (adaptive ranking)")
                st.caption(
                    "Starts from a hybrid score (60% NLP review sentiment + 40% star "
                    "rating, 200K+ drugs.com reviews) and **adapts with your 👍/👎 "
                    "feedback** via a Thompson-sampling bandit — every vote updates "
                    f"the Beta posterior. Conditions: {', '.join(sentiment_conditions_for(disease))}."
                )
                sc1, sc2 = st.columns([1.25, 1])
                with sc1:
                    hc = st.columns([3.2, 1.4, 1.1, 0.6, 0.6])
                    for h, txt in zip(hc, ["**Medicine**", "**Score**", "**Votes**", "", ""]):
                        h.markdown(txt)
                    for _, row in ranked.iterrows():
                        drug = row["drugName"]
                        rc = st.columns([3.2, 1.4, 1.1, 0.6, 0.6])
                        rc[0].markdown(drug)
                        rc[1].markdown(f"{row['posterior_mean']*100:.0f}%")
                        rc[2].markdown(f"👍{int(row['ups'])} 👎{int(row['downs'])}")
                        if rc[3].button("👍", key=f"up_{disease}_{drug}"):
                            record_feedback(user["username"], disease, drug, 1)
                            log_event(user["username"], "feedback", {"disease": disease, "drug": drug, "vote": 1})
                            st.rerun()
                        if rc[4].button("👎", key=f"down_{disease}_{drug}"):
                            record_feedback(user["username"], disease, drug, -1)
                            log_event(user["username"], "feedback", {"disease": disease, "drug": drug, "vote": -1})
                            st.rerun()
                with sc2:
                    sfig = go.Figure(
                        go.Bar(
                            x=(ranked["posterior_mean"] * 100)[::-1],
                            y=ranked["drugName"][::-1],
                            orientation="h",
                            marker_color="#10b981",
                            text=[f"{v*100:.0f}" for v in ranked["posterior_mean"]][::-1],
                            textposition="auto",
                        )
                    )
                    sfig.update_layout(
                        title="Adaptive helpfulness score (0–100)", height=340,
                        margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Posterior mean",
                    )
                    st.plotly_chart(sfig, width="stretch")

            # Graph-based recommendations: knowledge-graph neighborhood + PPR.
            st.divider()
            st.subheader("🕸 Knowledge Graph")
            g1, g2 = st.columns([1.3, 1])
            with g1:
                gfig = _ego_graph_fig(disease)
                if gfig is not None:
                    st.plotly_chart(gfig, width="stretch")
            with g2:
                st.markdown("##### Graph-walk related diseases")
                st.caption(
                    "Personalized PageRank over a medical knowledge graph "
                    "(diseases ↔ symptoms ↔ medications ↔ specialists). "
                    "Connections can flow through multi-hop paths — e.g., a "
                    "shared specialist or medication — not just direct "
                    "symptom overlap."
                )
                for r in graph_related_diseases(disease, top_n=5):
                    st.markdown(
                        f'<span class="pill-ghost">{r["disease"]}</span> '
                        f'<small style="color:#94a3b8">score {r["score"]}</small>',
                        unsafe_allow_html=True,
                    )

            st.markdown(
                '<div class="disclaimer">⚠️ These recommendations are general and '
                "educational. Medication names are drug classes or examples, not "
                "prescriptions. Please consult a licensed physician.</div>",
                unsafe_allow_html=True,
            )

# =========================================================================== #
# Tab 2 — risk screening (prefilled from profile)
# =========================================================================== #
def _clinical_form(which: str) -> dict:
    """Render a clinical input form from the saved spec; return feature values."""
    spec = clinical_specs()[which]
    values = {}
    cols = st.columns(3)
    for i, (feat, f) in enumerate(spec["fields"].items()):
        with cols[i % 3]:
            if f["kind"] == "select":
                choice = st.selectbox(f["label"], list(f["options"].keys()), key=f"{which}_{feat}")
                values[feat] = f["options"][choice]
            elif f["kind"] == "float":
                values[feat] = st.number_input(
                    f["label"], float(f["min"]), float(f["max"]), float(f["default"]),
                    step=0.1, key=f"{which}_{feat}",
                )
            else:
                values[feat] = st.number_input(
                    f["label"], int(f["min"]), int(f["max"]), int(f["default"]),
                    key=f"{which}_{feat}",
                )
    return values


def _risk_gauge(prob_pct: float, title: str):
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number", value=prob_pct, number={"suffix": "%"},
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": ACCENT},
                "steps": [
                    {"range": [0, 35], "color": "#dcfce7"},
                    {"range": [35, 60], "color": "#fef9c3"},
                    {"range": [60, 100], "color": "#fee2e2"},
                ],
            },
        )
    )
    gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10))
    st.plotly_chart(gauge, width="stretch")


with tab2:
    screen = st.radio(
        "Choose a screening:",
        ["🩺 General screening", "❤️ Heart disease risk", "🩸 Diabetes risk"],
        horizontal=True,
    )

    if screen != "🩺 General screening":
        which = "heart" if "Heart" in screen else "diabetes"
        spec = clinical_specs()[which]
        cm = clinical_metrics()[which]
        st.subheader(spec["title"].split(" (")[0])
        st.caption(
            f"Trained on **real clinical data** — {spec['title'].split('(')[1].rstrip(')')} · "
            f"{cm['best_model']} · test AUC **{cm['test_auc']:.2f}** · "
            f"accuracy {cm['test_accuracy']*100:.0f}% (baseline {cm['majority_baseline']*100:.0f}%)."
        )
        vals = _clinical_form(which)
        if st.button(f"Assess {which} risk", type="primary", width="stretch"):
            res = predict_clinical_risk(which, vals)
            prob = res["risk_probability"] * 100
            log_event(user["username"], f"{which}_risk_check",
                      {"probability": res["risk_probability"], "label": res["risk_label"]})
            _risk_gauge(prob, f"{'Heart disease' if which == 'heart' else 'Diabetes'} risk")
            if res["risk_label"] == "High":
                st.error(f"⚠️ **High risk** ({prob:.1f}%). Please consult a "
                         f"{'cardiologist' if which == 'heart' else 'diabetologist'}.")
            elif res["risk_label"] == "Moderate":
                st.warning(f"🟡 **Moderate risk** ({prob:.1f}%). A check-up is a good idea.")
            else:
                st.success(f"✅ **Low risk** ({prob:.1f}%). Keep up the healthy habits.")
            st.markdown(
                '<div class="disclaimer">⚠️ A statistical screening from a research '
                "dataset — not a diagnosis. Always consult a doctor.</div>",
                unsafe_allow_html=True,
            )
    else:
        st.subheader("Enter your health profile")
        prof = get_profile(user["username"])
        if prof:
            st.caption("✅ Prefilled from your saved profile — adjust as needed.")
        else:
            st.caption("Tip: save your profile in the sidebar to prefill this form.")

        level_map = {"Low": 0, "Normal": 1, "High": 2}
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 1, 120, int(prof.get("age", 35)), key="risk_age")
            gender = st.radio(
                "Gender", ["Female", "Male"],
                index=1 if prof.get("gender") == "Male" else 0, horizontal=True, key="risk_gender",
            )
        with col2:
            fever = st.checkbox("Fever")
            cough = st.checkbox("Cough")
            fatigue = st.checkbox("Fatigue")
            breathing = st.checkbox("Difficulty breathing")
        with col3:
            bp = st.select_slider("Blood pressure", ["Low", "Normal", "High"], value=prof.get("bp", "Normal"), key="risk_bp")
            chol = st.select_slider("Cholesterol level", ["Low", "Normal", "High"], value=prof.get("chol", "Normal"), key="risk_chol")

        if st.button("📈 Assess Risk", type="primary", width="stretch"):
            profile = {
                "fever": int(fever), "cough": int(cough), "fatigue": int(fatigue),
                "difficulty_breathing": int(breathing), "age": age,
                "gender": 1 if gender == "Male" else 0,
                "blood_pressure": level_map[bp], "cholesterol_level": level_map[chol],
            }
            res = predict_risk(profile)
            prob = res["probability"] * 100
            log_event(user["username"], "risk_check", {"outcome": res["outcome"], "probability": res["probability"]})

            _risk_gauge(prob, "Likelihood of positive diagnosis")

            if res["outcome"] == "Positive":
                st.error(f"⚠️ Elevated risk — model predicts a **Positive** outcome ({prob:.1f}%). Consider consulting a doctor.")
            else:
                st.success(f"✅ Lower risk — model predicts a **Negative** outcome (positive likelihood {prob:.1f}%).")

            st.markdown(
                '<div class="disclaimer">⚠️ This screening is a statistical estimate on a '
                "small dataset, not a diagnosis. Consult a doctor for any health concern.</div>",
                unsafe_allow_html=True,
            )

# =========================================================================== #
# Tab 3 — medicine sentiment explorer (NLP)
# =========================================================================== #
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
                        "drugName": "Medicine", "n_reviews": "Reviews",
                        "avg_rating": "Avg rating (/10)", "sentiment_score": "Sentiment",
                        "total_useful": "Helpful votes",
                    }
                )[["Medicine", "Reviews", "Avg rating (/10)", "Sentiment", "Helpful votes"]]
                st.dataframe(show, width="stretch", hide_index=True, height=430)
            with e2:
                efig = go.Figure(
                    go.Scatter(
                        x=rows["n_reviews"], y=rows["sentiment_score"] * 100,
                        mode="markers+text", text=rows["drugName"],
                        textposition="top center", textfont=dict(size=9),
                        marker=dict(
                            size=rows["avg_rating"] * 2.2,
                            color=rows["sentiment_score"] * 100,
                            colorscale="RdYlGn", cmin=0, cmax=100,
                            showscale=True, colorbar=dict(title="Sent. %"),
                        ),
                    )
                )
                efig.update_layout(
                    title=f"Drugs for {cond}: sentiment vs. review volume",
                    xaxis_title="Number of reviews", yaxis_title="Positive sentiment (%)",
                    height=430, margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(efig, width="stretch")

    st.divider()
    st.subheader("🧪 Try the sentiment model live")
    demo_text = st.text_area(
        "Review text:",
        placeholder="e.g. This medication worked wonders for my headaches, no side effects at all!",
        height=100,
    )
    if st.button("Analyze sentiment", width="stretch"):
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

# =========================================================================== #
# Tab 4 — analytics dashboard
# =========================================================================== #
with tab4:
    st.subheader("📈 Analytics & Reporting")

    events = get_events()
    my_events = [e for e in events if e["user"] == user["username"]]
    predictions = [e for e in events if e["type"] == "disease_prediction"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total predictions (all users)", len(predictions))
    k2.metric("My activity events", len(my_events))
    insights = _dataset_insights()
    k3.metric("Diseases covered", insights["n_diseases"])
    k4.metric("Symptoms tracked", insights["n_symptoms"])

    st.divider()
    d1, d2 = st.columns(2)

    with d1:
        st.markdown("##### 🔥 Most predicted diseases (popularity ranking)")
        if predictions:
            pop = pd.Series([p["detail"]["disease"] for p in predictions]).value_counts().head(10)
            st.plotly_chart(
                _bar(pop.values[::-1], pop.index[::-1], "", xaxis="Predictions"),
                width="stretch",
            )
        else:
            st.info("No predictions yet — run one in the Disease Prediction tab.")

        st.markdown("##### 🧪 Model performance comparison")
        m = _metrics()
        names, accs = [], []
        if "disease_metrics.json" in m:
            names.append("Disease (41-class)"); accs.append(m["disease_metrics.json"]["test_accuracy"] * 100)
        if "risk_metrics.json" in m:
            names.append("Risk screening"); accs.append(m["risk_metrics.json"]["test_accuracy"] * 100)
        if "sentiment_metrics.json" in m:
            names.append("Review sentiment (NLP)"); accs.append(m["sentiment_metrics.json"]["test_accuracy"] * 100)
        st.plotly_chart(
            _bar(accs, names, "", color="#10b981", xaxis="Test accuracy (%)"),
            width="stretch",
        )

    with d2:
        st.markdown("##### 🩹 Most common symptoms in the training data")
        ts = insights["top_symptoms"]
        st.plotly_chart(
            _bar(ts.values[::-1], [humanize(s) for s in ts.index][::-1], "", color="#0ea5e9", xaxis="Occurrences", height=430),
            width="stretch",
        )

    st.divider()
    if user["role"] == "Admin":
        st.markdown("##### 👥 User activity (Admin)")
        a1, a2 = st.columns(2)
        with a1:
            udf = pd.DataFrame(list_users())
            st.dataframe(udf, width="stretch", hide_index=True)
        with a2:
            if events:
                edf = pd.DataFrame(
                    [
                        {
                            "time": e["ts"][:19].replace("T", " "),
                            "user": e["user"],
                            "event": e["type"],
                            "detail": e["detail"].get("disease") or e["detail"].get("outcome") or "",
                        }
                        for e in events[:50]
                    ]
                )
                st.dataframe(edf, width="stretch", hide_index=True)
            else:
                st.info("No activity yet.")
    else:
        st.markdown("##### 🕘 My prediction history")
        mine = [e for e in my_events if e["type"] in ("disease_prediction", "risk_check")]
        if mine:
            hdf = pd.DataFrame(
                [
                    {
                        "time": e["ts"][:19].replace("T", " "),
                        "event": e["type"],
                        "result": e["detail"].get("disease") or e["detail"].get("outcome") or "",
                        "confidence": e["detail"].get("confidence") or e["detail"].get("probability") or "",
                    }
                    for e in mine[:50]
                ]
            )
            st.dataframe(hdf, width="stretch", hide_index=True)
        else:
            st.info("No history yet — make a prediction!")

st.markdown(
    '<div class="footer">Built with scikit-learn, XGBoost & Streamlit · '
    "Internship project @ Zidio Development · Educational use only</div>",
    unsafe_allow_html=True,
)
