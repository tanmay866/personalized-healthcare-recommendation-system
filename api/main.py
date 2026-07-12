"""
REST API backend — FastAPI + JWT authentication.

Exposes the ML system as a proper web service, independent of the Streamlit UI:

    POST /auth/signup           create an account
    POST /auth/login            get a JWT access token
    GET  /symptoms              list the 132 symptoms the model understands
    POST /predict/disease       (JWT) symptoms -> disease + recommendations
    POST /predict/risk          (JWT) vitals -> outcome likelihood
    GET  /recommend/{disease}   (JWT) knowledge-base entry for a disease
    GET  /sentiment/{condition} (JWT) top drugs for a condition by sentiment
    GET  /admin/users           (JWT, Admin role) list all users
    GET  /health                liveness probe

Run:   uvicorn api.main:app --reload --port 8000
Docs:  http://localhost:8000/docs  (interactive Swagger UI)

JWT secret: set the API_JWT_SECRET env var in production; a development
fallback is used otherwise.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auth import list_users, log_event, register_user, verify_user  # noqa: E402
from bandit import rank_medicines, record_feedback  # noqa: E402
from clinical import clinical_metrics, predict_clinical_risk  # noqa: E402
from knowledge_graph import ego_graph_data, graph_related_diseases, graph_stats  # noqa: E402
from recommend import (  # noqa: E402
    condition_sentiment,
    get_drug_sentiment,
    get_recommendation,
    list_sentiment_conditions,
    list_symptoms,
    predict_disease,
    predict_risk,
    related_diseases,
)

JWT_SECRET = os.environ.get("API_JWT_SECRET", "dev-secret-change-in-production")
JWT_ALGORITHM = "HS256"
TOKEN_TTL_HOURS = 24

app = FastAPI(
    title="Personalized Healthcare Recommendation API",
    description="ML-powered disease prediction, medicine recommendations, "
    "risk screening and drug-review sentiment — secured with JWT.",
    version="1.0.0",
)
bearer = HTTPBearer()


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class SignupRequest(BaseModel):
    username: str = Field(..., examples=["tanmay"])
    name: str = Field(..., examples=["Tanmay Patel"])
    password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_hours: int = TOKEN_TTL_HOURS


class DiseaseRequest(BaseModel):
    symptoms: list[str] = Field(
        ..., examples=[["itching", "skin_rash", "nodal_skin_eruptions"]]
    )
    top_k: int = Field(3, ge=1, le=10)


class FeedbackRequest(BaseModel):
    disease: str = Field(..., examples=["Migraine"])
    drug: str = Field(..., examples=["Amerge"])
    vote: int = Field(..., description="+1 helpful, -1 not helpful", ge=-1, le=1)


class RiskRequest(BaseModel):
    fever: bool = False
    cough: bool = False
    fatigue: bool = False
    difficulty_breathing: bool = False
    age: int = Field(..., ge=1, le=120)
    gender: str = Field("Female", pattern="^(Male|Female)$")
    blood_pressure: str = Field("Normal", pattern="^(Low|Normal|High)$")
    cholesterol_level: str = Field("Normal", pattern="^(Low|Normal|High)$")


# --------------------------------------------------------------------------- #
# JWT helpers
# --------------------------------------------------------------------------- #
def _make_token(user: dict) -> str:
    payload = {
        "sub": user["username"],
        "name": user["name"],
        "role": user["role"],
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_TTL_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
) -> dict:
    """Decode and validate the Bearer token; return the user claims."""
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")
    return {"username": payload["sub"], "name": payload["name"], "role": payload["role"]}


def admin_only(user: dict = Depends(current_user)) -> dict:
    if user["role"] != "Admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin role required")
    return user


# --------------------------------------------------------------------------- #
# Auth endpoints
# --------------------------------------------------------------------------- #
@app.post("/auth/signup", status_code=201, tags=["auth"])
def signup(body: SignupRequest):
    ok, msg = register_user(body.username, body.name, body.password)
    if not ok:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, msg)
    return {"message": msg}


@app.post("/auth/login", response_model=TokenResponse, tags=["auth"])
def login(body: LoginRequest):
    user = verify_user(body.username, body.password)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
    log_event(user["username"], "api_login", {})
    return TokenResponse(access_token=_make_token(user))


# --------------------------------------------------------------------------- #
# ML endpoints (JWT-protected)
# --------------------------------------------------------------------------- #
@app.get("/symptoms", tags=["ml"])
def symptoms():
    return {"symptoms": list_symptoms()}


@app.post("/predict/disease", tags=["ml"])
def predict_disease_ep(body: DiseaseRequest, user: dict = Depends(current_user)):
    known = set(list_symptoms())
    bad = [s for s in body.symptoms if s not in known]
    if bad:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Unknown symptoms: {bad}")
    if not body.symptoms:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Provide at least one symptom")

    result = predict_disease(body.symptoms, top_k=body.top_k)
    result["related_diseases"] = related_diseases(result["disease"])
    sent = get_drug_sentiment(result["disease"], top_n=5)
    result["top_real_medicines"] = (
        sent.to_dict(orient="records") if sent is not None else []
    )
    log_event(user["username"], "disease_prediction", {
        "disease": result["disease"], "confidence": result["confidence"],
        "n_symptoms": len(body.symptoms), "via": "api",
    })
    return result


@app.post("/predict/risk", tags=["ml"])
def predict_risk_ep(body: RiskRequest, user: dict = Depends(current_user)):
    level = {"Low": 0, "Normal": 1, "High": 2}
    res = predict_risk({
        "fever": int(body.fever), "cough": int(body.cough),
        "fatigue": int(body.fatigue),
        "difficulty_breathing": int(body.difficulty_breathing),
        "age": body.age, "gender": 1 if body.gender == "Male" else 0,
        "blood_pressure": level[body.blood_pressure],
        "cholesterol_level": level[body.cholesterol_level],
    })
    log_event(user["username"], "risk_check", {
        "outcome": res["outcome"], "probability": res["probability"], "via": "api",
    })
    return res


@app.get("/recommend/{disease}", tags=["ml"])
def recommend_ep(disease: str, user: dict = Depends(current_user)):
    rec = get_recommendation(disease)
    if not rec:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown disease: {disease}")
    return {"disease": disease, "recommendation": rec,
            "related_diseases": related_diseases(disease)}


@app.get("/sentiment/{condition}", tags=["ml"])
def sentiment_ep(condition: str, user: dict = Depends(current_user)):
    rows = condition_sentiment(condition, top_n=15)
    if rows is None or rows.empty:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"No sentiment data for '{condition}'. See /sentiment for options.",
        )
    return {"condition": condition, "drugs": rows.to_dict(orient="records")}


@app.get("/sentiment", tags=["ml"])
def sentiment_conditions_ep(user: dict = Depends(current_user)):
    return {"conditions": list_sentiment_conditions()}


@app.post("/predict/clinical/{which}", tags=["ml"])
def clinical_ep(which: str, values: dict, user: dict = Depends(current_user)):
    """Disease-specific risk from models trained on real clinical data.

    ``which`` is ``heart`` (UCI Cleveland) or ``diabetes`` (Pima Indians).
    ``values`` maps feature name -> value; missing features are median-imputed.
    """
    if which not in ("heart", "diabetes"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model must be 'heart' or 'diabetes'")
    known = set(clinical_metrics()[which]["features"])
    bad = [k for k in values if k not in known]
    if bad:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"Unknown features: {bad}. Valid: {sorted(known)}",
        )
    res = predict_clinical_risk(which, values)
    log_event(user["username"], f"{which}_risk_check", {**res, "via": "api"})
    return {"model": which, **res, "model_metrics": clinical_metrics()[which]}


@app.post("/feedback", status_code=201, tags=["ml"])
def feedback_ep(body: FeedbackRequest, user: dict = Depends(current_user)):
    """Record 👍/👎 medicine feedback — trains the Thompson-sampling bandit."""
    if body.vote == 0:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "vote must be +1 or -1")
    record_feedback(user["username"], body.disease, body.drug, body.vote)
    log_event(user["username"], "feedback", {
        "disease": body.disease, "drug": body.drug, "vote": body.vote, "via": "api",
    })
    return {"message": "Feedback recorded — ranking will adapt."}


@app.get("/medicines/{disease}", tags=["ml"])
def medicines_ep(disease: str, user: dict = Depends(current_user)):
    """Adaptive (bandit-ranked) real-medicine recommendations for a disease."""
    sent = get_drug_sentiment(disease, top_n=8)
    if sent is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"No sentiment-ranked medicines for '{disease}'"
        )
    ranked = rank_medicines(disease, sent)
    return {"disease": disease, "medicines": ranked.to_dict(orient="records")}


@app.get("/graph/{disease}", tags=["ml"])
def graph_ep(disease: str, user: dict = Depends(current_user)):
    """Knowledge-graph neighborhood + graph-walk related diseases."""
    ego = ego_graph_data(disease)
    if ego is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown disease: {disease}")
    return {
        "disease": disease,
        "ego_graph": ego,
        "graph_related": graph_related_diseases(disease),
        "graph_stats": graph_stats(),
    }


# --------------------------------------------------------------------------- #
# Admin & ops
# --------------------------------------------------------------------------- #
@app.get("/admin/users", tags=["admin"])
def admin_users(user: dict = Depends(admin_only)):
    return {"users": list_users()}


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok"}
