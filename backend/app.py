import os, io, json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .schemas import PredictRequest, PredictResponse, Thresholds
from .utils import (
    env, load_model_and_features, extract_feature_names, ensure_dataframe,
    safe_predict_proba, thresholds_load, thresholds_save, calibration_metrics,
    parse_class_map,
)

API_KEY       = env("API_KEY")
MODEL_PATH    = env("MODEL_PATH", "")
FEATURES_PATH = env("FEATURES_PATH")
THRESHOLD     = float(env("THRESHOLD", "0.5"))
LOW_THR       = float(env("LOW_THRESH", "0.3"))
HIGH_THR      = float(env("HIGH_THRESH", "0.7"))
CLASS_MAP_STR = env("CLASS_MAP", "0:not candidate,1:candidate")
CLASS_MAP     = parse_class_map(CLASS_MAP_STR)

# Treat these tokens as missing on CSV upload
NA_TOKENS = ['.', 'NA', 'N/A', 'na', 'NaN', '', 'null', 'NULL']

app = FastAPI(title="CardioVar", version="2.4.2", docs_url="/docs", redoc_url="/redoc")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory="frontend"), name="static")
app.mount("/files", StaticFiles(directory="sample"),   name="files")

@app.on_event("startup")
def _load():
    global MODEL, EXPECTED_FEATURES, FEATURE_NAMES, LOW, HIGH
    MODEL, EXPECTED_FEATURES = load_model_and_features(MODEL_PATH, FEATURES_PATH)
    FEATURE_NAMES = extract_feature_names(MODEL)
    LOW, HIGH = thresholds_load(LOW_THR, HIGH_THR)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH or "(baseline)",
        "features_listed": bool(EXPECTED_FEATURES),
        "threshold": THRESHOLD,
        "triage": {"low": LOW, "high": HIGH},
        "class_map": CLASS_MAP,
    }

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(open("frontend/index.html","r",encoding="utf-8").read())

@app.get("/get_thresholds")
def get_thresholds():
    return {"low": float(LOW), "high": float(HIGH)}

@app.post("/set_thresholds")
def set_thresholds(t: Thresholds):
    global LOW, HIGH
    if not (0.0 <= t.low < t.high <= 1.0):
        raise HTTPException(status_code=400, detail="Require 0 <= low < high <= 1")
    LOW, HIGH = float(t.low), float(t.high)
    thresholds_save(LOW, HIGH)
    return {"low": LOW, "high": HIGH}

def triage_label(p: float):
    if p < LOW:  return "Benign"
    if p > HIGH: return "Pathogenic"
    return "VUS"

def to_label_name(raw):
    try:
        raw = int(raw)
    except Exception:
        pass
    return CLASS_MAP.get(raw, str(raw))

# -------- JSON prediction --------
@app.post("/predict", response_model=PredictResponse)
def predict_json(payload: PredictRequest):
    df    = ensure_dataframe(payload.records, EXPECTED_FEATURES)
    proba = safe_predict_proba(MODEL, df)
    preds  = proba.tolist()
    labels = [int(p >= THRESHOLD) for p in preds]
    names  = [to_label_name(l) for l in labels]
    return {
        "n": len(df),
        "threshold": THRESHOLD,
        "columns": list(df.columns),
        "predictions": preds,
        "labels": labels,
        "label_names": names,
        "shap_topk": None,  # kept simple for this build
    }

# -------- CSV prediction (Upload & Run CardioVar button) --------
@app.post("/predict_csv")
def predict_csv(file: UploadFile = File(...)):
    if not file.filename.endswith((".csv",".tsv")):
        raise HTTPException(status_code=400, detail="Please upload a CSV/TSV file")

    raw = file.file.read()
    sep = "," if file.filename.endswith(".csv") else "\t"
    # Robust missing values handling
    df_in = pd.read_csv(io.BytesIO(raw), sep=sep, low_memory=False,
                        keep_default_na=True, na_values=NA_TOKENS)

    # Build model matrix from input, aligned to expected features (adds missing as NaN)
    df = ensure_dataframe(df_in.to_dict(orient="records"), EXPECTED_FEATURES)

    # Predict
    proba = safe_predict_proba(MODEL, df)
    labels = (proba >= THRESHOLD).astype(int)
    triage = [triage_label(p) for p in proba]
    names  = [to_label_name(int(l)) for l in labels]

    out_df = df_in.copy()
    out_df["CardioVar_probability"]   = proba
    out_df["CardioVar_label"]         = labels
    out_df["CardioVar_label_name"]    = names
    out_df["CardioVar_triage"]        = triage

    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=cardiovar_scored_{file.filename}"}
    )

# -------- Calibration (optional) --------
@app.post("/calibrate_csv")
def calibrate_csv(file: UploadFile = File(...), bins: int = 10):
    if not file.filename.endswith((".csv",".tsv")):
        raise HTTPException(status_code=400, detail="Please upload a CSV/TSV file")

    raw = file.file.read()
    sep = "," if file.filename.endswith(".csv") else "\t"
    df = pd.read_csv(io.BytesIO(raw), sep=sep, low_memory=False,
                     keep_default_na=True, na_values=NA_TOKENS)

    prob_col = "CardioVar_probability" if "CardioVar_probability" in df.columns else None
    if prob_col is None:
        raise HTTPException(status_code=400, detail="Missing 'CardioVar_probability' column")
    if "target" not in df.columns:
        raise HTTPException(status_code=400, detail="Missing 'target' (0/1 labels) column")

    probs = df[prob_col].astype(float).values
    ytrue = df["target"].astype(int).values

    mets, (conf, acc, counts) = calibration_metrics(ytrue, probs, n_bins=bins)

    fig, ax = plt.subplots(figsize=(5,5), dpi=200)
    ax.plot([0,1], [0,1], linestyle="--", color="#9CA3AF", label="Perfect")
    ax.scatter(conf, acc,
               s=(counts/np.nanmax(counts+1e-9))*200+10,
               alpha=0.9, edgecolor="white", linewidth=0.5, label="Bins")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram"); ax.grid(alpha=.2); ax.legend(frameon=True)

    img = io.BytesIO()
    fig.tight_layout()
    fig.savefig(img, format="png")
    plt.close(fig)
    img.seek(0)

    headers = {"X-Metrics": json.dumps({"metrics": mets})}
    return StreamingResponse(img, media_type="image/png", headers=headers)
