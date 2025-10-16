import os, json, joblib, numpy as np, pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

CAL_FILE = Path("calibration.json")
MISSING_TOKENS = {'.', 'NA', 'N/A', 'na', 'NaN', '', 'null', 'NULL'}

# Simple built-in baseline so the app runs even without a trained model
class BaselineModel:
    """
    p = sigmoid( 1.2*CADD + 3.0*REVEL + 0.8*GERP - 1.0*AF_pos )
    where AF_pos maps AF_log10 in [-6,0] -> [0,1]. Missing -> 0.
    """
    def __init__(self, feature_order: Optional[List[str]] = None):
        self.feature_order = feature_order or ["AF_log10","CADD_phred","REVEL_score","GERP++_RS"]
        self.n_features_in_ = len(self.feature_order)

    def _num(self, df: pd.DataFrame, name: str, alts: Tuple[str,...]=()) -> pd.Series:
        for c in (name,) + tuple(alts):
            if c in df.columns:
                s = pd.to_numeric(df[c].replace(list(MISSING_TOKENS), np.nan), errors='coerce')
                return s
        return pd.Series(np.nan, index=df.index)

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        for c in self.feature_order:
            if c not in X.columns:
                X[c] = np.nan

        af = self._num(X, 'AF_log10')
        if af.isna().all() and 'AF_any' in X.columns:
            af_any = self._num(X, 'AF_any')
            af = np.log10(af_any + 1e-6)

        af_pos = ((af + 6.0) / 6.0).clip(0,1).fillna(0.0).values
        cadd  = (self._num(X, 'CADD_phred', ('CADD_PHRED','cadd_phred')).fillna(0.0).values) / 30.0
        revel =  self._num(X, 'REVEL_score', ('REVEL','revel_score')).fillna(0.0).values
        gerp  = (self._num(X, 'GERP++_RS', ('GERP_RS','GERP_RS')).fillna(0.0).values) / 6.0

        z = 1.2*cadd + 3.0*revel + 0.8*gerp - 1.0*af_pos
        return 1/(1+np.exp(-z))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self._transform(X)
        return np.vstack([1-p, p]).T

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = self._transform(X)
        return (p >= 0.5).astype(int)

def env(n, d=None):
    v = os.environ.get(n, d)
    return v if v not in ('', None) else d

def load_model_and_features(model_path: str, features_path: Optional[str] = None):
    feats = None
    if features_path and os.path.exists(features_path):
        feats = [l.strip() for l in open(features_path, encoding='utf-8') if l.strip()]
    try:
        if model_path and os.path.exists(model_path):
            model = joblib.load(model_path)
            return model, feats
    except Exception as e:
        print(f"[WARN] Failed to load model at {model_path}: {e}")
    if not feats:
        feats = ["AF_log10","CADD_phred","REVEL_score","GERP++_RS"]
    print("[INFO] Using BaselineModel fallback for demo/testing.")
    return BaselineModel(feature_order=feats), feats

def extract_feature_names(fitted) -> List[str]:
    try:
        if hasattr(fitted, 'get_feature_names_out'):
            return list(fitted.get_feature_names_out())
    except Exception:
        pass
    if hasattr(fitted, 'n_features_in_'):
        return [f'f{i}' for i in range(fitted.n_features_in_)]
    return []

def ensure_dataframe(records: List[dict], expected_cols: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.DataFrame.from_records([r for r in records])
    if not df.empty:
        df = df.replace({t: np.nan for t in MISSING_TOKENS})
    drop_like = {'chrom','chromosome','chr','pos','position','start','end','ref','alt',
                 'id','variant_id','rsid','rs','sample','sample_id','gene','gene_symbol','symbol'}
    to_drop = [c for c in df.columns if c.lower() in drop_like]
    if to_drop:
        df = df.drop(columns=to_drop)
    if expected_cols:
        for c in expected_cols:
            if c not in df.columns:
                df[c] = np.nan
        df = df[expected_cols]
    return df

def safe_predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, 'predict_proba'):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] == 2:
            return p[:, 1]
        if p.ndim == 1:
            return p
    if hasattr(model, 'decision_function'):
        s = model.decision_function(X)
        return 1/(1+np.exp(-s))
    yhat = model.predict(X)
    return yhat.astype(float)

def thresholds_load(default_low: float, default_high: float) -> Tuple[float,float]:
    if CAL_FILE.exists():
        try:
            obj = json.loads(CAL_FILE.read_text(encoding='utf-8'))
            return float(obj.get('low', default_low)), float(obj.get('high', default_high))
        except Exception:
            pass
    return default_low, default_high

def thresholds_save(low: float, high: float):
    CAL_FILE.write_text(json.dumps({'low': float(low), 'high': float(high)}, indent=2), encoding='utf-8')

def calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins:int=10):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0,1,n_bins+1)
    idx = np.digitize(y_prob, bins) - 1
    idx = np.clip(idx, 0, n_bins-1)
    acc, conf, counts = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            acc.append(np.nan); conf.append((bins[b]+bins[b+1])/2); counts.append(0)
        else:
            acc.append(y_true[m].mean()); conf.append(y_prob[m].mean()); counts.append(int(m.sum()))
    return np.array(conf), np.array(acc), np.array(counts)

def calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins:int=10):
    conf, acc, counts = calibration_bins(y_true, y_prob, n_bins)
    mask = ~np.isnan(acc)
    ece = np.sum(np.abs(acc[mask]-conf[mask]) * (counts[mask]/counts[mask].sum())) if counts[mask].sum()>0 else np.nan
    bs  = brier_score_loss(y_true, y_prob)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = float('nan')
    try:
        auprc = average_precision_score(y_true, y_prob)
    except Exception:
        auprc = float('nan')
    return {'ECE': float(ece), 'Brier': float(bs), 'AUROC': float(auroc), 'AUPRC': float(auprc)}, (conf, acc, counts)

def parse_class_map(s: str) -> Dict:
    m = {}
    for part in s.split(','):
        if ':' in part:
            k, v = part.split(':', 1)
            k = k.strip(); v = v.strip()
            try:
                k = int(k)
            except Exception:
                pass
            m[k] = v
    return m
