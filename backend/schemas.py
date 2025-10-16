from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(...)
    return_shap: bool = Field(default=False)

class PredictResponse(BaseModel):
    n: int
    threshold: float
    columns: List[str]
    predictions: List[float]
    labels: List[int]
    label_names: List[str]
    shap_topk: Optional[List[Dict[str, float]]] = None

class Thresholds(BaseModel):
    low: float
    high: float
