"""
inference_wrapper/router_core.py
Loads the trained binary router and returns routing decisions.
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib

warnings.filterwarnings("ignore", message="X does not have valid feature names")

WORKSPACE  = Path(__file__).resolve().parent.parent
MODEL_PATH = WORKSPACE / "router" / "artifacts" / "router_model.joblib"
SCHEMA_PATH= WORKSPACE / "router" / "artifacts" / "feature_schema.json"

_router = None
_schema = None

def _load():
    global _router, _schema
    if _router is None:
        _router = joblib.load(MODEL_PATH)
        with open(SCHEMA_PATH) as f:
            _schema = json.load(f)


def predict(features: Dict[str, Any]) -> Tuple[str, float, float]:
    """
    Returns (tier, tier1_prob, tier2_prob).
    tier1_prob is also used as the 'local confidence' signal
    (high tier1_prob = easy prompt = local model likely sufficient).
    """
    _load()
    cols = _schema["feature_cols"]
    X = pd.DataFrame([[features.get(c, 0.0) for c in cols]], columns=cols)
    proba = _router.predict_proba(X)[0]
    tier1_p, tier2_p = float(proba[0]), float(proba[1])
    tier = "tier1" if tier1_p >= 0.5 else "tier2"
    return tier, tier1_p, tier2_p
