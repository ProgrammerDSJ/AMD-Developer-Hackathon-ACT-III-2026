import joblib, json
import numpy as np
import pandas as pd
from pathlib import Path

model = joblib.load("router/artifacts/router_model.joblib")
schema = json.loads(Path("router/artifacts/feature_schema.json").read_text())
cols = schema["feature_cols"]

print("Model type:", type(model).__name__)
print("Features:", cols)
print()

test_cases = [
    {"label": "Explain AI like a kid",   "task_type": "factual",  "complexity_score": 0.0, "depth": 2},
    {"label": "What is 2+2?",            "task_type": "factual",  "complexity_score": 0.0, "depth": 1},
    {"label": "Hard coding problem",     "task_type": "code",     "complexity_score": 1.0, "depth": 4, "has_code_block": 1},
    {"label": "Factual depth=1",         "task_type": "factual",  "complexity_score": 0.0, "depth": 1},
    {"label": "Factual depth=3",         "task_type": "factual",  "complexity_score": 0.0, "depth": 3},
    {"label": "Factual complexity=0.8",  "task_type": "factual",  "complexity_score": 0.8, "depth": 2},
]

print("Tier probabilities:")
for tc in test_cases:
    row = [tc.get(c, 0) for c in cols]
    X = pd.DataFrame([row], columns=cols)
    proba = model.predict_proba(X)[0]
    tier = "TIER1" if proba[0] >= 0.5 else "TIER2"
    print(tier + "  t1=" + str(round(proba[0],3)) + "  t2=" + str(round(proba[1],3)) + "  | " + tc["label"])

print()
if hasattr(model, "feature_importances_"):
    imp = sorted(zip(cols, model.feature_importances_), key=lambda x: -x[1])
    print("Top feature importances:")
    for feat, score in imp[:10]:
        print("  " + feat.ljust(30) + str(round(score, 4)))
