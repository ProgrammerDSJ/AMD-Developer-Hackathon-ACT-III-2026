"""
router/train_router_v2.py
--------------------------
Improved binary router training with:
  1. Drop dead feature (llm_context_dependency = 0% importance)
  2. Feature interactions (cross-product of top features)
  3. Hyperparameter tuning via RandomizedSearchCV
  4. class_weight='balanced' for slight 62:38 imbalance
  5. Soft-vote ensemble: XGBoost + LightGBM

Usage: python router/train_router_v2.py
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

WORKSPACE    = Path(__file__).resolve().parent.parent
DATASET_PATH = WORKSPACE / "data_builder" / "dataset_sweep.csv"
ARTIFACTS    = WORKSPACE / "router" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Features — drop llm_context_dependency (0% importance in v1)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "source_task_type_encoded",
    "prompt_length",
    "has_code_block",
    "has_math_symbols",
    "question_type_encoded",
    "num_sentences",
    "avg_word_length",
    "complexity_heuristic",
    "llm_reasoning_depth",
    "llm_ambiguity_score",
    # "llm_context_dependency",   <-- DROPPED: 0% importance in v1
    "llm_requires_factual_recall",
    "llm_task_type_encoded",
    # Interaction features
    "feat_length_x_depth",      # prompt_length * llm_reasoning_depth
    "feat_complexity_x_code",   # complexity_heuristic * has_code_block
    "feat_math_x_depth",        # has_math_symbols * llm_reasoning_depth
]

TARGET_COL  = "label_encoded"   # 0=tier1 (gpt-oss-20b), 1=tier2 (glm-5p2)
LABEL_NAMES = ["tier1", "tier2"]

# ---------------------------------------------------------------------------
# Load + preprocess + engineer features
# ---------------------------------------------------------------------------

def load_data(path: Path):
    print(f"[*] Loading {path}...")
    df = pd.read_csv(path)
    print(f"    {len(df)} rows, {len(df.columns)} columns")

    # Encode llm_task_type
    task_type_map = {"classification": 0, "QA": 1, "generation": 2}
    df["llm_task_type_encoded"] = df["llm_task_type"].map(task_type_map).fillna(2).astype(int)

    # --- Feature interactions ---
    df["feat_length_x_depth"]   = df["prompt_length"] * df["llm_reasoning_depth"]
    df["feat_complexity_x_code"]= df["complexity_heuristic"] * df["has_code_block"]
    df["feat_math_x_depth"]     = df["has_math_symbols"] * df["llm_reasoning_depth"]

    df[TARGET_COL] = df[TARGET_COL].astype(int)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing: {missing}")

    X = df[FEATURE_COLS].values.astype(float)
    y = df[TARGET_COL].values

    print(f"    Feature matrix : {X.shape}")
    print(f"    Labels         : tier1={int((y==0).sum())} | tier2={int((y==1).sum())}")
    return X, y, df

# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------

def tune_xgb(X_train, y_train):
    from xgboost import XGBClassifier
    print("\n[*] RandomizedSearchCV for XGBoost (50 iterations)...")
    param_dist = {
        "n_estimators":    [200, 300, 400, 500],
        "max_depth":       [4, 5, 6, 7, 8],
        "learning_rate":   [0.01, 0.03, 0.05, 0.08, 0.1],
        "subsample":       [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree":[0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight":[1, 2, 3, 5],
        "gamma":           [0, 0.1, 0.2, 0.3],
        "scale_pos_weight":[1.0, 1.5, 2.0],  # handles 62:38 imbalance
    }
    base = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=50,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"    Best CV accuracy: {search.best_score_:.4f}")
    print(f"    Best params: {search.best_params_}")
    return search.best_estimator_, search.best_score_

def tune_lgbm(X_train, y_train):
    from lightgbm import LGBMClassifier
    print("\n[*] RandomizedSearchCV for LightGBM (50 iterations)...")
    param_dist = {
        "n_estimators":    [200, 300, 400, 500],
        "max_depth":       [4, 5, 6, 7, 8],
        "learning_rate":   [0.01, 0.03, 0.05, 0.08, 0.1],
        "subsample":       [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree":[0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_samples":[5, 10, 20, 30],
        "num_leaves":      [15, 31, 63, 127],
        "class_weight":    ["balanced", None],
    }
    base = LGBMClassifier(random_state=42, verbose=-1)
    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=50,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"    Best CV accuracy: {search.best_score_:.4f}")
    print(f"    Best params: {search.best_params_}")
    return search.best_estimator_, search.best_score_

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train():
    X, y, df = load_data(DATASET_PATH)

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"\n[*] Split: {len(X_train)} train / {len(X_test)} test")

    # --- Tune both models ---
    best_xgb,  xgb_cv  = tune_xgb(X_train, y_train)
    best_lgbm, lgbm_cv = tune_lgbm(X_train, y_train)

    # --- Soft-vote ensemble ---
    print("\n[*] Building soft-vote ensemble (XGBoost + LightGBM)...")
    ensemble = VotingClassifier(
        estimators=[("xgb", best_xgb), ("lgbm", best_lgbm)],
        voting="soft",
    )
    ensemble.fit(X_train, y_train)

    # --- Evaluate all three on test set ---
    print("\n" + "="*60)
    print(" FINAL TEST SET COMPARISON")
    print("="*60)

    results = {}
    for name, model in [("XGBoost (tuned)", best_xgb),
                        ("LightGBM (tuned)", best_lgbm),
                        ("Ensemble (XGB+LGB)", ensemble)]:
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average="macro", zero_division=0)
        results[name] = {"acc": acc, "f1": f1, "model": model, "pred": y_pred}
        print(f"\n  {name}:")
        print(f"    Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
        print(f"    Macro-F1 : {f1:.4f}")

    # Pick winner
    best_name = max(results, key=lambda k: results[k]["acc"])
    best      = results[best_name]
    print(f"\n  Winner: {best_name}")

    print("\n" + "="*60)
    print(f" WINNER DETAILS: {best_name}")
    print("="*60)
    print(classification_report(y_test, best["pred"], target_names=LABEL_NAMES, zero_division=0))
    cm = confusion_matrix(y_test, best["pred"])
    print("  Confusion Matrix:")
    print(f"              tier1  tier2")
    for i, row in enumerate(cm):
        print(f"  {LABEL_NAMES[i]:<8}  {str(row[0]):>5}  {str(row[1]):>5}")

    # Feature importances (from XGBoost)
    if hasattr(best_xgb, "feature_importances_"):
        importances = dict(zip(FEATURE_COLS, best_xgb.feature_importances_))
        sorted_imp  = sorted(importances.items(), key=lambda x: -x[1])
        print("\n  Feature Importances (tuned XGBoost):")
        for feat, imp in sorted_imp:
            bar = "#" * int(imp * 60)
            print(f"    {feat:<35}  {imp:.4f}  {bar}")

    # --- Save best model ---
    model_path = ARTIFACTS / "router_model.joblib"
    joblib.dump(best["model"], model_path)
    print(f"\n[OK] Model saved -> {model_path}")

    schema = {
        "feature_cols":      FEATURE_COLS,
        "llm_task_type_map": {"classification": 0, "QA": 1, "generation": 2},
        "label_map":         {0: "tier1 (gpt-oss-20b)", 1: "tier2 (glm-5p2)"},
        "n_classes":         2,
        "dropped_features":  ["llm_context_dependency"],
        "interaction_features": {
            "feat_length_x_depth":    "prompt_length * llm_reasoning_depth",
            "feat_complexity_x_code": "complexity_heuristic * has_code_block",
            "feat_math_x_depth":      "has_math_symbols * llm_reasoning_depth",
        },
    }
    schema_path = ARTIFACTS / "feature_schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"[OK] Feature schema saved -> {schema_path}")

    metrics = {
        "model":            best_name,
        "n_features":       len(FEATURE_COLS),
        "feature_cols":     FEATURE_COLS,
        "train_rows":       int(len(X_train)),
        "test_rows":        int(len(X_test)),
        "xgb_cv_acc":       round(xgb_cv, 4),
        "lgbm_cv_acc":      round(lgbm_cv, 4),
        "test_accuracy":    round(best["acc"], 4),
        "test_macro_f1":    round(best["f1"], 4),
        "all_model_results": {k: {"acc": round(v["acc"], 4), "f1": round(v["f1"], 4)}
                              for k, v in results.items()},
        "label_map":        {"0": "tier1 (gpt-oss-20b)", "1": "tier2 (glm-5p2)"},
    }
    with open(ARTIFACTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"  Baseline (v1)    : 80.13%")
    for name, v in results.items():
        delta = v["acc"] - 0.8013
        sign  = "+" if delta >= 0 else ""
        print(f"  {name:<25}: {v['acc']*100:.2f}%  ({sign}{delta*100:.2f}%)")
    print(f"\n  Best model       : {best_name}")
    print(f"  Test Accuracy    : {best['acc']*100:.2f}%")
    print(f"  Test Macro-F1    : {best['f1']:.4f}")
    print("="*60)


if __name__ == "__main__":
    train()
