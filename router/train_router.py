"""
router/train_router.py
-----------------------
Phase 5 -- Router Model Training

Pipeline:
  1. Load data_builder/dataset_sweep.csv
  2. Encode string features (llm_task_type)
  3. Stratified 80/20 train/test split
  4. 5-fold Stratified Cross-Validation on the 80% training set
     (each fold: 4 folds train, 1 fold validate)
  5. Train final model on full 80% training set
  6. Evaluate on held-out 20% test set
  7. Save model + artifacts to router/artifacts/

Models compared: XGBoost  vs  LightGBM
Winner selected by mean CV accuracy.

Usage:
  python router/train_router.py
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

WORKSPACE    = Path(__file__).resolve().parent.parent
DATASET_PATH = WORKSPACE / "data_builder" / "dataset_sweep.csv"
ARTIFACTS    = WORKSPACE / "router" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature columns (ML input)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "source_task_type_encoded",   # int  -- encoded benchmark source
    "prompt_length",              # int  -- word count
    "has_code_block",             # 0/1
    "has_math_symbols",           # 0/1
    "question_type_encoded",      # int  -- factual/analytical/math/creative/instruct
    "num_sentences",              # int
    "avg_word_length",            # float
    "complexity_heuristic",       # float [0,1]
    "llm_reasoning_depth",        # int 1-5
    "llm_ambiguity_score",        # float [0,1]
    "llm_context_dependency",     # 0/1
    "llm_requires_factual_recall",# 0/1
    "llm_task_type_encoded",      # int  -- encoded from string (generation/classification/QA)
]

TARGET_COL = "label_encoded"   # 0=tier1 (gpt-oss-20b), 1=tier2 (glm-5p2)
LABEL_NAMES = ["tier1", "tier2"]

# ---------------------------------------------------------------------------
# Load + preprocess
# ---------------------------------------------------------------------------

def load_data(path: Path):
    print(f"[*] Loading {path}...")
    df = pd.read_csv(path)
    print(f"    {len(df)} rows loaded.")

    # Encode llm_task_type string -> int
    task_type_map = {"classification": 0, "QA": 1, "generation": 2}
    df["llm_task_type_encoded"] = df["llm_task_type"].map(task_type_map).fillna(2).astype(int)

    # Ensure label_encoded is int
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Check all feature cols exist
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[FEATURE_COLS].values.astype(float)
    y = df[TARGET_COL].values

    print(f"    Feature matrix shape: {X.shape}")
    print(f"    Label distribution  : " + " | ".join(
        f"{LABEL_NAMES[i]}={int((y==i).sum())}" for i in range(2)
    ))
    return X, y, df

# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def make_xgb(n_classes: int):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

def make_lgbm(n_classes: int):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )

# ---------------------------------------------------------------------------
# 5-Fold Cross Validation
# ---------------------------------------------------------------------------

def cross_validate_model(model_fn, X_train, y_train, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accs = []
    fold_f1s  = []

    print(f"\n    5-Fold CV:")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        model = model_fn()
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1  = f1_score(y_val, y_pred, average="macro", zero_division=0)
        fold_accs.append(acc)
        fold_f1s.append(f1)

        print(f"      Fold {fold}: Accuracy={acc:.4f}  Macro-F1={f1:.4f}"
              f"  (train={len(X_tr)}, val={len(X_val)})")

    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    mean_f1  = np.mean(fold_f1s)
    print(f"      " + "-"*53)
    print(f"      Mean CV Accuracy : {mean_acc:.4f}  (±{std_acc:.4f})")
    print(f"      Mean CV Macro-F1 : {mean_f1:.4f}")

    return mean_acc, mean_f1, fold_accs

# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train():
    X, y, df = load_data(DATASET_PATH)

    # -- 80 / 20 stratified split ------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"\n[*] Data split:")
    print(f"    Train : {len(X_train)} rows (80%)")
    print(f"    Test  : {len(X_test)} rows (20%)  [held-out, not seen during CV]")

    # -- Cross-validate both models on training set ------------------------
    n_classes = 2  # binary: tier1 vs tier2

    print("\n" + "="*60)
    print(" XGBOOST -- 5-Fold Stratified Cross-Validation")
    print("="*60)
    xgb_cv_acc, xgb_cv_f1, xgb_folds = cross_validate_model(
        lambda: make_xgb(n_classes), X_train, y_train
    )

    print("\n" + "="*60)
    print(" LIGHTGBM -- 5-Fold Stratified Cross-Validation")
    print("="*60)
    lgbm_cv_acc, lgbm_cv_f1, lgbm_folds = cross_validate_model(
        lambda: make_lgbm(n_classes), X_train, y_train
    )

    # -- Select winner ----------------------------------------------------
    print("\n" + "="*60)
    print(" MODEL SELECTION")
    print("="*60)
    print(f"  XGBoost  CV Accuracy : {xgb_cv_acc:.4f}")
    print(f"  LightGBM CV Accuracy : {lgbm_cv_acc:.4f}")

    if xgb_cv_acc >= lgbm_cv_acc:
        winner_name = "XGBoost"
        final_model = make_xgb(n_classes)
        cv_acc = xgb_cv_acc
        cv_f1  = xgb_cv_f1
        cv_folds = xgb_folds
    else:
        winner_name = "LightGBM"
        final_model = make_lgbm(n_classes)
        cv_acc = lgbm_cv_acc
        cv_f1  = lgbm_cv_f1
        cv_folds = lgbm_folds

    print(f"\n  Winner: {winner_name}")

    # -- Train final model on FULL 80% training set ------------------------
    print(f"\n[*] Training final {winner_name} on full 80% training set ({len(X_train)} rows)...")
    final_model.fit(X_train, y_train)
    print("    Done.")

    # -- Evaluate on held-out 20% test set ---------------------------------
    y_pred_test = final_model.predict(X_test)
    test_acc    = accuracy_score(y_test, y_pred_test)
    test_f1     = f1_score(y_test, y_pred_test, average="macro", zero_division=0)

    print("\n" + "="*60)
    print(f" FINAL TEST SET RESULTS  ({len(X_test)} rows -- never seen during training)")
    print("="*60)
    print(f"  Test Accuracy  : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  Test Macro-F1  : {test_f1:.4f}")
    print()
    print("  Classification Report:")
    print(classification_report(
        y_test, y_pred_test,
        target_names=LABEL_NAMES,
        zero_division=0
    ))
    print("  Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred_test)
    print(f"              tier1  tier2")
    for i, row in enumerate(cm):
        print(f"  {LABEL_NAMES[i]:<8}  {str(row[0]):>5}  {str(row[1]):>5}")

    # -- Feature importances -----------------------------------------------
    importances = dict(zip(FEATURE_COLS, final_model.feature_importances_))
    sorted_imp  = sorted(importances.items(), key=lambda x: -x[1])
    print("\n  Feature Importances (top 13):")
    for feat, imp in sorted_imp:
        bar = "#" * int(imp * 60)
        print(f"    {feat:<35}  {imp:.4f}  {bar}")

    # -- Save artifacts ----------------------------------------------------
    model_path = ARTIFACTS / "router_model.joblib"
    joblib.dump(final_model, model_path)
    print(f"\n[OK] Model saved -> {model_path}")

    metrics = {
        "model":           winner_name,
        "n_features":      len(FEATURE_COLS),
        "feature_cols":    FEATURE_COLS,
        "train_rows":      int(len(X_train)),
        "test_rows":       int(len(X_test)),
        "cv_folds":        5,
        "cv_fold_accs":    [round(a, 4) for a in cv_folds],
        "cv_mean_acc":     round(cv_acc, 4),
        "cv_mean_f1":      round(cv_f1, 4),
        "test_accuracy":   round(test_acc, 4),
        "test_macro_f1":   round(test_f1, 4),
        "label_map":       {"0": "tier1 (gpt-oss-20b)", "1": "tier2 (glm-5p2)"},
        "feature_importances": {k: round(float(v), 4) for k, v in sorted_imp},
    }
    metrics_path = ARTIFACTS / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Metrics saved -> {metrics_path}")

    # Save feature schema for inference
    schema_path = ARTIFACTS / "feature_schema.json"
    schema = {
        "feature_cols": FEATURE_COLS,
        "llm_task_type_map": {"classification": 0, "QA": 1, "generation": 2},
        "label_map": {0: "tier1 (gpt-oss-20b)", 1: "tier2 (glm-5p2)"},
        "n_classes": 2,
    }
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"[OK] Feature schema saved -> {schema_path}")

    print("\n" + "="*60)
    print(f"  SUMMARY")
    print("="*60)
    print(f"  Model            : {winner_name} (binary)")
    print(f"  Tiers            : 0=tier1 (gpt-oss-20b)  1=tier2 (glm-5p2)")
    print(f"  CV Mean Accuracy : {cv_acc*100:.2f}%  (5-fold on 80% train set)")
    print(f"  Test Accuracy    : {test_acc*100:.2f}%  (held-out 20%)")
    print(f"  Test Macro-F1    : {test_f1:.4f}")
    print("="*60)


if __name__ == "__main__":
    train()
