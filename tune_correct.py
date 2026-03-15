"""
tune_correct.py — Proper cross-validation with SMOTE INSIDE each fold.
This prevents data leakage where synthetic samples cross fold boundaries.

The correct approach:
  WRONG: resample entire training set → then cross-validate
  RIGHT: put resampling INSIDE the CV pipeline so each fold resamples independently

Run: python tune_correct.py
Results saved to: outputs/correct_cv_results.csv
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, average_precision_score,
                              make_scorer)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.metrics import geometric_mean_score
from xgboost import XGBClassifier
import lightgbm as lgb

from src.data_loader import load_and_preprocess

import os
os.makedirs("outputs", exist_ok=True)

print("=" * 65)
print("  Correct Cross-Validation — SMOTE Inside Each Fold")
print("  Tharun R. Mopuru | University of Cincinnati")
print("=" * 65)
print()
print("  WHY THIS MATTERS:")
print("  The previous CV (CV=1.0) was wrong — SMOTE was applied")
print("  before CV, causing synthetic samples to leak across folds.")
print("  This script fixes that by putting SMOTE inside the pipeline")
print("  so each fold resamples independently. Results will be lower")
print("  but they are HONEST and methodologically correct.")
print()

# ── 1. Load data ──
print("[1/4] Loading data...")
X_train, X_test, y_train, y_test = load_and_preprocess("data/creditcard.csv")
print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Scoring functions ──
scoring = {
    "roc_auc"  : "roc_auc",
    "f1"       : make_scorer(f1_score),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall"   : make_scorer(recall_score),
    "avg_prec" : "average_precision",
}

results = []

# ════════════════════════════════════════
# 2. XGBoost + SMOTE — Best tuned params
# ════════════════════════════════════════
print("[2/4] Running correct 5-fold CV for XGBoost + SMOTE...")
print("      Best params from tuning: colsample_bytree=0.8, learning_rate=0.2,")
print("      max_depth=6, min_child_weight=1, n_estimators=300, subsample=0.8")
print()

xgb_pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("model", XGBClassifier(
        colsample_bytree=0.8,
        learning_rate=0.2,
        max_depth=6,
        min_child_weight=1,
        n_estimators=300,
        subsample=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    ))
])

xgb_cv = cross_validate(
    xgb_pipeline, X_train, y_train,
    cv=skf, scoring=scoring,
    n_jobs=-1, verbose=1, return_train_score=False
)

xgb_cv_results = {
    "Model"         : "XGBoost + SMOTE (Tuned) — Correct CV",
    "CV_AUROC_mean" : round(xgb_cv["test_roc_auc"].mean(), 4),
    "CV_AUROC_std"  : round(xgb_cv["test_roc_auc"].std(), 4),
    "CV_F1_mean"    : round(xgb_cv["test_f1"].mean(), 4),
    "CV_F1_std"     : round(xgb_cv["test_f1"].std(), 4),
    "CV_Precision_mean": round(xgb_cv["test_precision"].mean(), 4),
    "CV_Recall_mean": round(xgb_cv["test_recall"].mean(), 4),
    "CV_PR_AUC_mean": round(xgb_cv["test_avg_prec"].mean(), 4),
}

print(f"\n  XGBoost + SMOTE — Correct 5-Fold CV Results:")
print(f"    AUROC  : {xgb_cv_results['CV_AUROC_mean']} ± {xgb_cv_results['CV_AUROC_std']}")
print(f"    F1     : {xgb_cv_results['CV_F1_mean']} ± {xgb_cv_results['CV_F1_std']}")
print(f"    Precision: {xgb_cv_results['CV_Precision_mean']}")
print(f"    Recall : {xgb_cv_results['CV_Recall_mean']}")
print(f"    PR-AUC : {xgb_cv_results['CV_PR_AUC_mean']}")
print(f"    Per-fold AUROC: {[round(x,4) for x in xgb_cv['test_roc_auc']]}")

# Also evaluate on test set
xgb_pipeline.fit(X_train, y_train)
y_pred_xgb  = xgb_pipeline.predict(X_test)
y_proba_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]

xgb_cv_results["Test_AUROC"]    = round(roc_auc_score(y_test, y_proba_xgb), 4)
xgb_cv_results["Test_PR_AUC"]   = round(average_precision_score(y_test, y_proba_xgb), 4)
xgb_cv_results["Test_F1"]       = round(f1_score(y_test, y_pred_xgb), 4)
xgb_cv_results["Test_Precision"]= round(precision_score(y_test, y_pred_xgb), 4)
xgb_cv_results["Test_Recall"]   = round(recall_score(y_test, y_pred_xgb), 4)
xgb_cv_results["Test_G_Mean"]   = round(geometric_mean_score(y_test, y_pred_xgb), 4)

print(f"\n  XGBoost + SMOTE — Test Set Results:")
print(f"    AUROC    : {xgb_cv_results['Test_AUROC']}")
print(f"    PR-AUC   : {xgb_cv_results['Test_PR_AUC']}")
print(f"    F1       : {xgb_cv_results['Test_F1']}")
print(f"    Precision: {xgb_cv_results['Test_Precision']}")
print(f"    Recall   : {xgb_cv_results['Test_Recall']}")
print(f"    G-Mean   : {xgb_cv_results['Test_G_Mean']}")

results.append(xgb_cv_results)

# ════════════════════════════════════════
# 3. LightGBM + SMOTE — Best tuned params
# ════════════════════════════════════════
print("\n[3/4] Running correct 5-fold CV for LightGBM + SMOTE...")
print("      Best params from tuning: colsample_bytree=0.8, learning_rate=0.1,")
print("      max_depth=8, n_estimators=300, num_leaves=63, subsample=0.8")
print()

lgb_pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("model", lgb.LGBMClassifier(
        colsample_bytree=0.8,
        learning_rate=0.1,
        max_depth=8,
        n_estimators=300,
        num_leaves=63,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ))
])

lgb_cv = cross_validate(
    lgb_pipeline, X_train, y_train,
    cv=skf, scoring=scoring,
    n_jobs=-1, verbose=1, return_train_score=False
)

lgb_cv_results = {
    "Model"         : "LightGBM + SMOTE (Tuned) — Correct CV",
    "CV_AUROC_mean" : round(lgb_cv["test_roc_auc"].mean(), 4),
    "CV_AUROC_std"  : round(lgb_cv["test_roc_auc"].std(), 4),
    "CV_F1_mean"    : round(lgb_cv["test_f1"].mean(), 4),
    "CV_F1_std"     : round(lgb_cv["test_f1"].std(), 4),
    "CV_Precision_mean": round(lgb_cv["test_precision"].mean(), 4),
    "CV_Recall_mean": round(lgb_cv["test_recall"].mean(), 4),
    "CV_PR_AUC_mean": round(lgb_cv["test_avg_prec"].mean(), 4),
}

print(f"\n  LightGBM + SMOTE — Correct 5-Fold CV Results:")
print(f"    AUROC  : {lgb_cv_results['CV_AUROC_mean']} ± {lgb_cv_results['CV_AUROC_std']}")
print(f"    F1     : {lgb_cv_results['CV_F1_mean']} ± {lgb_cv_results['CV_F1_std']}")
print(f"    Precision: {lgb_cv_results['CV_Precision_mean']}")
print(f"    Recall : {lgb_cv_results['CV_Recall_mean']}")
print(f"    PR-AUC : {lgb_cv_results['CV_PR_AUC_mean']}")
print(f"    Per-fold AUROC: {[round(x,4) for x in lgb_cv['test_roc_auc']]}")

# Evaluate on test set
lgb_pipeline.fit(X_train, y_train)
y_pred_lgb  = lgb_pipeline.predict(X_test)
y_proba_lgb = lgb_pipeline.predict_proba(X_test)[:, 1]

lgb_cv_results["Test_AUROC"]    = round(roc_auc_score(y_test, y_proba_lgb), 4)
lgb_cv_results["Test_PR_AUC"]   = round(average_precision_score(y_test, y_proba_lgb), 4)
lgb_cv_results["Test_F1"]       = round(f1_score(y_test, y_pred_lgb), 4)
lgb_cv_results["Test_Precision"]= round(precision_score(y_test, y_pred_lgb), 4)
lgb_cv_results["Test_Recall"]   = round(recall_score(y_test, y_pred_lgb), 4)
lgb_cv_results["Test_G_Mean"]   = round(geometric_mean_score(y_test, y_pred_lgb), 4)

print(f"\n  LightGBM + SMOTE — Test Set Results:")
print(f"    AUROC    : {lgb_cv_results['Test_AUROC']}")
print(f"    PR-AUC   : {lgb_cv_results['Test_PR_AUC']}")
print(f"    F1       : {lgb_cv_results['Test_F1']}")
print(f"    Precision: {lgb_cv_results['Test_Precision']}")
print(f"    Recall   : {lgb_cv_results['Test_Recall']}")
print(f"    G-Mean   : {lgb_cv_results['Test_G_Mean']}")

results.append(lgb_cv_results)

# ── 4. Save ──
print("\n[4/4] Saving results...")
df = pd.DataFrame(results)
df.to_csv("outputs/correct_cv_results.csv", index=False)

print("\n" + "=" * 65)
print("  CORRECT CV COMPLETE!")
print("=" * 65)
print()
print("  COMPARISON — Wrong CV vs Correct CV:")
print(f"  XGBoost  Wrong CV AUROC : 1.0000 (data leakage!)")
print(f"  XGBoost  Correct CV AUROC: {xgb_cv_results['CV_AUROC_mean']} ± {xgb_cv_results['CV_AUROC_std']}")
print()
print(f"  LightGBM Wrong CV AUROC : 1.0000 (data leakage!)")
print(f"  LightGBM Correct CV AUROC: {lgb_cv_results['CV_AUROC_mean']} ± {lgb_cv_results['CV_AUROC_std']}")
print()
print("  These are your REAL, honest, defensible CV scores.")
print("  Saved: outputs/correct_cv_results.csv")
print()
print("  Upload correct_cv_results.csv and I will build")
print("  the final report and slides with 100% real numbers!")
