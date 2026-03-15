"""
tune.py — Hyperparameter tuning for XGBoost and LightGBM
Run: python tune.py
Results saved to: outputs/tuning_results.csv
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, average_precision_score)
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import lightgbm as lgb

from src.data_loader import load_and_preprocess

import os
os.makedirs("outputs", exist_ok=True)

print("=" * 65)
print("  Hyperparameter Tuning — XGBoost & LightGBM")
print("  Tharun R. Mopuru | University of Cincinnati")
print("=" * 65)

# ── 1. Load data ──
print("\n[1/5] Loading data...")
X_train, X_test, y_train, y_test = load_and_preprocess("data/creditcard.csv")

# ── 2. Apply SMOTE ──
print("[2/5] Applying SMOTE...")
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
print(f"   SMOTE: {len(y_smote):,} samples | fraud={y_smote.sum():,}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ════════════════════════════════════════
# 3. XGBOOST TUNING
# ════════════════════════════════════════
print("\n[3/5] Tuning XGBoost — this takes ~20-30 mins...")
print("      Watch the [CV N/N] progress below:\n")

xgb_param_grid = {
    "n_estimators"    : [100, 200, 300],
    "max_depth"       : [3, 4, 6],
    "learning_rate"   : [0.05, 0.1, 0.2],
    "subsample"       : [0.8, 1.0],
    "min_child_weight": [1, 3],
    "colsample_bytree": [0.8, 1.0],
}

xgb_base = XGBClassifier(
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

xgb_grid = GridSearchCV(
    xgb_base,
    xgb_param_grid,
    cv=skf,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=2,
    refit=True
)
xgb_grid.fit(X_smote, y_smote)

print(f"\n  XGBoost Best Params : {xgb_grid.best_params_}")
print(f"  XGBoost Best CV AUROC: {xgb_grid.best_score_:.4f}")

# Evaluate tuned XGBoost on test set
xgb_best = xgb_grid.best_estimator_
y_pred_xgb  = xgb_best.predict(X_test)
y_proba_xgb = xgb_best.predict_proba(X_test)[:, 1]

xgb_results = {
    "Model"    : "XGBoost + SMOTE (Tuned)",
    "CV_AUROC" : round(xgb_grid.best_score_, 4),
    "AUROC"    : round(roc_auc_score(y_test, y_proba_xgb), 4),
    "PR_AUC"   : round(average_precision_score(y_test, y_proba_xgb), 4),
    "F1"       : round(f1_score(y_test, y_pred_xgb), 4),
    "Precision": round(precision_score(y_test, y_pred_xgb), 4),
    "Recall"   : round(recall_score(y_test, y_pred_xgb), 4),
    "G_Mean"   : round(geometric_mean_score(y_test, y_pred_xgb), 4),
}

print(f"\n  XGBoost Tuned Test Results:")
for k, v in xgb_results.items():
    if k != "Model":
        print(f"    {k:<12}: {v}")

# ════════════════════════════════════════
# 4. LIGHTGBM TUNING
# ════════════════════════════════════════
print("\n[4/5] Tuning LightGBM — this takes ~15-20 mins...")
print("      Watch the [CV N/N] progress below:\n")

lgb_param_grid = {
    "n_estimators"   : [100, 200, 300],
    "max_depth"      : [4, 6, 8],
    "learning_rate"  : [0.05, 0.1, 0.2],
    "num_leaves"     : [31, 63, 127],
    "subsample"      : [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

lgb_base = lgb.LGBMClassifier(
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_grid = GridSearchCV(
    lgb_base,
    lgb_param_grid,
    cv=skf,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=2,
    refit=True
)
lgb_grid.fit(X_smote, y_smote)

print(f"\n  LightGBM Best Params : {lgb_grid.best_params_}")
print(f"  LightGBM Best CV AUROC: {lgb_grid.best_score_:.4f}")

# Evaluate tuned LightGBM on test set
lgb_best = lgb_grid.best_estimator_
y_pred_lgb  = lgb_best.predict(X_test)
y_proba_lgb = lgb_best.predict_proba(X_test)[:, 1]

lgb_results = {
    "Model"    : "LightGBM + SMOTE (Tuned)",
    "CV_AUROC" : round(lgb_grid.best_score_, 4),
    "AUROC"    : round(roc_auc_score(y_test, y_proba_lgb), 4),
    "PR_AUC"   : round(average_precision_score(y_test, y_proba_lgb), 4),
    "F1"       : round(f1_score(y_test, y_pred_lgb), 4),
    "Precision": round(precision_score(y_test, y_pred_lgb), 4),
    "Recall"   : round(recall_score(y_test, y_pred_lgb), 4),
    "G_Mean"   : round(geometric_mean_score(y_test, y_pred_lgb), 4),
}

print(f"\n  LightGBM Tuned Test Results:")
for k, v in lgb_results.items():
    if k != "Model":
        print(f"    {k:<12}: {v}")

# ════════════════════════════════════════
# 5. CROSS-VALIDATION ON BEST MODEL
# ════════════════════════════════════════
print("\n[5/5] Running 5-fold cross-validation on best tuned model...")

# Compare both tuned models, pick the better one
best_model  = xgb_best if xgb_results["AUROC"] >= lgb_results["AUROC"] else lgb_best
best_name   = "XGBoost" if xgb_results["AUROC"] >= lgb_results["AUROC"] else "LightGBM"

cv_auroc = cross_val_score(best_model, X_smote, y_smote,
                            cv=skf, scoring="roc_auc", n_jobs=-1)
cv_f1    = cross_val_score(best_model, X_smote, y_smote,
                            cv=skf, scoring="f1", n_jobs=-1)

print(f"\n  5-Fold CV Results — Tuned {best_name} + SMOTE:")
print(f"    AUROC per fold : {[round(x,4) for x in cv_auroc]}")
print(f"    AUROC mean±std : {cv_auroc.mean():.4f} ± {cv_auroc.std():.4f}")
print(f"    F1    per fold : {[round(x,4) for x in cv_f1]}")
print(f"    F1    mean±std : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

# ── Save results ──
tuning_df = pd.DataFrame([xgb_results, lgb_results])
tuning_df["CV_AUROC_Mean"] = [xgb_grid.best_score_, lgb_grid.best_score_]

cv_row = {
    "Model"         : f"Tuned {best_name} + SMOTE (5-Fold CV)",
    "CV_AUROC"      : round(cv_auroc.mean(), 4),
    "AUROC"         : f"{cv_auroc.mean():.4f} ± {cv_auroc.std():.4f}",
    "PR_AUC"        : "-",
    "F1"            : f"{cv_f1.mean():.4f} ± {cv_f1.std():.4f}",
    "Precision"     : "-",
    "Recall"        : "-",
    "G_Mean"        : "-",
    "CV_AUROC_Mean" : round(cv_auroc.mean(), 4),
}
tuning_df = pd.concat([tuning_df, pd.DataFrame([cv_row])], ignore_index=True)
tuning_df.to_csv("outputs/tuning_results.csv", index=False)

# ── Save best params ──
best_params_df = pd.DataFrame([
    {"Model": "XGBoost", **xgb_grid.best_params_},
    {"Model": "LightGBM", **lgb_grid.best_params_},
])
best_params_df.to_csv("outputs/best_params.csv", index=False)

print("\n" + "=" * 65)
print("  TUNING COMPLETE!")
print("=" * 65)
print(f"\n  XGBoost  (Tuned) — AUROC={xgb_results['AUROC']}  F1={xgb_results['F1']}  Recall={xgb_results['Recall']}")
print(f"  LightGBM (Tuned) — AUROC={lgb_results['AUROC']}  F1={lgb_results['F1']}  Recall={lgb_results['Recall']}")
print(f"\n  CV AUROC ({best_name}): {cv_auroc.mean():.4f} ± {cv_auroc.std():.4f}")
print(f"  CV F1    ({best_name}): {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
print("\n  Saved: outputs/tuning_results.csv")
print("  Saved: outputs/best_params.csv")
print("\n  Share these results and I'll update the report!")
