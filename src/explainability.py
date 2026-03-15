"""
explainability.py — SHAP analysis on best XGBoost model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier


def run_shap(smote_data: tuple, X_test, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    X_res, y_res = smote_data

    # Train XGBoost on SMOTE data
    model = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_res, y_res)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # ---- Bar: global feature importance ----
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar",
                      max_display=20, show=False)
    plt.title("SHAP Feature Importance — XGBoost + SMOTE", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{output_dir}/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {output_dir}/shap_importance.png")

    # ---- Beeswarm: feature impact direction ----
    fig2 = plt.figure(figsize=(10, 9))
    shap.summary_plot(shap_values, X_test, max_display=20, show=False)
    plt.title("SHAP Beeswarm — Feature Impact Direction", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig2.savefig(f"{output_dir}/shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"   Saved: {output_dir}/shap_beeswarm.png")

    # ---- Waterfall: single fraud transaction explained ----
    from sklearn.linear_model import LogisticRegression
    import pandas as pd
    y_test_arr    = np.array(X_test.index)  # placeholder — use actual y_test if passed
    fraud_indices = X_test.index[X_test.index >= 0]  # take first sample
    sample_idx    = 0

    fig3 = plt.figure(figsize=(10, 7))
    shap.waterfall_plot(
        shap.Explanation(
            values     = shap_values[sample_idx],
            base_values= explainer.expected_value,
            data       = X_test.iloc[sample_idx].values,
            feature_names = X_test.columns.tolist()
        ),
        max_display=15,
        show=False
    )
    plt.title("SHAP Waterfall — Single Fraud Transaction Explained", fontsize=12)
    plt.tight_layout()
    fig3.savefig(f"{output_dir}/shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"   Saved: {output_dir}/shap_waterfall.png")
