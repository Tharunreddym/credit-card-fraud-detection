"""
visualization.py — EDA plots, model comparison, ROC/PR curves, confusion matrix.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = ["#E8593C", "#3B8BD4", "#1D9E75", "#534AB7", "#BA7517"]


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {path}")


def plot_eda(csv_path: str, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # ---- 1. Class distribution ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Exploratory Data Analysis", fontsize=15, fontweight="bold")

    counts = df["Class"].value_counts()
    bars   = axes[0].bar(["Normal", "Fraud"], counts.values,
                         color=["#3B8BD4", "#E8593C"], edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                     f"{val:,}", ha="center", fontweight="bold")
    axes[0].set_yscale("log")
    axes[0].set_title("Class Distribution (Imbalanced)")
    axes[0].set_ylabel("Count (log scale)")

    # ---- 2. Amount distribution ----
    df[df["Class"] == 0]["Amount"].hist(bins=60, ax=axes[1], alpha=0.7, color="#3B8BD4", label="Normal")
    df[df["Class"] == 1]["Amount"].hist(bins=60, ax=axes[1], alpha=0.8, color="#E8593C", label="Fraud")
    axes[1].set_xlim(0, 2500)
    axes[1].set_title("Transaction Amount Distribution")
    axes[1].set_xlabel("Amount ($)")
    axes[1].legend()

    # ---- 3. Time distribution ----
    df[df["Class"] == 0]["Time"].hist(bins=48, ax=axes[2], alpha=0.7, color="#3B8BD4", label="Normal")
    df[df["Class"] == 1]["Time"].hist(bins=48, ax=axes[2], alpha=0.8, color="#E8593C", label="Fraud")
    axes[2].set_title("Transaction Time Distribution")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].legend()

    plt.tight_layout()
    _save(fig, f"{output_dir}/eda_overview.png")

    # ---- 4. Correlation heatmap ----
    corr_top = df.corr()["Class"].abs().sort_values(ascending=False).index[1:16]
    fig2, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df[list(corr_top) + ["Class"]].corr(),
                annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.5, square=True, ax=ax)
    ax.set_title("Correlation Heatmap — Top 15 Features vs Fraud", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig2, f"{output_dir}/correlation_heatmap.png")


def plot_comparison(results_df: pd.DataFrame, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    top10  = results_df.head(10)
    labels = [r[:38] + "…" if len(r) > 38 else r for r in top10["Model"]]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Top 10 Configurations — Performance Comparison", fontsize=14, fontweight="bold")

    for ax, metric, color in zip(axes, ["AUROC", "F1"], ["#3B8BD4", "#E8593C"]):
        bars = ax.barh(range(len(top10)), top10[metric], color=color, alpha=0.85, edgecolor="white")
        ax.set_yticks(range(len(top10)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(f"{metric} Score", fontsize=13)
        ax.set_xlim(top10[metric].min() * 0.97, 1.005)
        for bar, val in zip(bars, top10[metric]):
            ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    _save(fig, f"{output_dir}/model_comparison.png")


def plot_roc_pr(results: list, y_test, top_n: int = 5, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    sorted_res = sorted(results, key=lambda x: x["AUROC"], reverse=True)[:top_n]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"ROC & Precision-Recall Curves — Top {top_n} Configurations",
                 fontsize=13, fontweight="bold")

    for res, color in zip(sorted_res, PALETTE):
        label = res["Model"][:35]
        fpr, tpr, _ = roc_curve(y_test, res["_y_proba"])
        axes[0].plot(fpr, tpr, color=color, lw=2,
                     label=f"{label} (AUC={res['AUROC']:.3f})")

        prec, rec, _ = precision_recall_curve(y_test, res["_y_proba"])
        axes[1].plot(rec, prec, color=color, lw=2,
                     label=f"{label} (AP={res['PR_AUC']:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    axes[0].legend(fontsize=8, loc="lower right")

    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    axes[1].legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    _save(fig, f"{output_dir}/roc_pr_curves.png")


def plot_confusion(results: list, y_test, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    best = sorted(results, key=lambda x: x["AUROC"], reverse=True)[0]
    cm   = confusion_matrix(y_test, best["_y_pred"])

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Fraud"],
                yticklabels=["Normal", "Fraud"],
                linewidths=0.5, ax=ax)
    ax.set_title(f"Confusion Matrix\n{best['Model']}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    tn, fp, fn, tp = cm.ravel()
    print(f"   True Positives  (fraud caught) : {tp:,}")
    print(f"   False Negatives (fraud missed) : {fn:,}  ← minimize this")
    print(f"   False Positives (false alarms) : {fp:,}")
    print(f"   True Negatives                 : {tn:,}")

    plt.tight_layout()
    _save(fig, f"{output_dir}/confusion_matrix.png")
