"""
evaluation.py — Aggregate results and print summary table.
"""

import pandas as pd


def evaluate_all(results: list, y_test) -> pd.DataFrame:
    """Convert results list to a sorted DataFrame."""
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    df   = pd.DataFrame(rows).sort_values("AUROC", ascending=False).reset_index(drop=True)
    return df


def print_summary_table(results_df: pd.DataFrame):
    print()
    print("=" * 80)
    print("  FINAL RESULTS — Sorted by AUROC")
    print("=" * 80)
    display_cols = ["Model", "AUROC", "PR_AUC", "F1", "Precision", "Recall", "G_Mean"]
    print(results_df[display_cols].to_string(index=False))
    print()
    best = results_df.iloc[0]
    print(f"  🏆 Best: {best['Model']}")
    print(f"     AUROC={best['AUROC']}  PR-AUC={best['PR_AUC']}  F1={best['F1']}  Recall={best['Recall']}")
    print("=" * 80)

    # Save CSV
    results_df[display_cols].to_csv("outputs/results_summary.csv", index=False)
    print("  Saved: outputs/results_summary.csv")
