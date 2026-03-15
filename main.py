"""
Investigating ML Techniques for Highly Imbalanced Credit Card Fraud Detection
Tharun R. Mopuru | University of Cincinnati | MS Computer Engineering

Run: python main.py
"""

from src.data_loader     import load_and_preprocess
from src.imbalance       import apply_samplers
from src.models          import train_all_models
from src.evaluation      import evaluate_all, print_summary_table
from src.visualization   import plot_eda, plot_comparison, plot_roc_pr, plot_confusion
from src.explainability  import run_shap

import warnings
warnings.filterwarnings('ignore')


def main():
    print("=" * 65)
    print("  Credit Card Fraud Detection — Capstone Pipeline")
    print("  Tharun R. Mopuru | University of Cincinnati")
    print("=" * 65)

    # 1. Load & preprocess
    print("\n[1/6] Loading & preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess("data/creditcard.csv")

    # 2. EDA plots
    print("[2/6] Generating EDA plots...")
    plot_eda("data/creditcard.csv", output_dir="outputs")

    # 3. Apply imbalance techniques
    print("[3/6] Applying imbalance handling techniques...")
    resampled = apply_samplers(X_train, y_train)

    # 4. Train all model x sampler combinations
    print("[4/6] Training models (this may take a few minutes)...")
    results = train_all_models(resampled, X_train, y_train, X_test, y_test)

    # 5. Evaluate & visualize
    print("[5/6] Evaluating & generating plots...")
    results_df = evaluate_all(results, y_test)
    print_summary_table(results_df)
    plot_comparison(results_df, output_dir="outputs")
    plot_roc_pr(results, y_test, output_dir="outputs")
    plot_confusion(results, y_test, output_dir="outputs")

    # 6. SHAP explainability
    print("[6/6] Running SHAP explainability on best XGBoost model...")
    run_shap(resampled["SMOTE"], X_test, output_dir="outputs")

    print("\n✅ Pipeline complete! Check the 'outputs/' folder for all plots.")
    print(f"   Best model: {results_df.iloc[0]['Model']}")
    print(f"   AUROC={results_df.iloc[0]['AUROC']}  F1={results_df.iloc[0]['F1']}  Recall={results_df.iloc[0]['Recall']}")


if __name__ == "__main__":
    main()
