# Credit Card Fraud Detection — Capstone Project

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![AUROC](https://img.shields.io/badge/Best%20AUROC-0.9832-orange)

> **Investigating Machine Learning Techniques for Highly Imbalanced Credit Card Fraud Detection Systems**

**Tharun R. Mopuru** | MS Computer Engineering | University of Cincinnati | March 2026

---

## Overview

Credit card fraud detection is one of the most challenging real-world ML problems due to **extreme class imbalance** — only 0.17% of transactions are fraudulent (492 out of 284,807). Standard classifiers fail completely on this data, achieving 99.8% accuracy while catching almost zero fraud.

This project systematically investigates and compares **5 imbalance-handling techniques** across **4 ML models** (20 combinations total) to find the best pipeline for detecting fraud.

---

## Results

### Best Model: LightGBM + SMOTEENN

| Metric | Score |
|--------|-------|
| **AUROC** | **0.9832** |
| **PR-AUC** | 0.7556 |
| **F1-Score** | 0.6084 |
| **Recall** | **88.78%** |

> Catches **87 out of 98** fraud cases in the test set, with only 11 missed.

### Top 10 Configurations

| Model + Technique | AUROC | F1 | Recall |
|---|---|---|---|
| LightGBM + SMOTEENN | 0.9832 | 0.6084 | 0.8878 |
| XGBoost + SMOTEENN | 0.9814 | 0.7281 | — |
| XGBoost + ADASYN | 0.9796 | 0.7678 | — |
| XGBoost + SMOTE | 0.9792 | 0.8018 | — |
| Stacking + SMOTE | 0.9752 | 0.8300 | — |
| Logistic Regression + ADASYN | 0.9725 | 0.0355 | — |

---

## Project Structure

```
credit-card-fraud-detection/
├── main.py                  ← Run this to execute full pipeline
├── requirements.txt         ← All dependencies
├── src/
│   ├── data_loader.py       ← Load & preprocess dataset
│   ├── imbalance.py         ← SMOTE, ADASYN, Undersampling, SMOTEENN
│   ├── models.py            ← LR, RF, XGBoost, LightGBM, Stacking
│   ├── evaluation.py        ← Metrics & results table
│   ├── visualization.py     ← EDA, comparison, ROC/PR, confusion matrix
│   └── explainability.py    ← SHAP feature importance
├── data/
│   └── creditcard.csv       ← Download from Kaggle (see below)
└── outputs/                 ← All plots saved here automatically
```

---

## Output Plots

| Plot | Description |
|------|-------------|
| `eda_overview.png` | Class distribution, amount & time analysis |
| `correlation_heatmap.png` | Top 15 features correlated with fraud |
| `model_comparison.png` | AUROC & F1 bar chart — top 10 configs |
| `roc_pr_curves.png` | ROC and Precision-Recall curves |
| `confusion_matrix.png` | Best model confusion matrix |
| `shap_importance.png` | SHAP global feature importance |
| `shap_beeswarm.png` | SHAP feature impact directions |
| `shap_waterfall.png` | Single fraud transaction explained |
| `results_summary.csv` | Full results table (all 20 combinations) |

---

## Key Findings

- **SMOTEENN** (hybrid oversampling + cleaning) combined with **LightGBM** achieves the best AUROC of **0.9832**
- **XGBoost + SMOTE** achieves the best F1 of **0.8018** — best balance of precision and recall
- **Gradient boosting models** (XGBoost, LightGBM) significantly outperform Logistic Regression and Random Forest
- **PR-AUC is more informative than AUROC** for severely imbalanced fraud datasets
- **SHAP analysis** reveals **V14** and **V4** are the strongest fraud indicators (PCA-derived behavioral features)
- Transaction **amount alone is NOT sufficient** for fraud detection — behavioral patterns matter more

---

## SHAP Explainability

SHAP (SHapley Additive exPlanations) applied to XGBoost + SMOTE reveals:

- **V14** — strongest fraud predictor (mean |SHAP| = 2.3)
- **V4** — second strongest (mean |SHAP| = 2.0)
- **V8, V12, V1** — significant contributors
- Low values of V14 strongly **increase** fraud probability
- High values of V4 strongly **decrease** fraud probability

---

## Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/Tharunreddym/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install dependencies
```bash
pip install numpy==2.1.0 pandas scikit-learn imbalanced-learn xgboost lightgbm shap matplotlib seaborn
```

### 3. Download the dataset
- Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Download `creditcard.csv`
- Place it in the `data/` folder

> **Note:** `creditcard.csv` is not included in this repo (143MB) due to GitHub's file size limit. Download directly from Kaggle.

### 4. Run the pipeline
```bash
python main.py
```

The full pipeline takes **5–10 minutes** to train all 20 model combinations. All plots save automatically to `outputs/`.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Kaggle — ULB ML Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Total transactions | 284,807 |
| Fraud cases | 492 (0.173%) |
| Imbalance ratio | 577:1 |
| Features | V1–V28 (PCA) + Amount + Time |
| Missing values | None |

---

## Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/-XGBoost-337AB7)
![LightGBM](https://img.shields.io/badge/-LightGBM-02569B)
![SHAP](https://img.shields.io/badge/-SHAP-FF6B6B)
![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white)

- **Imbalance:** imbalanced-learn (SMOTE, ADASYN, SMOTEENN)
- **Models:** scikit-learn, XGBoost, LightGBM
- **Explainability:** SHAP
- **Visualization:** Matplotlib, Seaborn
- **IDE:** IntelliJ IDEA

---

## Author

**Tharun R. Mopuru**
- Email: tharunreddymopuru@gmail.com
- LinkedIn: [linkedin.com/in/tharunrm](https://linkedin.com/in/tharunrm)
- GitHub: [github.com/Tharunreddym](https://github.com/Tharunreddym)

---

## License

This project is licensed under the MIT License.
