# Credit Card Fraud Detection — Capstone Project
**Tharun R. Mopuru | University of Cincinnati | MS Computer Engineering**

> Investigating Machine Learning Techniques for Highly Imbalanced Credit Card Fraud Detection Systems

---

## Project Structure

```
capstone/
├── main.py                  ← Run this
├── requirements.txt
├── data/
│   └── creditcard.csv       ← Download from Kaggle (link below)
├── src/
│   ├── data_loader.py       ← Load & preprocess dataset
│   ├── imbalance.py         ← SMOTE, ADASYN, Undersampling, SMOTEENN
│   ├── models.py            ← LR, RF, XGBoost, LightGBM, Stacking
│   ├── evaluation.py        ← Metrics & results table
│   ├── visualization.py     ← EDA, comparison, ROC/PR, confusion matrix
│   └── explainability.py    ← SHAP feature importance
└── outputs/                 ← All plots saved here automatically
```

---

## Setup in IntelliJ IDEA

### Step 1 — Open Project
- `File → Open` → select the `capstone/` folder

### Step 2 — Create Python Interpreter
- `File → Project Structure → SDK → Add SDK → Python SDK`
- Choose **Virtualenv Environment** → create new in `capstone/venv/`
- Select Python 3.10 or 3.11

### Step 3 — Install Dependencies
Open the IntelliJ terminal (`Alt+F12`) and run:
```bash
pip install -r requirements.txt
```

### Step 4 — Download Dataset
- Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Download `creditcard.csv`
- Place it inside the `data/` folder

### Step 5 — Run
Right-click `main.py` → **Run 'main'**
Or in terminal:
```bash
python main.py
```

---

## What the Pipeline Does

| Stage | Description |
|-------|-------------|
| EDA | Class distribution, amount/time plots, correlation heatmap |
| Imbalance | SMOTE, ADASYN, Random Undersampling, SMOTEENN, Class Weights |
| Models | Logistic Regression, Random Forest, XGBoost, LightGBM, Stacking |
| Metrics | AUROC, PR-AUC, F1, Precision, Recall, G-Mean |
| Explainability | SHAP importance, beeswarm, waterfall plots |

---

## Output Files (in `outputs/`)

| File | Description |
|------|-------------|
| `eda_overview.png` | Class distribution, amount, time plots |
| `correlation_heatmap.png` | Top 15 feature correlations |
| `model_comparison.png` | AUROC & F1 bar chart — top 10 configs |
| `roc_pr_curves.png` | ROC and Precision-Recall curves |
| `confusion_matrix.png` | Best model confusion matrix |
| `shap_importance.png` | SHAP global feature importance |
| `shap_beeswarm.png` | SHAP feature impact directions |
| `shap_waterfall.png` | Single fraud transaction explained |
| `results_summary.csv` | Full results table |

---

## Dataset Info
- **Source**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions | 492 fraud (0.17%)
- Features V1–V28 are PCA-transformed for privacy
- Imbalance ratio: ~578:1
