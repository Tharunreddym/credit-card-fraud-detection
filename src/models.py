"""
models.py — Train all model x sampler combinations + Stacking ensemble.
"""

import numpy as np
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, StackingClassifier
from sklearn.metrics         import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from imblearn.metrics        import geometric_mean_score
from xgboost                 import XGBClassifier
import lightgbm              as lgb


def _get_base_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost"            : XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0),
        "LightGBM"           : lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
    }


def _fit_evaluate(model, X_tr, y_tr, X_te, y_te, name):
    model.fit(X_tr, y_tr)
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    return {
        "Model"    : name,
        "AUROC"    : round(roc_auc_score(y_te, y_proba), 4),
        "PR_AUC"   : round(average_precision_score(y_te, y_proba), 4),
        "F1"       : round(f1_score(y_te, y_pred), 4),
        "Precision": round(precision_score(y_te, y_pred), 4),
        "Recall"   : round(recall_score(y_te, y_pred), 4),
        "G_Mean"   : round(geometric_mean_score(y_te, y_pred), 4),
        "_model"   : model,
        "_y_proba" : y_proba,
        "_y_pred"  : y_pred,
    }


def train_all_models(resampled: dict, X_train, y_train, X_test, y_test) -> list:
    """
    Train every (model, sampler) combination and return list of result dicts.
    """
    results = []
    scale   = (y_train == 0).sum() / (y_train == 1).sum()

    for sampler_name, (X_res, y_res) in resampled.items():
        for model_name, model in _get_base_models().items():

            # Apply class weighting for the ClassWeight strategy
            if sampler_name == "ClassWeight":
                params = model.get_params()
                if model_name in ("Logistic Regression", "Random Forest"):
                    params["class_weight"] = "balanced"
                else:  # XGBoost / LightGBM
                    params["scale_pos_weight"] = scale
                model = model.__class__(**params)

            full_name = f"{model_name} + {sampler_name}"
            print(f"   Training: {full_name}")
            res = _fit_evaluate(model, X_res, y_res, X_test, y_test, full_name)
            print(f"   → AUROC={res['AUROC']}  F1={res['F1']}  Recall={res['Recall']}")
            results.append(res)

    # ---- Stacking Ensemble (best combo: SMOTE) ----
    print("   Training: Stacking Ensemble (LR + RF + XGB) + SMOTE")
    X_smote, y_smote = resampled["SMOTE"]
    stacking = StackingClassifier(
        estimators=[
            ("lr",  LogisticRegression(max_iter=1000, random_state=42)),
            ("rf",  RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ("xgb", XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0)),
        ],
        final_estimator=LogisticRegression(),
        cv=5, n_jobs=-1
    )
    res = _fit_evaluate(stacking, X_smote, y_smote, X_test, y_test,
                        "Stacking (LR+RF+XGB) + SMOTE")
    print(f"   → AUROC={res['AUROC']}  F1={res['F1']}  Recall={res['Recall']}")
    results.append(res)

    return results
