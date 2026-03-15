"""
imbalance.py — Apply SMOTE, ADASYN, Undersampling, and SMOTEENN to training data.
"""

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN


def apply_samplers(X_train, y_train, random_state: int = 42) -> dict:
    """
    Apply five imbalance-handling strategies to the training set.

    Returns:
        dict mapping sampler name -> (X_resampled, y_resampled)
        Also includes 'ClassWeight' key with original data (no resampling).
    """
    samplers = {
        "SMOTE"              : SMOTE(random_state=random_state),
        "ADASYN"             : ADASYN(random_state=random_state),
        "RandomUnderSampler" : RandomUnderSampler(random_state=random_state),
        "SMOTEENN"           : SMOTEENN(random_state=random_state),
    }

    resampled = {}

    # No resampling — class_weight='balanced' will be set in models.py
    resampled["ClassWeight"] = (X_train, y_train)
    print(f"   {'ClassWeight':<22}: original  total={len(y_train):,}")

    for name, sampler in samplers.items():
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        resampled[name] = (X_res, y_res)
        fraud  = y_res.sum()
        normal = len(y_res) - fraud
        print(f"   {name:<22}: total={len(y_res):,}  normal={normal:,}  fraud={fraud:,}  ratio={normal//fraud}:1")

    return resampled
