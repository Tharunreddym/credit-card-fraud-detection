"""
data_loader.py — Load and preprocess the credit card fraud dataset.
Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Place creditcard.csv inside the data/ folder before running.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_and_preprocess(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Load creditcard.csv, scale Amount & Time, and return stratified train/test splits.

    Returns:
        X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(csv_path)

    # ---- Dataset info ----
    fraud_count  = df["Class"].sum()
    total        = len(df)
    normal_count = total - fraud_count
    print(f"   Total rows     : {total:,}")
    print(f"   Normal         : {normal_count:,} ({normal_count/total*100:.3f}%)")
    print(f"   Fraud          : {fraud_count:,}  ({fraud_count/total*100:.3f}%)")
    print(f"   Imbalance ratio: {normal_count // fraud_count}:1")
    print(f"   Missing values : {df.isnull().sum().sum()}")

    # ---- Feature engineering ----
    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    df["Time_scaled"]   = scaler.fit_transform(df[["Time"]])
    df.drop(["Amount", "Time"], axis=1, inplace=True)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # ---- Stratified split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"   Train size     : {len(X_train):,}")
    print(f"   Test  size     : {len(X_test):,}")

    return X_train, X_test, y_train, y_test
