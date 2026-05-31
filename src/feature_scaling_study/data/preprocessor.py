"""Preprocessing pipeline for feature scaling study.

Applies: missing value imputation (median), stratified split,
and 6 scaling techniques (+ raw baseline).

Scalers:
- Raw (no scaling)
- Min-Max Normalization [0, 1]
- Z-Score Standardization (zero mean, unit variance)
- Robust Scaling (IQR-based)
- MaxAbsScaler (scale to [-1, 1] preserving sparsity)
- QuantileTransformer (non-linear, uniform output)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import MaxAbsScaler, QuantileTransformer


SCALERS = {
    "raw": None,
    "minmax": MinMaxScaler(),
    "zscore": StandardScaler(),
    "robust": RobustScaler(),
    "maxabs": MaxAbsScaler(),
    "quantile": QuantileTransformer(output_distribution="uniform", random_state=42, n_quantiles=100),
}

SCALER_LABELS = {
    "raw": "None (Raw)",
    "minmax": "Min-Max Normalization",
    "zscore": "Z-Score Standardization",
    "robust": "Robust Scaling (IQR)",
    "maxabs": "MaxAbs Scaling",
    "quantile": "Quantile Transformation (Uniform)",
}


def impute_invalid_zeros(X: pd.DataFrame) -> pd.DataFrame:
    """Replace biologically invalid zeros with median (per original paper).

    Applies to: Glucose, BloodPressure, SkinThickness, Insulin, BMI.
    Only works if these column names exist.
    """
    cols_to_clean = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    X = X.copy()
    for col in cols_to_clean:
        if col in X.columns:
            median_val = X.loc[X[col] > 0, col].median()
            X[col] = X[col].replace(0, median_val)
    return X


def apply_scaler(X_train: pd.DataFrame, X_test: pd.DataFrame,
                 scaler_name: str) -> tuple:
    """Apply a scaler to train/test sets.

    Returns (X_train_scaled, X_test_scaled) as DataFrames.
    For 'raw', returns original data unchanged.
    """
    if scaler_name not in SCALERS:
        raise ValueError(f"Unknown scaler: {scaler_name}. "
                         f"Choose from {list(SCALERS.keys())}")

    if scaler_name == "raw":
        return X_train.copy(), X_test.copy()

    scaler = SCALERS[scaler_name]
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Return as DataFrames with original column names
    X_train_scaled = pd.DataFrame(X_train_scaled,
                                  columns=X_train.columns,
                                  index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled,
                                 columns=X_test.columns,
                                 index=X_test.index)
    return X_train_scaled, X_test_scaled


def preprocess_pipeline(dataset: dict, random_state: int = 42):
    """Full preprocessing pipeline for one dataset.

    Yields (scaler_name, X_train, X_test, y_train, y_test) for each scaler.
    """
    from sklearn.model_selection import train_test_split

    X = dataset["X"]
    y = dataset["y"]

    # Step 1: Impute invalid zeros (if Pima)
    if "Glucose" in X.columns:
        X = impute_invalid_zeros(X)

    # Step 2: Stratified split (70:30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=random_state,
    )

    # Step 3: Apply each scaler
    for scaler_name in SCALERS:
        X_train_s, X_test_s = apply_scaler(X_train, X_test, scaler_name)
        yield scaler_name, X_train_s, X_test_s, y_train, y_test
