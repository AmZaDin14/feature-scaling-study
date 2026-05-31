"""Unified dataset loader for feature scaling study.

Loads 6 benchmark datasets:
- Pima Indians Diabetes (binary, medical)
- Breast Cancer Wisconsin (binary, medical)
- Wine (multi-class, chemical)
- Digits (multi-class, image)
- Yeast (multi-class, biological)
- Segment (multi-class, image segmentation)

Provides both binary and multi-class evaluation contexts.
"""

import urllib.request
import os
import pandas as pd
import numpy as np
from sklearn.datasets import (
    load_breast_cancer, load_wine, load_digits, fetch_openml, load_iris
)


# Path for cached Pima dataset
PIMA_CACHE = os.path.join(os.path.dirname(__file__), "pima_diabetes.csv")
PIMA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
PIMA_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]


def _load_pima() -> tuple:
    """Load Pima Indians Diabetes from local cache or download."""
    if os.path.exists(PIMA_CACHE):
        df = pd.read_csv(PIMA_CACHE)
    else:
        try:
            df = pd.read_csv(PIMA_URL, names=PIMA_COLUMNS)
            df.to_csv(PIMA_CACHE, index=False)
            print(f"  [cache: downloaded Pima to {PIMA_CACHE}]")
        except Exception:
            raise RuntimeError(
                "Cannot download Pima dataset. "
                f"Place it manually at {PIMA_CACHE} with columns: {PIMA_COLUMNS}"
            )

    X = df.drop("Outcome", axis=1).astype(float)
    y = df["Outcome"].astype(int)
    return X, y, PIMA_COLUMNS[:-1]


def _load_yeast() -> tuple:
    """Load Yeast dataset (10 classes, 8 features, 1,484 samples).

    Downloads from UCI directly. Falls back to synthetic data.
    """
    YEAST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"
    YEAST_COLUMNS = [
        "name", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc", "class"
    ]
    YEAST_CLASSES = [
        "MIT", "NUC", "CYT", "ME1", "ME2", "ME3", "EXC", "VAC", "POX", "ERL"
    ]
    try:
        raw = urllib.request.urlopen(YEAST_URL, timeout=10).read().decode()
        rows = [line.strip().split() for line in raw.strip().split("\n")]
        df = pd.DataFrame(rows, columns=YEAST_COLUMNS)
        # Drop name column (index 0)
        X = df.drop(["name", "class"], axis=1).astype(float)
        y = df["class"].map({c: i for i, c in enumerate(YEAST_CLASSES)}).astype(int)
        print(f"  [loaded yeast from UCI: {len(X)} samples, {X.shape[1]} features, {y.nunique()} classes]")
        return X, y, YEAST_COLUMNS[1:9]
    except Exception as e:
        print(f"  [UCI yeast unavailable ({e}), using synthetic multi-class fallback]")
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=500, n_features=10, n_classes=5, n_informative=8,
            n_redundant=1, n_repeated=0, class_sep=1.2, random_state=42
        )
        feature_names = [f"feat_{i}" for i in range(10)]
        return pd.DataFrame(X, columns=feature_names), pd.Series(y), feature_names


def _load_segment() -> tuple:
    """Load Segment dataset (7 classes, 19 features, 2,310 samples).

    Downloads from UCI Statlog repository directly. Falls back to synthetic data.
    """
    SEGMENT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/segment/segment.dat"
    try:
        raw = urllib.request.urlopen(SEGMENT_URL, timeout=10).read().decode()
        rows = [line.strip().split() for line in raw.strip().split("\n")]
        df = pd.DataFrame(rows, dtype=float)
        X = df.iloc[:, :-1].astype(float)
        y = df.iloc[:, -1].astype(int) - 1  # Classes are 1-7, convert to 0-6
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X.columns = feature_names
        print(f"  [loaded segment from UCI: {len(X)} samples, {X.shape[1]} features, {y.nunique()} classes]")
        return X, y, feature_names
    except Exception as e:
        print(f"  [UCI segment unavailable ({e}), using synthetic multi-class fallback]")
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=500, n_features=15, n_classes=4, n_informative=10,
            n_redundant=2, n_repeated=0, class_sep=0.8, random_state=99
        )
        feature_names = [f"feat_{i}" for i in range(15)]
        return pd.DataFrame(X, columns=feature_names), pd.Series(y), feature_names


DATASET_LOADERS = {
    "pima_diabetes": {
        "loader": _load_pima,
        "name": "Pima Indians Diabetes",
    },
    "breast_cancer": {
        "loader": lambda: (
            load_breast_cancer(as_frame=True).data.astype(float),
            load_breast_cancer(as_frame=True).target.astype(int),
            list(load_breast_cancer()["feature_names"]),
        ),
        "name": "Breast Cancer Wisconsin",
    },
    "wine": {
        "loader": lambda: (
            load_wine(as_frame=True).data.astype(float),
            load_wine(as_frame=True).target.astype(int),  # 3-class, no binarization
            list(load_wine()["feature_names"]),
        ),
        "name": "Wine",
    },
    "digits": {
        "loader": lambda: (
            load_digits(as_frame=True).data.astype(float),
            load_digits(as_frame=True).target.astype(int),  # 10-class, no binarization
            [f"pixel_{i}" for i in range(64)],
        ),
        "name": "Digits",
    },
    "yeast": {
        "loader": _load_yeast,
        "name": "Yeast",
    },
    "segment": {
        "loader": _load_segment,
        "name": "Segment",
    },
}


def load_dataset(name: str) -> dict:
    """Load a dataset by name.

    Returns dict with keys: name, X (DataFrame), y (Series),
    feature_names, n_samples, n_features, n_classes, class_distribution.
    """
    config = DATASET_LOADERS[name]

    X, y, feature_names = config["loader"]()

    # Ensure DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=feature_names)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    return {
        "name": config["name"],
        "X": X,
        "y": y,
        "feature_names": list(feature_names),
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_classes": len(np.unique(y)),
        "class_distribution": y.value_counts().to_dict(),
    }


def load_all_datasets() -> dict[str, dict]:
    """Load all 6 datasets."""
    datasets = {}
    for name in DATASET_LOADERS:
        print(f"Loading {name}...")
        datasets[name] = load_dataset(name)
        d = datasets[name]
        print(f"  → {d['n_samples']} samples, {d['n_features']} features, "
              f"{d['n_classes']} classes, dist={d['class_distribution']}")
    return datasets


if __name__ == "__main__":
    datasets = load_all_datasets()
    print("\n✓ All datasets loaded.")
