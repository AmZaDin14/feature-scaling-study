"""Experiment runner — runs all classifier × scaler × dataset combinations.

Uses 5-fold stratified cross-validation for robust estimates with confidence intervals.
Supports both binary and multi-class evaluation.
Saves results to results/results.csv with mean ± std for each metric.
"""

import time
import warnings
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from src.feature_scaling_study.data.loader import load_all_datasets, load_dataset
from src.feature_scaling_study.data.preprocessor import (
    preprocess_pipeline, apply_scaler, impute_invalid_zeros, SCALER_LABELS, SCALERS
)

warnings.filterwarnings("ignore")

N_FOLDS = 5
RANDOM_STATE = 42

# --- Classifier configurations ---

CLASSIFIERS = {
    "knn_euclidean": {
        "label": "k-NN (Euclidean)",
        "factory": lambda: KNeighborsClassifier(metric="euclidean"),
        "needs_k_search": True,
    },
    "knn_manhattan": {
        "label": "k-NN (Manhattan)",
        "factory": lambda: KNeighborsClassifier(metric="manhattan"),
        "needs_k_search": True,
    },
    "knn_cosine": {
        "label": "k-NN (Cosine)",
        "factory": lambda: KNeighborsClassifier(metric="cosine"),
        "needs_k_search": True,
    },
    "svm_linear": {
        "label": "SVM (Linear)",
        "factory": lambda: SVC(kernel="linear", probability=True, random_state=RANDOM_STATE, max_iter=5000),
        "needs_k_search": False,
    },
    "svm_rbf": {
        "label": "SVM (RBF)",
        "factory": lambda: SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, max_iter=5000),
        "needs_k_search": False,
    },
    "decision_tree": {
        "label": "Decision Tree",
        "factory": lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
        "needs_k_search": False,
    },
    "random_forest": {
        "label": "Random Forest",
        "factory": lambda: RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "needs_k_search": False,
    },
}

K_VALUES = list(range(1, 31))


def _calc_multiclass_metrics(cm: np.ndarray) -> tuple:
    """Calculate macro-averaged sensitivity and specificity from confusion matrix.
    
    Returns (sensitivity, specificity) as floats.
    """
    n_classes = cm.shape[0]
    sensitivities = []
    specificities = []
    
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivities.append(sens)
        specificities.append(spec)
    
    return np.mean(sensitivities), np.mean(specificities)


def evaluate(classifier, X_train, y_train, X_test, y_test) -> dict:
    """Train and evaluate a classifier, returning per-fold metrics.
    
    Handles both binary and multi-class automatically.
    """
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)

    n_classes = len(np.unique(y_test))
    
    # Accuracy works for both binary and multi-class
    acc = accuracy_score(y_test, y_pred)
    
    if n_classes == 2:
        # Binary case
        avg_type = "binary"
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba[:, 1])
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        # Multi-class case
        avg_type = "weighted"
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        
        # AUC OVR for multi-class
        try:
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
        except Exception:
            auc = 0.5
        
        cm = confusion_matrix(y_test, y_pred)
        sensitivity, specificity = _calc_multiclass_metrics(cm)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "specificity": specificity,
        "sensitivity": sensitivity,
    }


def run_cv_experiment(dataset_name: str, dataset: dict,
                      clf_key: str, clf_config: dict,
                      k_value: int | None = None) -> list[dict]:
    """Run one classifier across all scalers using 5-fold CV.

    For each scaler:
      1. Impute invalid zeros (if Pima)
      2. Stratified 5-fold split
      3. Within each fold: fit scaler, train classifier, evaluate
      4. Aggregate: store mean ± std of each metric across folds

    Returns list of result dicts (one per scaler).
    """
    results = []
    X = dataset["X"].copy()
    y = dataset["y"].copy()

    # Impute invalid zeros (global preprocessing, not scaling)
    if "Glucose" in X.columns:
        X = impute_invalid_zeros(X)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for scaler_name in SCALERS:
        start = time.time()
        fold_metrics = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Apply scaler within fold
            if scaler_name == "raw":
                X_tr_s, X_val_s = X_train.copy(), X_val.copy()
            else:
                scaler = SCALERS[scaler_name]
                X_tr_s = scaler.fit_transform(X_train)
                X_val_s = scaler.transform(X_val)

            # Build classifier
            clf = clf_config["factory"]()
            if clf_config["needs_k_search"]:
                assert k_value is not None
                clf.set_params(n_neighbors=k_value)

            metrics = evaluate(clf, X_tr_s, y_train, X_val_s, y_val)
            fold_metrics.append(metrics)

        elapsed = time.time() - start

        # Aggregate across folds
        avg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        std = {k: np.std([m[k] for m in fold_metrics]) for k in fold_metrics[0]}

        result = {
            "dataset": dataset["name"],
            "dataset_key": dataset_name,
            "scaler": scaler_name,
            "scaler_label": SCALER_LABELS[scaler_name],
            "classifier": clf_key,
            "classifier_label": clf_config["label"],
            "k_value": k_value if clf_config["needs_k_search"] else "N/A",
            "n_folds": N_FOLDS,
            "accuracy": round(avg["accuracy"], 4),
            "accuracy_std": round(std["accuracy"], 4),
            "precision": round(avg["precision"], 4),
            "precision_std": round(std["precision"], 4),
            "recall": round(avg["recall"], 4),
            "recall_std": round(std["recall"], 4),
            "f1": round(avg["f1"], 4),
            "f1_std": round(std["f1"], 4),
            "auc": round(avg["auc"], 4),
            "auc_std": round(std["auc"], 4),
            "specificity": round(avg["specificity"], 4),
            "specificity_std": round(std["specificity"], 4),
            "sensitivity": round(avg["sensitivity"], 4),
            "sensitivity_std": round(std["sensitivity"], 4),
            "time_seconds": round(elapsed / N_FOLDS, 3),
        }
        results.append(result)

    return results


def run_full_experiment() -> pd.DataFrame:
    """Run all combinations and return results DataFrame."""
    all_results = []
    datasets = load_all_datasets()

    total_configs = 0
    for clf_key, clf_config in CLASSIFIERS.items():
        if clf_config["needs_k_search"]:
            total_configs += len(K_VALUES) * len(SCALERS) * len(datasets)
        else:
            total_configs += len(SCALERS) * len(datasets)
    total_evals = total_configs * N_FOLDS

    print(f"Total configurations: {total_configs} ({total_evals} individual evaluations across {N_FOLDS} folds)")

    for dataset_name, dataset in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset['name']} ({dataset['n_samples']} samples, {dataset['n_classes']} classes)")
        print(f"{'='*60}")

        for clf_key, clf_config in CLASSIFIERS.items():
            print(f"  Classifier: {clf_config['label']}")

            if clf_config["needs_k_search"]:
                for k in K_VALUES:
                    if k % 5 == 0 or k == 1:
                        print(f"    k={k}...")
                    try:
                        results = run_cv_experiment(
                            dataset_name, dataset, clf_key, clf_config, k_value=k
                        )
                        all_results.extend(results)
                    except Exception as e:
                        print(f"      ERROR: {e}")
            else:
                try:
                    results = run_cv_experiment(
                        dataset_name, dataset, clf_key, clf_config
                    )
                    all_results.extend(results)
                except Exception as e:
                    print(f"      ERROR: {e}")

            print(f"    ✓")

    df = pd.DataFrame(all_results)
    return df


if __name__ == "__main__":
    print(f"Starting full experiment with {N_FOLDS}-fold stratified CV...")
    t0 = time.time()
    df = run_full_experiment()
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"Experiment complete in {elapsed:.1f}s")
    print(f"Total results: {len(df)} rows (mean across {N_FOLDS} folds)")
    print(f"Columns: {list(df.columns)}")

    # Save results
    df.to_csv("results/results.csv", index=False)
    df.to_excel("results/results.xlsx", index=False)
    print(f"Saved to results/results.csv and results/results.xlsx")

    # Summary
    print(f"\nBest accuracy per dataset (mean across all classifiers):")
    for ds_key in df["dataset_key"].unique():
        subset = df[df["dataset_key"] == ds_key]
        best = subset.loc[subset.groupby("scaler")["accuracy"].transform("mean").idxmax()]
        print(f"  {best['dataset']}: best scaler = {best['scaler_label']}, "
              f"mean acc = {subset.groupby('scaler')['accuracy'].mean().max():.4f}")
    
    print(f"\nOverall scaler ranking (mean accuracy across all configs):")
    ranking = df.groupby("scaler_label")["accuracy"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    print(ranking.to_string())
