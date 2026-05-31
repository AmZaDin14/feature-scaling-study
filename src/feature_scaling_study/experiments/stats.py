"""Statistical analysis: Wilcoxon signed-rank test across scaler comparisons.

Compares each scaler pair across all dataset × classifier combinations.
Outputs: significance tables and effect sizes.

Supports both binary and multi-class evaluation contexts.
"""

import warnings
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

SCALER_ORDER = ["raw", "minmax", "zscore", "robust", "maxabs", "quantile"]
SCALER_LABELS = {
    "raw": "None (Raw)",
    "minmax": "Min-Max",
    "zscore": "Z-Score",
    "robust": "Robust",
    "maxabs": "MaxAbs",
    "quantile": "Quantile",
}


def _pairwise_wilcoxon(df: pd.DataFrame, metric: str = "accuracy") -> pd.DataFrame:
    """Run Wilcoxon signed-rank for each scaler pair on a given metric.

    Returns a matrix DataFrame with scaler1 × scaler2 → (statistic, p-value).
    """
    scalers = [s for s in SCALER_ORDER if s in df["scaler"].unique()]
    results = []

    for s1 in scalers:
        for s2 in scalers:
            if s1 == s2:
                continue

            # Align by (dataset, classifier, k_value) for paired test
            merged = df[df["scaler"] == s1][
                ["dataset_key", "classifier", "k_value", metric]
            ].merge(
                df[df["scaler"] == s2][
                    ["dataset_key", "classifier", "k_value", metric]
                ],
                on=["dataset_key", "classifier", "k_value"],
                suffixes=("_1", "_2"),
            )

            if len(merged) < 5:
                continue

            try:
                stat, p = wilcoxon(
                    merged[f"{metric}_1"], merged[f"{metric}_2"], alternative="two-sided"
                )
                # Direction
                mean_1 = merged[f"{metric}_1"].mean()
                mean_2 = merged[f"{metric}_2"].mean()
                direction = ">" if mean_1 > mean_2 else ("<" if mean_1 < mean_2 else "=")

                results.append({
                    "scaler_1": SCALER_LABELS[s1],
                    "scaler_2": SCALER_LABELS[s2],
                    "statistic": round(stat, 2),
                    "p_value": round(p, 6),
                    "significant": p < 0.05,
                    "direction": direction,
                    "mean_diff": round(mean_1 - mean_2, 4),
                    "n_pairs": len(merged),
                })
            except (ValueError, RuntimeError):
                continue

    return pd.DataFrame(results)


def full_statistical_report(df: pd.DataFrame) -> dict:
    """Generate full statistical report across key metrics."""
    metrics = ["accuracy", "f1", "auc"]
    report = {}

    for metric in metrics:
        report[metric] = _pairwise_wilcoxon(df, metric)

    return report


def best_scaler_per_dataset_classifier(df: pd.DataFrame) -> pd.DataFrame:
    """For each (dataset, classifier), find which scaler performs best."""
    idx = df.groupby(["dataset_key", "classifier"])["accuracy"].idxmax()
    best = df.loc[idx, [
        "dataset", "classifier_label", "scaler_label",
        "accuracy", "f1", "auc", "k_value"
    ]].copy()
    best = best.sort_values(["dataset", "classifier_label"])
    return best


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate: mean accuracy per scaler per dataset."""
    return df.groupby(["dataset", "scaler_label"])["accuracy"].agg(
        ["mean", "std", "max"]
    ).round(4)


if __name__ == "__main__":
    df = pd.read_csv("results/results.csv")

    print("=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    # Best per combination
    print("\n--- Best scaler per dataset × classifier ---")
    best = best_scaler_per_dataset_classifier(df)
    print(best.to_string(index=False))

    # Wilcoxon
    print("\n\n--- Wilcoxon Signed-Rank (accuracy) ---")
    stats = full_statistical_report(df)
    acc_stats = stats["accuracy"]
    if len(acc_stats) > 0:
        sig_count = acc_stats["significant"].sum()
        total = len(acc_stats)
        print(f"  {sig_count}/{total} comparisons significant (p<0.05)")
        print(acc_stats.to_string(index=False))

    # Summary
    print("\n\n--- Accuracy summary per scaler per dataset ---")
    summ = summary_table(df)
    print(summ.to_string())

    # Overall rank
    print("\n\n--- Overall scaler ranking ---")
    ranking = df.groupby("scaler_label")["accuracy"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    print(ranking.to_string())
    print("\n✓ Statistical analysis complete")
