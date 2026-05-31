# /// notebook_name = "feature_scaling_dashboard"
# /// marimo-version = "0.23.8"

import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(r"""## Feature Scaling Study — Interactive Dashboard
Explore **1,504 experiment evaluations** across 4 datasets, 4 scaling techniques, and 7 classifiers.
""")
    return


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    return mo, pd, np, plt, sns


@app.cell
def __(mo, pd):
    df = pd.read_csv("results/results.csv")
    datasets_list = sorted(df["dataset"].unique())
    scaler_labels_map = {
        "raw": "None (Raw)", "minmax": "Min-Max",
        "zscore": "Z-Score", "robust": "Robust",
    }
    df["scaler_label"] = df["scaler"].map(scaler_labels_map)
    colors = {"raw": "#6c757d", "minmax": "#e67e22", "zscore": "#27ae60", "robust": "#3498db"}
    mo.md(f"**Loaded {len(df)} results** from {len(datasets_list)} datasets")
    return df, datasets_list, scaler_labels_map, colors


@app.cell
def __(mo):
    mo.md("## 1. Global Scaler Ranking")
    return


@app.cell
def __(df, mo):
    ranking = df.groupby("scaler_label")["accuracy"].agg(["mean", "std", "max", "min"]).round(4).sort_values("mean", ascending=False)
    mo.ui.table(ranking)
    return ranking,


@app.cell
def __(mo):
    mo.md("## 2. Dataset Explorer")
    return


@app.cell
def __(datasets_list, mo):
    dataset_picker = mo.ui.dropdown(options=datasets_list, value=datasets_list[0], label="Select dataset")
    dataset_picker
    return dataset_picker,


@app.cell
def __(dataset_picker, df, mo):
    ds_subset = df[df["dataset"] == dataset_picker.value]
    best = ds_subset.loc[ds_subset.groupby("classifier")["accuracy"].idxmax()]
    best = best[["classifier_label", "scaler_label", "accuracy", "f1", "auc", "k_value"]].sort_values("accuracy", ascending=False)
    mo.ui.table(best)
    return best, ds_subset


@app.cell
def __(ds_subset, mo, plt, sns):
    pivot = ds_subset.pivot_table(values="accuracy", index="scaler_label", columns="classifier_label", aggfunc="mean")
    _fig, _ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=_ax, cbar_kws={"label": "Mean Accuracy"})
    _ax.set_title("Accuracy by Scaler x Classifier")
    plt.tight_layout()
    mo.mpl.interactive(_ax)
    return pivot,


@app.cell
def __(mo):
    mo.md("## 3. k-NN Stability Curves")
    return


@app.cell
def __(mo):
    knn_metric = mo.ui.dropdown(options=["euclidean", "manhattan", "cosine"], value="euclidean", label="Distance metric")
    knn_scalers = mo.ui.multiselect(options=["raw", "minmax", "zscore", "robust"], value=["raw", "minmax", "zscore", "robust"], label="Scalers")
    mo.hstack([knn_metric, knn_scalers])
    return knn_metric, knn_scalers


@app.cell
def __(colors, ds_subset, knn_metric, knn_scalers, mo, plt, scaler_labels_map):
    clf_key = f"knn_{knn_metric.value}"
    knn_sub = ds_subset[(ds_subset["classifier"] == clf_key) & (ds_subset["scaler"].isin(knn_scalers.value))]
    _fig, _ax = plt.subplots(figsize=(8, 4))
    for s in knn_scalers.value:
        sd = knn_sub[knn_sub["scaler"] == s].sort_values("k_value")
        _ax.plot(sd["k_value"], sd["accuracy"], label=scaler_labels_map[s],
                color=colors[s], linewidth=2, alpha=0.85)
    _ax.set_xlabel("k (neighbors)"); _ax.set_ylabel("Accuracy")
    _ax.legend(); _ax.set_ylim(0.4, 1.05)
    plt.tight_layout()
    mo.mpl.interactive(_ax)
    return clf_key, knn_sub,


@app.cell
def __(colors, ds_subset, mo, plt, sns):
    order = ["raw", "minmax", "zscore", "robust"]
    pal = [colors[s] for s in order]
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=ds_subset, x="scaler", y="accuracy", order=order, palette=pal, ax=_ax, width=0.5)
    sns.stripplot(data=ds_subset, x="scaler", y="accuracy", order=order, color="black", alpha=0.2, size=3, ax=_ax)
    _ax.set_xticklabels(["None (Raw)", "Min-Max", "Z-Score", "Robust"], rotation=15)
    _ax.set_title("Accuracy Distribution by Scaler")
    plt.tight_layout()
    mo.mpl.interactive(_ax)
    return order, pal,


@app.cell
def __(mo):
    mo.md("## 4. Classifier Deep-Dive")
    return


@app.cell
def __(ds_subset, mo):
    clf_picker = mo.ui.dropdown(
        options=sorted(ds_subset["classifier_label"].unique()),
        value=sorted(ds_subset["classifier_label"].unique())[0],
        label="Classifier"
    )
    clf_picker
    return clf_picker,


@app.cell
def __(clf_picker, ds_subset):
    clf_sub = ds_subset[ds_subset["classifier_label"] == clf_picker.value]
    summary = clf_sub.groupby("scaler_label")[["accuracy", "precision", "recall", "f1", "auc"]].mean().round(4)
    summary
    return clf_sub, summary,


if __name__ == "__main__":
    app.run()
