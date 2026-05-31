"""Visualization module for the feature scaling study.

Generates:
1. Accuracy heatmap (scalers x classifiers per dataset)
2. t-SNE plots (2D projection per scaler)
3. k-NN stability curves (accuracy vs k per scaler)
4. Boxplot (accuracy distribution per scaler)

All figures saved to results/figures/.
Supports both binary and multi-class datasets.
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.feature_scaling_study.data.loader import load_all_datasets, load_dataset
from src.feature_scaling_study.data.preprocessor import preprocess_pipeline, SCALER_LABELS

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = {
    "raw": "#6c757d", "minmax": "#e67e22", "zscore": "#27ae60",
    "robust": "#3498db", "maxabs": "#9b59b6", "quantile": "#e74c3c"
}
OUTPUT_DIR = "results/figures"


def _ensure_dir():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_accuracy_heatmap(df):
    _ensure_dir()
    datasets = df["dataset"].unique()
    n_datasets = len(datasets)
    n_cols = 3
    n_rows = int(np.ceil(n_datasets / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    axes = axes.flatten()
    for ax, dataset in zip(axes, datasets):
        subset = df[df["dataset"] == dataset]
        pivot = subset.pivot_table(values="accuracy", index="scaler_label", columns="classifier_label", aggfunc="mean")
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Accuracy"})
        ax.set_title(dataset, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Scaler")
    for ax in axes[n_datasets:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/accuracy_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR}/accuracy_heatmap.png")


def plot_tsne_comparison(datasets):
    scaler_keys = ["raw", "minmax", "zscore", "robust", "maxabs", "quantile"]
    for ds_key, dataset in datasets.items():
        n_scalers = len(scaler_keys)
        n_cols = 3
        n_rows = int(np.ceil(n_scalers / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4.5))
        axes = axes.flatten()
        n_classes = dataset["n_classes"]
        is_binary = n_classes <= 2
        for ax, scaler_name in zip(axes, scaler_keys):
            gen = preprocess_pipeline(dataset)
            Xp = yp = None
            for s_name, X_train, _, y_train, _ in gen:
                if s_name == scaler_name:
                    if len(X_train) > 600:
                        idx = np.random.RandomState(42).choice(len(X_train), 600, replace=False)
                        Xp = X_train.iloc[idx]
                        yp = y_train.iloc[idx]
                    else:
                        Xp = X_train
                        yp = y_train
                    break
            if Xp is None or len(Xp) < 3:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
                ax.set_title(SCALER_LABELS[scaler_name])
                continue
            perp = min(30, len(Xp) - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perp, max_iter=500)
            Xt = tsne.fit_transform(Xp)
            if is_binary:
                ax.scatter(Xt[yp == 0, 0], Xt[yp == 0, 1], c=COLORS[scaler_name], alpha=0.6, s=15, label="Class 0")
                ax.scatter(Xt[yp == 1, 0], Xt[yp == 1, 1], c="red", alpha=0.6, s=15, label="Class 1")
            else:
                clrs = plt.cm.tab10(np.linspace(0, 1, min(n_classes, 10)))
                for cls in range(min(n_classes, 10)):
                    mask = yp == cls
                    if mask.sum() > 0:
                        ax.scatter(Xt[mask, 0], Xt[mask, 1], c=[clrs[cls]], alpha=0.6, s=15, label=f"Class {cls}")
            ax.set_title(SCALER_LABELS[scaler_name])
            ax.set_xticks([])
            ax.set_yticks([])
            if scaler_name == "raw":
                ax.legend(fontsize=7, loc="best")
        for ax in axes[n_scalers:]:
            ax.set_visible(False)
        fig.suptitle(dataset["name"], fontweight="bold", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(f"{OUTPUT_DIR}/tsne_{ds_key}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {OUTPUT_DIR}/tsne_{ds_key}.png")


def plot_knn_stability(df):
    _ensure_dir()
    knn_cf = [c for c in df["classifier"].unique() if c.startswith("knn_")]
    knn_lb = {"knn_euclidean": "Euclidean", "knn_manhattan": "Manhattan", "knn_cosine": "Cosine"}
    all_sc = [s for s in ["raw", "minmax", "zscore", "robust", "maxabs", "quantile"] if s in df["scaler"].unique()]
    datasets = df["dataset"].unique()
    for dataset in datasets:
        n_met = len(knn_cf)
        if n_met == 0:
            continue
        fig, axes = plt.subplots(1, n_met, figsize=(6 * n_met, 5))
        if n_met == 1:
            axes = [axes]
        for ax, clf_key in zip(axes, knn_cf):
            subset = df[(df["dataset"] == dataset) & (df["classifier"] == clf_key)]
            for scaler in all_sc:
                sdata = subset[subset["scaler"] == scaler].sort_values("k_value")
                if len(sdata) > 0:
                    ax.plot(sdata["k_value"], sdata["accuracy"],
                            label=SCALER_LABELS[scaler], color=COLORS[scaler], linewidth=2, alpha=0.85)
            ax.set_title(f"k-NN ({knn_lb[clf_key]})")
            ax.set_xlabel("k (neighbors)")
            ax.set_ylabel("Accuracy")
            ax.legend(fontsize=8)
            ax.set_ylim(0.3, 1.0)
        fig.suptitle(dataset, fontweight="bold", fontsize=14)
        fig.tight_layout()
        safe_name = dataset.lower().replace(" ", "_").replace(",", "")
        fig.savefig(f"{OUTPUT_DIR}/knn_stability_{safe_name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: knn_stability plot for {dataset}")


def plot_boxplot(df):
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(10, 5))
    all_sc = [s for s in ["raw", "minmax", "zscore", "robust", "maxabs", "quantile"] if s in df["scaler"].unique()]
    palette = [COLORS[s] for s in all_sc]
    sns.boxplot(data=df, x="scaler", y="accuracy", order=all_sc, palette=palette, ax=ax, width=0.5, fliersize=3)
    sns.stripplot(data=df, x="scaler", y="accuracy", order=all_sc, color="black", alpha=0.1, size=2, ax=ax)
    ax.set_xticklabels([SCALER_LABELS[s] for s in all_sc], rotation=15)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    ax.set_title("Accuracy Distribution by Scaling Technique", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/accuracy_boxplot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR}/accuracy_boxplot.png")


def generate_all_plots(df, datasets):
    _ensure_dir()
    print("Generating visualizations...")
    print("1/4: Accuracy heatmap...")
    plot_accuracy_heatmap(df)
    print("2/4: t-SNE comparisons...")
    plot_tsne_comparison(datasets)
    print("3/4: k-NN stability curves...")
    plot_knn_stability(df)
    print("4/4: Accuracy boxplot...")
    plot_boxplot(df)
    print("All visualizations saved to", OUTPUT_DIR)


if __name__ == "__main__":
    df = pd.read_csv("results/results.csv")
    datasets = load_all_datasets()
    generate_all_plots(df, datasets)
