from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for script execution
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


LABEL_NAMES = ["SLI", "TD"]   # alphabetical = sklearn's default sort order


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES, output_dict=True)
    metrics = {
        "accuracy": report["accuracy"],
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "sli_f1": report["SLI"]["f1-score"],
        "sli_recall": report["SLI"]["recall"],
        "sli_precision": report["SLI"]["precision"],
    }
    if y_prob is not None:
        try:
            y_binary = (np.array(y_true) == "SLI").astype(int)
            metrics["auc_roc"] = roc_auc_score(y_binary, y_prob)
        except ValueError:
            pass
    return metrics


def print_metrics(metrics: dict, model_name: str = "") -> None:
    header = f"=== {model_name} ===" if model_name else "=== Results ==="
    print(header)
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  Macro F1   : {metrics['macro_f1']:.4f}")
    print(f"  SLI F1     : {metrics['sli_f1']:.4f}")
    print(f"  SLI Recall : {metrics['sli_recall']:.4f}")
    if "auc_roc" in metrics:
        print(f"  AUC-ROC    : {metrics['auc_roc']:.4f}")


def plot_confusion_matrix(y_true, y_pred, model_name: str = "", save_path: Path | None = None) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(model_name or "Confusion Matrix")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    if not save_path:
        plt.show()


def plot_model_comparison(results: dict[str, dict], save_path: Path | None = None) -> None:
    """
    results: {model_name: metrics_dict}
    """
    models = list(results.keys())
    macro_f1 = [results[m]["macro_f1"] for m in models]
    sli_recall = [results[m]["sli_recall"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, macro_f1, width, label="Macro F1", color="#4C72B0")
    ax.bar(x + width / 2, sli_recall, width, label="SLI Recall", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    if not save_path:
        plt.show()
