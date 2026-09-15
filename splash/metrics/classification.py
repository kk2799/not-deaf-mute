"""Classification metrics for UATR (clip-level and recording-level).

Reports macro/weighted/per-class F1, accuracy, and a confusion matrix. Label
order is fixed to the dataset's canonical ``label_ids`` so matrices and per-class
arrays are comparable across models/experiments.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)


def classification_metrics(
    y_true,
    y_pred,
    label_ids: list[int],
    label_names: dict[int, str] | None = None,
) -> dict:
    """Core metrics. ``label_ids`` fixes the row/column order of the confusion
    matrix and per-class arrays."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(label_ids)
    names = label_names or {i: str(i) for i in labels}

    f1_per = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "f1_per_class": {names[labels[i]]: float(f1_per[i]) for i in range(len(labels))},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [names[l] for l in labels],
    }


def format_metrics(m: dict) -> str:
    """One-line human-readable summary for logging."""
    return (
        f"n={m['n']}  acc={m['accuracy']:.3f}  "
        f"F1(macro)={m['f1_macro']:.3f}  F1(w)={m['f1_weighted']:.3f}"
    )
