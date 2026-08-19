from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error


VALID_LABELS = ("Tốt", "Trung bình", "Xấu")


def classify_pm25(values, good_max: float, moderate_max: float):
    array = np.asarray(values, dtype=float)
    return np.select([array <= good_max, array < moderate_max], VALID_LABELS[:2], default=VALID_LABELS[2])


def regression_and_classification_metrics(y_true, y_pred, thresholds: dict[str, float]):
    true_labels = classify_pm25(y_true, thresholds["good_max"], thresholds["moderate_max"])
    predicted_labels = classify_pm25(y_pred, thresholds["good_max"], thresholds["moderate_max"])
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "macro_f1": float(f1_score(true_labels, predicted_labels, labels=VALID_LABELS, average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(true_labels, predicted_labels, labels=VALID_LABELS, weights="quadratic")),
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels, labels=VALID_LABELS).tolist(),
    }

