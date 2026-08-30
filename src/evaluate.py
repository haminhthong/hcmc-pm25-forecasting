"""Đánh giá hồi quy, phân lớp mức PM2.5 và các baseline thời gian."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

VALID_LABELS = ("Tốt", "Trung bình", "Xấu")


def classify_pm25(values, good_max: float, moderate_max: float):
    """Chuyển nồng độ PM2.5 thành ba mức chất lượng được cấu hình."""
    array = np.asarray(values, dtype=float)
    return np.select([array <= good_max, array < moderate_max], VALID_LABELS[:2], default=VALID_LABELS[2])


def regression_and_classification_metrics(
    y_true,
    y_pred,
    thresholds: dict[str, float],
) -> dict:
    """Tính đồng thời metric hồi quy, phân lớp và nhóm PM2.5 cao."""
    true_labels = classify_pm25(y_true, thresholds["good_max"], thresholds["moderate_max"])
    predicted_labels = classify_pm25(y_pred, thresholds["good_max"], thresholds["moderate_max"])
    high_mask = np.asarray(y_true, dtype=float) >= thresholds["moderate_max"]
    if len(set(true_labels) | set(predicted_labels)) < 2:
        qwk = None
    else:
        qwk_value = float(
            cohen_kappa_score(
                true_labels,
                predicted_labels,
                labels=VALID_LABELS,
                weights="quadratic",
            )
        )
        qwk = qwk_value if np.isfinite(qwk_value) else None
    result = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=VALID_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "qwk": qwk,
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels, labels=VALID_LABELS).tolist(),
        "classification_report": classification_report(
            true_labels, predicted_labels, labels=VALID_LABELS, output_dict=True, zero_division=0
        ),
    }
    result["high_pm25_mae"] = (
        float(
            mean_absolute_error(
                np.asarray(y_true)[high_mask],
                np.asarray(y_pred)[high_mask],
            )
        )
        if high_mask.any()
        else None
    )
    result["high_pm25_recall"] = float(result["classification_report"]["Xấu"]["recall"])
    return result


def persistence_predictions(frame: pd.DataFrame, target_column: str) -> np.ndarray:
    """Baseline persistence: giá trị giờ tới bằng quan trắc hiện tại."""
    return frame[target_column].to_numpy(dtype=float)


def seasonal_naive_predictions(frame: pd.DataFrame, target_column: str) -> np.ndarray:
    """Baseline chu kỳ 24 giờ; fallback về persistence nếu lịch sử chưa đủ."""
    return frame["seasonal_naive_24h"].fillna(frame[target_column]).to_numpy(dtype=float)


def metrics_by_station(
    frame: pd.DataFrame,
    predictions,
    station_column: str,
    thresholds: dict[str, float],
) -> dict:
    """Tính metric riêng cho từng trạm quan trắc."""
    result = {}
    values = np.asarray(predictions)
    for station, indices in frame.groupby(station_column).indices.items():
        subset = frame.iloc[indices]["target_next_hour"]
        result[str(station)] = regression_and_classification_metrics(subset, values[indices], thresholds)
    return result
