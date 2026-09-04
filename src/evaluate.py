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

VALID_LABELS = ("Thấp", "Trung bình", "Cao")


def get_threshold_params(thresholds: dict) -> tuple[float, float, list[str]]:
    """Trích xuất ngưỡng và nhãn từ cấu hình thresholds."""
    low_max = float(thresholds.get("low_max", thresholds.get("good_max", 12.0)))
    medium_max = float(thresholds.get("medium_max", thresholds.get("moderate_max", 35.5)))
    labels = list(thresholds.get("labels", VALID_LABELS))
    return low_max, medium_max, labels


def classify_pm25(values, low_max: float = 12.0, medium_max: float = 35.5, labels: list[str] | tuple[str, ...] = VALID_LABELS):
    """Chuyển nồng độ PM2.5 thành ba mức phân tích nội bộ (Thấp, Trung bình, Cao)."""
    array = np.asarray(values, dtype=float)
    label_list = list(labels)
    return np.select([array <= low_max, array < medium_max], label_list[:2], default=label_list[2])


def regression_and_classification_metrics(
    y_true,
    y_pred,
    thresholds: dict,
) -> dict:
    """Tính đồng thời metric hồi quy (MAE, RMSE, Bias, P90 AE), phân lớp và nhóm PM2.5 cao."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    low_max, medium_max, labels = get_threshold_params(thresholds)

    true_labels = classify_pm25(y_true_arr, low_max, medium_max, labels)
    predicted_labels = classify_pm25(y_pred_arr, low_max, medium_max, labels)
    high_mask = y_true_arr >= medium_max

    if len(set(true_labels) | set(predicted_labels)) < 2:
        qwk = None
    else:
        qwk_value = float(
            cohen_kappa_score(
                true_labels,
                predicted_labels,
                labels=labels,
                weights="quadratic",
            )
        )
        qwk = qwk_value if np.isfinite(qwk_value) else None

    abs_errors = np.abs(y_true_arr - y_pred_arr)
    report = classification_report(
        true_labels, predicted_labels, labels=labels, output_dict=True, zero_division=0
    )

    high_label = labels[2] if len(labels) > 2 else "Cao"
    high_recall = float(report[high_label]["recall"]) if high_label in report else 0.0

    result = {
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(mean_squared_error(y_true_arr, y_pred_arr) ** 0.5),
        "bias": float(np.mean(y_pred_arr - y_true_arr)),
        "p90_absolute_error": float(np.quantile(abs_errors, 0.9)),
        "macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "qwk": qwk,
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels, labels=labels).tolist(),
        "classification_report": report,
    }
    result["high_pm25_mae"] = (
        float(mean_absolute_error(y_true_arr[high_mask], y_pred_arr[high_mask]))
        if high_mask.any()
        else None
    )
    result["high_pm25_recall"] = high_recall
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
