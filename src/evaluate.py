"""Đánh giá hồi quy, phân lớp mức PM2.5 và các baseline thời gian."""

from __future__ import annotations

from typing import Any

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


def classify_pm25(
    values,
    low_max: float = 12.0,
    medium_max: float = 35.5,
    labels: list[str] | tuple[str, ...] = VALID_LABELS,
):
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
    """Baseline persistence: giá trị giờ tới bằng quan trắc hiện tại (t+1 = t)."""
    return frame[target_column].to_numpy(dtype=float)


def seasonal_naive_predictions(frame: pd.DataFrame, target_column: str) -> np.ndarray:
    r"""Baseline chu kỳ 24 giờ: dự báo t+1 bằng quan trắc tại cùng giờ hôm trước (t - 23h).

    Công thức: \hat{y}_{t+1}^{seasonal24} = y_{(t+1)-24} = y_{t-23}.
    Fallback về persistence nếu lịch sử chưa đủ 24 giờ.
    """
    return frame["seasonal_naive_24h"].fillna(frame[target_column]).to_numpy(dtype=float)



def compute_mase(y_true, y_pred, y_naive) -> float:
    """Tính MASE (Mean Absolute Scaled Error) so với naive baseline: MAE(model) / MAE(naive)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    y_naive_arr = np.asarray(y_naive, dtype=float)

    mae_model = float(mean_absolute_error(y_true_arr, y_pred_arr))
    mae_naive = float(mean_absolute_error(y_true_arr, y_naive_arr))
    if mae_naive == 0.0:
        return 1.0 if mae_model == 0.0 else float("inf")
    return float(mae_model / mae_naive)


def compute_skill_score(y_true, y_pred, y_baseline) -> float:
    """Tính MAE Skill Score so với baseline: 1 - MAE(model) / MAE(baseline)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    y_base_arr = np.asarray(y_baseline, dtype=float)

    mae_model = float(mean_absolute_error(y_true_arr, y_pred_arr))
    mae_base = float(mean_absolute_error(y_true_arr, y_base_arr))
    if mae_base == 0.0:
        return 0.0
    return float(1.0 - (mae_model / mae_base))


def conformal_interval_metrics(y_true, lower, upper) -> dict[str, float]:
    """Đo lường độ phủ thực tế (PICP) và độ rộng (MPIW / median width) của khoảng Conformal."""
    y_true_arr = np.asarray(y_true, dtype=float)
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)
    widths = upper_arr - lower_arr
    covered = (y_true_arr >= lower_arr) & (y_true_arr <= upper_arr)
    return {
        "picp": float(np.mean(covered)) if len(covered) > 0 else 0.0,
        "mean_interval_width": float(np.mean(widths)) if len(widths) > 0 else 0.0,
        "median_interval_width": float(np.median(widths)) if len(widths) > 0 else 0.0,
    }


def metrics_by_station(
    frame: pd.DataFrame,
    predictions,
    station_column: str,
    thresholds: dict[str, float],
    conformal_residual_q90: float | None = None,
) -> dict:
    """Tính metric riêng cho từng trạm quan trắc, bao gồm cả conformal coverage nếu có."""
    result = {}
    values = np.asarray(predictions, dtype=float)
    for station, indices in frame.groupby(station_column).indices.items():
        subset = frame.iloc[indices]["target_next_hour"]
        station_preds = values[indices]
        station_metrics = regression_and_classification_metrics(
            subset, station_preds, thresholds
        )
        if conformal_residual_q90 is not None:
            lower = np.maximum(0.0, station_preds - conformal_residual_q90)
            upper = station_preds + conformal_residual_q90
            station_metrics["conformal_interval"] = conformal_interval_metrics(subset, lower, upper)
        result[str(station)] = station_metrics
    return result


def sliced_error_analysis(
    frame: pd.DataFrame,
    predictions,
    thresholds: dict[str, float],
    timestamp_column: str = "timestamp",
    station_column: str = "station",
    target_column: str = "target_next_hour",
) -> dict[str, Any]:
    """Phân tích lỗi đa chiều theo: Trạm (Station), Giờ trong ngày (Hour-of-day), và Mức ô nhiễm."""
    y_true = frame[target_column].to_numpy(dtype=float)
    y_pred = np.asarray(predictions, dtype=float)
    low_max, medium_max, labels = get_threshold_params(thresholds)

    # 1. Theo giờ trong ngày (0..23)
    hours = pd.to_datetime(frame[timestamp_column]).dt.hour
    by_hour = {}
    for hour_val in sorted(hours.unique()):
        mask = (hours == hour_val).to_numpy()
        if mask.any():
            by_hour[int(hour_val)] = {
                "count": int(mask.sum()),
                "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
                "rmse": float(mean_squared_error(y_true[mask], y_pred[mask]) ** 0.5),
            }

    # 2. Theo mức ô nhiễm thực tế (Low / Medium / High)
    regimes = classify_pm25(y_true, low_max, medium_max, labels)
    by_regime = {}
    for label in labels:
        mask = regimes == label
        if mask.any():
            by_regime[str(label)] = {
                "count": int(mask.sum()),
                "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
                "rmse": float(mean_squared_error(y_true[mask], y_pred[mask]) ** 0.5),
            }

    return {
        "by_hour_of_day": by_hour,
        "by_pollution_regime": by_regime,
    }

