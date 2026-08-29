"""Tạo đặc trưng thời gian và lịch sử mà không sử dụng dữ liệu tương lai."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def add_time_features(frame: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    """Mã hóa giờ theo chu kỳ và bổ sung thứ trong tuần."""
    result = frame.copy()
    hour = result[timestamp_column].dt.hour
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["day_of_week"] = result[timestamp_column].dt.dayofweek
    return result


def build_features(frame: pd.DataFrame, config: dict[str, Any], include_target: bool = True) -> pd.DataFrame:
    """Tạo đặc trưng chỉ từ quan sát hiện tại/quá khứ trong từng trạm."""
    data_config = config["data"]
    station = data_config["station_column"]
    timestamp = data_config["timestamp_column"]
    target = data_config["target_column"]
    result = frame.sort_values([station, timestamp], kind="stable").copy()
    grouped_target = result.groupby(station, sort=False)[target]
    for lag in config["features"]["lags"]:
        result[f"{target}_lag_{lag}"] = grouped_target.shift(lag)
    for window in config["features"]["rolling_windows"]:
        result[f"{target}_rolling_mean_{window}"] = grouped_target.transform(
            lambda values, current_window=window: values.shift(1)
            .rolling(current_window, min_periods=1)
            .mean()
        )
    result = add_time_features(result, timestamp)
    if include_target:
        result["target_next_hour"] = grouped_target.shift(-1)
    return result


def model_feature_columns(config: dict[str, Any]) -> list[str]:
    """Trả về danh sách cột đầu vào theo đúng thứ tự của model."""
    target = config["data"]["target_column"]
    history = [f"{target}_lag_{lag}" for lag in config["features"]["lags"]]
    rolling = [f"{target}_rolling_mean_{window}" for window in config["features"]["rolling_windows"]]
    return [
        target,
        *history,
        *rolling,
        *config["features"]["exogenous_columns"],
        "hour_sin",
        "hour_cos",
        "day_of_week",
    ]
