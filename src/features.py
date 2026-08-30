"""Tạo đặc trưng thời gian và lịch sử mà không sử dụng dữ liệu tương lai."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def lookup_pm25_at_offset(
    frame: pd.DataFrame,
    station_column: str,
    timestamp_column: str,
    target_column: str,
    offset_hours: int,
) -> np.ndarray:
    """Tra PM2.5 tại một độ lệch giờ chính xác trong cùng trạm.

    Hàm dùng khóa ``(station, timestamp)`` thay vì dịch theo số dòng. Vì vậy,
    nếu dữ liệu bị khuyết một giờ, lag 1 giờ sẽ là thiếu thay vì lấy nhầm quan
    trắc gần nhất cách đó nhiều giờ.
    """
    lookup = frame.set_index([station_column, timestamp_column])[target_column]
    lookup_keys = pd.MultiIndex.from_arrays(
        [
            frame[station_column],
            frame[timestamp_column] + pd.to_timedelta(offset_hours, unit="h"),
        ],
        names=[station_column, timestamp_column],
    )
    return lookup.reindex(lookup_keys).to_numpy()


def add_rolling_features(
    frame: pd.DataFrame,
    station_column: str,
    timestamp_column: str,
    target_column: str,
    windows: list[int],
) -> pd.DataFrame:
    """Tạo rolling mean theo cửa sổ giờ và loại trừ quan trắc hiện tại."""
    result = frame.copy()
    for window in windows:
        column = f"{target_column}_rolling_mean_{window}"
        result[column] = np.nan
        for _, station_frame in result.groupby(station_column, sort=False):
            series = station_frame.set_index(timestamp_column)[target_column]
            rolling = series.rolling(
                f"{window}h",
                closed="left",
                min_periods=1,
            ).mean()
            result.loc[station_frame.index, column] = rolling.to_numpy()
    return result


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

    # Lag được tra theo số giờ thực tế, không theo vị trí dòng.
    for lag in config["features"]["lags"]:
        result[f"{target}_lag_{lag}"] = lookup_pm25_at_offset(
            result,
            station,
            timestamp,
            target,
            offset_hours=-lag,
        )

    result = add_rolling_features(
        result,
        station,
        timestamp,
        target,
        config["features"]["rolling_windows"],
    )
    result = add_time_features(result, timestamp)
    if include_target:
        result["target_next_hour"] = lookup_pm25_at_offset(
            result,
            station,
            timestamp,
            target,
            offset_hours=1,
        )
        # Dự báo cho t+1 bằng giá trị cùng giờ hôm trước, tức quan trắc tại t-23h.
        result["seasonal_naive_24h"] = lookup_pm25_at_offset(
            result,
            station,
            timestamp,
            target,
            offset_hours=-23,
        )
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
