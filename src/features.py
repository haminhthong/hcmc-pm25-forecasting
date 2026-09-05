r"""Tạo đặc trưng thời gian và lịch sử mà không sử dụng dữ liệu tương lai.

Nguyên tắc thiết kế chống Data Leakage:
1. Clock-time lag, not row-position lag: Lag được tra theo đúng mốc thời gian thực
   (timestamp - lag), không dịch dòng mù quáng (shift). Nếu thiếu quan trắc, lag nhận NaN.
2. Rolling window với closed='left': Thống kê rolling (mean, std) chỉ tính trên lịch sử
   quan trắc nghiêm ngặt TRƯỚC thời điểm t.
3. Current PM2.5 at t: Quan trắc tại mốc t được dùng làm đặc trưng hiện tại hợp lệ vì mục
   tiêu là dự báo t+1.
4. Seasonal Naive 24h: Giá trị cùng giờ hôm trước cho t+1 được tính theo công thức:
   \hat{y}_{t+1}^{seasonal24} = y_{(t+1)-24} = y_{t-23} (tức quan trắc tại t-23h).
"""


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
    """Tra PM2.5 tại một độ lệch giờ chính xác trong cùng trạm (Exact Clock-Time Lookup).

    Hàm dùng khóa ``(station, timestamp + offset_hours)`` thay vì dịch theo số dòng
    (row-position shift). Vì vậy, nếu dữ liệu bị khuyết một giờ, lag 1 giờ sẽ là NaN
    thay vì lấy nhầm quan trắc gần nhất cách đó nhiều giờ.
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
    include_std: bool = True,
) -> pd.DataFrame:
    """Tạo rolling mean và rolling std theo cửa sổ giờ với closed='left'.

    closed='left' đảm bảo quan trắc tại thời điểm t hiện tại bị loại trừ khỏi cửa sổ rolling.
    """
    result = frame.copy()
    for window in windows:
        mean_col = f"{target_column}_rolling_mean_{window}"
        result[mean_col] = np.nan
        std_col = f"{target_column}_rolling_std_{window}" if include_std else None
        if std_col:
            result[std_col] = np.nan

        for _, station_frame in result.groupby(station_column, sort=False):
            series = station_frame.set_index(timestamp_column)[target_column]
            rolling = series.rolling(
                f"{window}h",
                closed="left",
                min_periods=1,
            )
            result.loc[station_frame.index, mean_col] = rolling.mean().to_numpy()
            if std_col:
                result.loc[station_frame.index, std_col] = rolling.std().to_numpy()
    return result


def add_trend_features(
    frame: pd.DataFrame,
    target_column: str,
    delta_lags: list[int] = (1, 3),
) -> pd.DataFrame:
    """Tạo đặc trưng độ dốc/chênh lệch PM2.5 theo mốc giờ thực: delta = y(t) - y(t-lag)."""
    result = frame.copy()
    for lag in delta_lags:
        lag_col = f"{target_column}_lag_{lag}"
        if lag_col in result.columns:
            result[f"{target_column}_delta_{lag}h"] = result[target_column] - result[lag_col]
    return result


def add_time_features(frame: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    """Mã hóa giờ theo chu kỳ (cyclic encoding sin/cos) và bổ sung thứ trong tuần."""
    result = frame.copy()
    hour = result[timestamp_column].dt.hour
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["day_of_week"] = result[timestamp_column].dt.dayofweek
    return result


def build_features(
    frame: pd.DataFrame, config: dict[str, Any], include_target: bool = True
) -> pd.DataFrame:
    """Tạo đặc trưng chỉ từ quan sát hiện tại/quá khứ trong từng trạm."""
    data_config = config["data"]
    station = data_config["station_column"]
    timestamp = data_config["timestamp_column"]
    target = data_config["target_column"]
    result = frame.sort_values([station, timestamp], kind="stable").copy()

    # 1. Clock-time lags: tra cứu theo số giờ thực tế, không dịch chuyển vị trí dòng
    for lag in config["features"]["lags"]:
        result[f"{target}_lag_{lag}"] = lookup_pm25_at_offset(
            result,
            station,
            timestamp,
            target,
            offset_hours=-lag,
        )

    # 2. Trend differences: chênh lệch nồng độ so với 1h và 3h trước
    delta_lags = config["features"].get("delta_lags", [1, 3])
    result = add_trend_features(result, target, delta_lags=delta_lags)

    # 3. Rolling statistics (mean & std) với closed='left' (chỉ dùng lịch sử trước t)
    include_rolling_std = config["features"].get("include_rolling_std", True)
    result = add_rolling_features(
        result,
        station,
        timestamp,
        target,
        config["features"]["rolling_windows"],
        include_std=include_rolling_std,
    )

    # 4. Cyclic time features
    result = add_time_features(result, timestamp)

    # 5. Target & Baselines
    if include_target:
        result["target_timestamp"] = result[timestamp] + pd.to_timedelta(1, unit="h")
        result["target_next_hour"] = lookup_pm25_at_offset(
            result,
            station,
            timestamp,
            target,
            offset_hours=1,
        )
        # Seasonal Naive 24h cho target(t+1):
        # \hat{y}_{t+1}^{seasonal24} = y_{(t+1)-24} = y_{t-23}
        # Tra cứu quan trắc tại cùng trạm ở mốc t - 23h.
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
    rolling_means = [
        f"{target}_rolling_mean_{window}" for window in config["features"]["rolling_windows"]
    ]
    include_rolling_std = config["features"].get("include_rolling_std", True)
    rolling_stds = (
        [f"{target}_rolling_std_{window}" for window in config["features"]["rolling_windows"]]
        if include_rolling_std
        else []
    )
    delta_lags = config["features"].get("delta_lags", [1, 3])
    deltas = [
        f"{target}_delta_{lag}h"
        for lag in delta_lags
        if lag in config["features"].get("lags", [])
    ]
    return [
        target,
        *history,
        *deltas,
        *rolling_means,
        *rolling_stds,
        *config["features"]["exogenous_columns"],
        "hour_sin",
        "hour_cos",
        "day_of_week",
    ]

