import pandas as pd
import pytest

from src.data import validate_schema
from src.features import build_features


CONFIG = {
    "data": {"station_column": "station", "timestamp_column": "timestamp", "target_column": "PM2.5"},
    "features": {"lags": [1], "rolling_windows": [2], "exogenous_columns": ["O3", "SO2"]},
}


def test_schema_csv():
    with pytest.raises(ValueError, match="station"):
        validate_schema(pd.DataFrame({"timestamp": [], "PM2.5": []}), ["timestamp", "station", "PM2.5"])


def test_sort_and_lag_do_not_use_future():
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01 02:00", "2024-01-01 00:00", "2024-01-01 01:00"]),
        "station": ["A", "A", "A"], "PM2.5": [30.0, 10.0, 20.0], "O3": [1, 1, 1], "SO2": [1, 1, 1],
    })
    result = build_features(frame, CONFIG)
    assert result["timestamp"].is_monotonic_increasing
    assert pd.isna(result.iloc[0]["PM2.5_lag_1"])
    assert result.iloc[1]["PM2.5_lag_1"] == 10.0

