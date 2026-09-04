import numpy as np
import pandas as pd
import pytest

from src.predict import Predictor


@pytest.fixture
def sample_config():
    return {
        "data": {"station_column": "station", "timestamp_column": "timestamp", "target_column": "PM2.5"},
        "features": {
            "lags": [1, 2, 3, 6, 12, 24],
            "rolling_windows": [3, 6, 24],
            "exogenous_columns": ["O3", "SO2"],
        },
        "thresholds": {"low_max": 12.0, "medium_max": 35.5, "labels": ["Thấp", "Trung bình", "Cao"]},
        "artifacts": {"directory": "artifacts", "model_file": "model.joblib"},
    }


def test_prediction_rejects_multiple_stations(sample_config, monkeypatch):
    class FakeModel:
        def predict(self, df):
            return np.array([20.0])

    monkeypatch.setattr("joblib.load", lambda path: FakeModel())
    predictor = Predictor(sample_config)

    records = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=25, freq="h"),
            "station": ["A"] * 24 + ["B"],
            "PM2.5": [10.0] * 25,
        }
    )
    with pytest.raises(ValueError, match="một trạm duy nhất"):
        predictor.predict(records)


def test_prediction_rejects_insufficient_history(sample_config, monkeypatch):
    class FakeModel:
        def predict(self, df):
            return np.array([20.0])

    monkeypatch.setattr("joblib.load", lambda path: FakeModel())
    predictor = Predictor(sample_config)

    records = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
            "station": ["A"] * 10,
            "PM2.5": [10.0] * 10,
        }
    )
    with pytest.raises(ValueError, match="Cần tối thiểu 25 quan trắc"):
        predictor.predict(records)


def test_prediction_rejects_duplicate_timestamps(sample_config, monkeypatch):
    class FakeModel:
        def predict(self, df):
            return np.array([20.0])

    monkeypatch.setattr("joblib.load", lambda path: FakeModel())
    predictor = Predictor(sample_config)

    dates = list(pd.date_range("2024-01-01", periods=24, freq="h")) + [pd.Timestamp("2024-01-01 00:00")]
    records = pd.DataFrame(
        {
            "timestamp": dates,
            "station": ["A"] * 25,
            "PM2.5": [10.0] * 25,
        }
    )
    with pytest.raises(ValueError, match="timestamp trùng lặp"):
        predictor.predict(records)


def test_prediction_rejects_negative_pm25(sample_config, monkeypatch):
    class FakeModel:
        def predict(self, df):
            return np.array([20.0])

    monkeypatch.setattr("joblib.load", lambda path: FakeModel())
    predictor = Predictor(sample_config)

    records = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=25, freq="h"),
            "station": ["A"] * 25,
            "PM2.5": [10.0] * 24 + [-5.0],
        }
    )
    with pytest.raises(ValueError, match="không được âm"):
        predictor.predict(records)


def test_missing_o3_so2_can_be_predicted_end_to_end(sample_config, monkeypatch):
    """Kiểm tra truyền dữ liệu thiếu hoàn toàn O3/SO2 qua pipeline dự báo vẫn trả kết quả hợp lệ."""
    class FakeModel:
        def predict(self, df):
            return np.array([25.4])

    monkeypatch.setattr("joblib.load", lambda path: FakeModel())
    predictor = Predictor(sample_config)
    predictor.metadata["serving_champion"] = "fake_model"

    records = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=25, freq="h"),
            "station": ["A"] * 25,
            "PM2.5": [15.0] * 25,
            "O3": [None] * 25,
            "SO2": [None] * 25,
        }
    )
    res = predictor.predict(records)
    assert res["predicted_pm25"] == 25.4
    assert np.isfinite(res["predicted_pm25"])
    assert res["level"] in ["Thấp", "Trung bình", "Cao"]
