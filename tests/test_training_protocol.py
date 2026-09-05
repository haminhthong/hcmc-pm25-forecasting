import numpy as np
import pandas as pd

from src.train import build_quality_gate, expanding_time_folds, split_by_time


def test_expanding_folds_never_train_on_future():
    frame = pd.DataFrame(
        {
            "timestamp": np.repeat(pd.date_range("2024-01-01", periods=50, freq="h"), 2),
            "station": ["A", "B"] * 50,
        }
    )
    folds = expanding_time_folds(frame, "timestamp", folds=4, minimum_train_periods=40)
    assert len(folds) == 4
    for train_indices, validation_indices in folds:
        train_periods = frame.iloc[train_indices]["timestamp"]
        validation_periods = frame.iloc[validation_indices]["timestamp"]
        assert train_periods.max() < validation_periods.min()
        assert set(train_periods).isdisjoint(set(validation_periods))
        assert len(set(train_indices) & set(validation_indices)) == 0


def test_time_split_preserves_order():
    frame = pd.DataFrame(
        {
            "timestamp": np.repeat(pd.date_range("2024-01-01", periods=10, freq="h"), 2),
            "station": ["A", "B"] * 10,
        }
    )
    train_frame, cal_frame, test_frame = split_by_time(
        frame, test_fraction=0.2, calibration_fraction=0.2
    )
    assert train_frame["timestamp"].max() < cal_frame["timestamp"].min()
    assert cal_frame["timestamp"].max() < test_frame["timestamp"].min()
    assert len(test_frame) == 4
    assert test_frame.groupby("timestamp")["station"].nunique().eq(2).all()


def test_quality_gate_requires_model_to_beat_persistence():
    cfg = {
        "quality_gate": {
            "minimum_mae_improvement": 0.05,
            "minimum_high_pm25_recall": 0.75,
            "maximum_rolling_mae_std": 1.0,
        }
    }
    passed = build_quality_gate({"mae": 1.0, "high_pm25_recall": 0.8}, {"mae": 2.0}, 0.5, cfg)
    failed = build_quality_gate({"mae": 1.98, "high_pm25_recall": 0.6}, {"mae": 2.0}, 0.5, cfg)
    assert passed["passes_baseline"] is True
    assert failed["passes_baseline"] is False
