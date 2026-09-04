"""Nạp artifact và dự báo PM2.5 cho quan trắc mới nhất."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import json

from src.evaluate import classify_pm25, get_threshold_params
from src.features import build_features, model_feature_columns


class Predictor:
    """Đóng gói preprocessing, model và metadata cho inference."""

    def __init__(self, config: dict):
        self.config = config
        artifact_dir = Path(config["artifacts"]["directory"])
        self.model = joblib.load(artifact_dir / config["artifacts"]["model_file"])
        metadata_path = artifact_dir / config["artifacts"].get("metadata_file", "metadata.json")
        if metadata_path.exists():
            with metadata_path.open(encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def predict(self, observations: pd.DataFrame) -> dict:
        """Dự báo giờ kế tiếp từ chuỗi quan trắc của một trạm."""
        station_column = self.config["data"]["station_column"]
        timestamp_column = self.config["data"]["timestamp_column"]
        target_column = self.config["data"]["target_column"]

        observations = observations.copy()
        if observations.empty:
            raise ValueError("Cần ít nhất một quan trắc để dự báo.")
        if observations[station_column].nunique(dropna=False) != 1:
            raise ValueError("Mỗi request chỉ được chứa dữ liệu của một trạm duy nhất.")

        lags = self.config["features"].get("lags", [24])
        windows = self.config["features"].get("rolling_windows", [24])
        required_history_hours = max(max(lags), max(windows))

        if len(observations) < required_history_hours + 1:
            raise ValueError(
                f"Cần tối thiểu {required_history_hours + 1} quan trắc để dự báo."
            )

        observations[timestamp_column] = pd.to_datetime(
            observations[timestamp_column],
            errors="coerce",
        )
        if observations[timestamp_column].isna().any():
            raise ValueError("Request chứa timestamp không hợp lệ.")
        if observations.duplicated([station_column, timestamp_column]).any():
            raise ValueError("Request chứa timestamp trùng lặp trong cùng trạm.")

        ordered = observations.sort_values(timestamp_column)
        gaps = ordered[timestamp_column].diff().dropna()
        if (gaps != pd.to_timedelta(1, unit="h")).any():
            raise ValueError("Chuỗi quan trắc phải liên tục theo từng giờ.")

        if (ordered[target_column] < 0).any():
            raise ValueError("Nồng độ PM2.5 không được âm.")

        featured = build_features(ordered, self.config, include_target=False)
        latest = featured.sort_values(timestamp_column).iloc[[-1]]
        columns = [*model_feature_columns(self.config), station_column]
        model_input = latest.reindex(columns=columns)

        serving_champion = self.metadata.get("serving_champion")
        if serving_champion == "persistence":
            value = max(0.0, float(latest[target_column].iloc[0]))
        else:
            value = max(0.0, float(self.model.predict(model_input)[0]))

        thresholds = self.config["thresholds"]
        low_max, medium_max, labels = get_threshold_params(thresholds)

        interval_info = self.metadata.get("prediction_interval", {})
        q90 = float(interval_info.get("residual_quantile", 5.0))
        lower = max(0.0, float(value - q90))
        upper = float(value + q90)

        origin_dt = latest[timestamp_column].iloc[0]
        target_dt = origin_dt + pd.to_timedelta(1, unit="h")

        return {
            "station": str(latest[station_column].iloc[0]),
            "forecast_origin": origin_dt.isoformat(),
            "forecast_for": target_dt.isoformat(),
            "current_pm25": float(latest[target_column].iloc[0]),
            "predicted_pm25": round(value, 3),
            "level": str(classify_pm25([value], low_max, medium_max, labels)[0]),
            "interval": {
                "lower": round(lower, 2),
                "upper": round(upper, 2),
                "coverage": 0.9,
            },
            "model_version": self.metadata.get("model_version", "2026-09-01"),
            "updated_at": datetime.now(UTC).isoformat(),
        }
