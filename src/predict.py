from __future__ import annotations

"""Nạp artifact và dự báo PM2.5 cho quan trắc mới nhất."""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluate import classify_pm25
from src.features import build_features, model_feature_columns


class Predictor:
    """Đóng gói preprocessing, model và metadata cho inference."""

    def __init__(self, config: dict):
        self.config = config
        artifact_dir = Path(config["artifacts"]["directory"])
        self.model = joblib.load(artifact_dir / config["artifacts"]["model_file"])
        metadata_path = artifact_dir / config["artifacts"]["metadata_file"]
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    def predict(self, observations: pd.DataFrame) -> dict:
        """Dự báo giờ kế tiếp từ chuỗi quan trắc của một trạm."""
        station_column = self.config["data"]["station_column"]
        timestamp_column = self.config["data"]["timestamp_column"]
        observations = observations.copy()
        observations[timestamp_column] = pd.to_datetime(observations[timestamp_column])
        featured = build_features(observations, self.config, include_target=False)
        latest = featured.sort_values(timestamp_column).iloc[[-1]]
        columns = [*model_feature_columns(self.config), station_column]
        model_input = latest.reindex(columns=columns)
        value = max(0.0, float(self.model.predict(model_input)[0]))
        thresholds = self.config["thresholds"]
        tree_predictions = []
        try:
            transformed = self.model.named_steps["preprocess"].transform(model_input)
            estimators = self.model.named_steps["model"].estimators_
            tree_predictions = [tree.predict(transformed)[0] for tree in estimators]
        except (AttributeError, KeyError):
            pass
        uncertainty = float(np.std(tree_predictions)) if tree_predictions else None
        confidence = (
            None
            if uncertainty is None
            else float(np.clip(1 - uncertainty / max(value, 1), 0, 1))
        )
        return {
            "station": str(latest[station_column].iloc[0]),
            "current_pm25": float(latest[self.config["data"]["target_column"]].iloc[0]),
            "predicted_pm25": round(value, 3),
            "level": str(classify_pm25([value], thresholds["good_max"], thresholds["moderate_max"])[0]),
            "confidence": None if confidence is None else round(confidence, 3),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
