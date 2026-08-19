from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data import load_air_quality, load_config
from src.evaluate import regression_and_classification_metrics
from src.features import build_features, model_feature_columns


def train(config_path: str) -> dict:
    config = load_config(config_path)
    frame = build_features(load_air_quality(config), config)
    station = config["data"]["station_column"]
    timestamp = config["data"]["timestamp_column"]
    numeric_features = model_feature_columns(config)
    usable = frame.dropna(subset=["target_next_hour"]).sort_values(timestamp, kind="stable").copy()
    cut = max(1, int(len(usable) * (1 - config["split"]["test_fraction"])))
    train_frame, test_frame = usable.iloc[:cut], usable.iloc[cut:]
    if test_frame.empty:
        raise ValueError("Dữ liệu quá ít để tạo tập kiểm thử theo thời gian.")
    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric_features),
        ("station", OneHotEncoder(handle_unknown="ignore"), [station]),
    ])
    model = RandomForestRegressor(random_state=config["project"]["random_state"], **config["model"]["params"])
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    columns = [*numeric_features, station]
    pipeline.fit(train_frame[columns], train_frame["target_next_hour"])
    prediction = pipeline.predict(test_frame[columns])
    metrics = regression_and_classification_metrics(test_frame["target_next_hour"], prediction, config["thresholds"])
    artifact_dir = Path(config["artifacts"]["directory"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / config["artifacts"]["model_file"]
    joblib.dump(pipeline, model_path)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": config["model"]["name"],
        "hyperparameters": config["model"]["params"],
        "features": columns,
        "train_period": [str(train_frame[timestamp].min()), str(train_frame[timestamp].max())],
        "test_period": [str(test_frame[timestamp].min()), str(test_frame[timestamp].max())],
        "metrics": metrics,
        "thresholds": config["thresholds"],
        "model_path": str(model_path),
    }
    metadata_path = artifact_dir / config["artifacts"]["metadata_file"]
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if config.get("mlflow", {}).get("enabled"):
        log_mlflow(config, metadata, model_path)
    return metadata


def log_mlflow(config: dict, metadata: dict, model_path: Path) -> None:
    import mlflow

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=metadata["model_name"]):
        mlflow.log_params(metadata["hyperparameters"])
        mlflow.log_metrics({key: value for key, value in metadata["metrics"].items() if key != "confusion_matrix"})
        mlflow.log_dict(metadata["features"], "features.json")
        mlflow.log_dict(metadata["metrics"]["confusion_matrix"], "confusion_matrix.json")
        mlflow.log_artifact(str(model_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình dự báo PM2.5 giờ kế tiếp")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(json.dumps(train(args.config), ensure_ascii=False, indent=2))
