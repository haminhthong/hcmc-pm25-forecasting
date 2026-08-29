"""Huấn luyện, backtest, chọn champion và lưu artifact tái lập."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data import audit_air_quality, load_air_quality, load_config, resolve_data_path
from src.evaluate import (
    metrics_by_station,
    persistence_predictions,
    regression_and_classification_metrics,
    seasonal_naive_predictions,
)
from src.features import build_features, model_feature_columns
from src.models import build_model
from src.utils import sha256_file


def make_pipeline(config: dict, model_name: str) -> tuple[Pipeline, list[str]]:
    """Tạo preprocessing và model trong cùng một sklearn Pipeline."""
    station = config["data"]["station_column"]
    numeric = model_feature_columns(config)
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median", keep_empty_features=True),
                numeric,
            ),
            (
                "station",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                [station],
            ),
        ]
    )
    params = (
        config["model"].get("params", {})
        if model_name == config["model"]["name"]
        else {}
    )
    model = build_model(model_name, config["project"]["random_state"], params)
    return Pipeline([("preprocess", preprocessor), ("model", model)]), [*numeric, station]


def expanding_folds(
    size: int,
    folds: int,
    minimum_train_rows: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Sinh expanding-window folds; validation luôn nằm sau train."""
    validation_size = max(1, (size - minimum_train_rows) // folds)
    for fold in range(folds):
        train_end = minimum_train_rows + fold * validation_size
        validation_end = size if fold == folds - 1 else min(size, train_end + validation_size)
        if train_end < validation_end:
            yield np.arange(train_end), np.arange(train_end, validation_end)


def split_by_time(
    frame: pd.DataFrame,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chia train/test theo vị trí thời gian, tuyệt đối không shuffle."""
    cut = max(1, int(len(frame) * (1 - test_fraction)))
    train_frame = frame.iloc[:cut].copy()
    test_frame = frame.iloc[cut:].copy()
    if test_frame.empty:
        raise ValueError("Dữ liệu quá ít để tạo tập kiểm thử cuối theo thời gian.")
    return train_frame, test_frame


def evaluate_candidate(
    name: str,
    train_frame: pd.DataFrame,
    config: dict,
    columns: list[str],
) -> dict:
    """Đánh giá một model trên toàn bộ expanding-window folds."""
    fold_metrics = []
    split = config["split"]
    for train_indices, validation_indices in expanding_folds(
        len(train_frame), split["backtest_folds"], split["minimum_train_rows"]
    ):
        pipeline, _ = make_pipeline(config, name)
        train_fold = train_frame.iloc[train_indices]
        validation_fold = train_frame.iloc[validation_indices]
        pipeline.fit(train_fold[columns], train_fold["target_next_hour"])
        prediction = pipeline.predict(validation_fold[columns])
        fold_metrics.append(
            regression_and_classification_metrics(
                validation_fold["target_next_hour"],
                prediction,
                config["thresholds"],
            )
        )
    if not fold_metrics:
        raise ValueError("Dữ liệu quá ít cho rolling backtest với cấu hình hiện tại.")
    return {
        "folds": fold_metrics,
        "mae_mean": float(np.mean([item["mae"] for item in fold_metrics])),
        "mae_std": float(np.std([item["mae"] for item in fold_metrics])),
        "rmse_mean": float(np.mean([item["rmse"] for item in fold_metrics])),
    }


def evaluate_baselines(
    test_frame: pd.DataFrame,
    target: str,
    thresholds: dict[str, float],
) -> dict:
    """Đánh giá hai baseline bắt buộc trên cùng tập test cuối."""
    target_next_hour = test_frame["target_next_hour"]
    return {
        "persistence": regression_and_classification_metrics(
            target_next_hour,
            persistence_predictions(test_frame, target),
            thresholds,
        ),
        "seasonal_naive_24h": regression_and_classification_metrics(
            target_next_hour,
            seasonal_naive_predictions(test_frame, target),
            thresholds,
        ),
    }


def build_quality_gate(champion_mae: float, persistence_mae: float) -> dict:
    """Xác định model có thắng baseline persistence hay không."""
    improvement = (
        (persistence_mae - champion_mae) / persistence_mae
        if persistence_mae
        else None
    )
    passes_baseline = champion_mae < persistence_mae
    return {
        "passes_baseline": passes_baseline,
        "mae_improvement_vs_persistence": improvement,
        "status": "đạt" if passes_baseline else "không đạt",
    }


def save_artifacts(
    pipeline: Pipeline,
    metadata: dict,
    evaluation: dict,
    config: dict,
) -> tuple[Path, Path]:
    """Lưu model, metadata và báo cáo đánh giá bằng định dạng tái lập."""
    artifact_dir = Path(config["artifacts"]["directory"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / config["artifacts"]["model_file"]
    evaluation_path = artifact_dir / config["artifacts"]["evaluation_file"]
    metadata_path = artifact_dir / config["artifacts"]["metadata_file"]

    joblib.dump(pipeline, model_path)
    evaluation_path.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return model_path, evaluation_path


def train(config_path: str, persist_artifacts: bool = True) -> dict:
    """Chạy pipeline huấn luyện đầy đủ và tùy chọn lưu artifact."""
    config = load_config(config_path)
    raw = load_air_quality(config)
    timestamp = config["data"]["timestamp_column"]
    station = config["data"]["station_column"]
    target = config["data"]["target_column"]
    frame = (
        build_features(raw, config)
        .dropna(subset=["target_next_hour"])
        .sort_values(timestamp, kind="stable")
        .reset_index(drop=True)
    )
    train_frame, test_frame = split_by_time(
        frame,
        config["split"]["test_fraction"],
    )

    _, columns = make_pipeline(config, config["model"]["name"])
    candidates = config.get("model_comparison", {}).get(
        "candidates", [config["model"]["name"]]
    )
    backtest = {
        name: evaluate_candidate(name, train_frame, config, columns)
        for name in candidates
    }
    champion = min(backtest, key=lambda name: backtest[name]["mae_mean"])
    pipeline, columns = make_pipeline(config, champion)
    pipeline.fit(train_frame[columns], train_frame["target_next_hour"])
    prediction = pipeline.predict(test_frame[columns])
    baselines = evaluate_baselines(test_frame, target, config["thresholds"])
    metrics = regression_and_classification_metrics(
        test_frame["target_next_hour"],
        prediction,
        config["thresholds"],
    )
    quality_gate = build_quality_gate(
        metrics["mae"],
        baselines["persistence"]["mae"],
    )
    artifact_dir = Path(config["artifacts"]["directory"])
    model_path = artifact_dir / config["artifacts"]["model_file"]
    evaluation = {
        "data_audit": audit_air_quality(raw, config),
        "baselines": baselines,
        "backtest": backtest,
        "champion_test": metrics,
        "quality_gate": quality_gate,
        "metrics_by_station": metrics_by_station(
            test_frame,
            prediction,
            station,
            config["thresholds"],
        ),
    }
    evaluation_path = artifact_dir / config["artifacts"]["evaluation_file"]
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "model_name": champion,
        "selection_rule": "MAE trung bình thấp nhất trên expanding-window backtest của tập train",
        "random_state": config["project"]["random_state"],
        "data_sha256": sha256_file(resolve_data_path(config["data"]["path"])),
        "features": columns,
        "train_period": [str(train_frame[timestamp].min()), str(train_frame[timestamp].max())],
        "test_period": [str(test_frame[timestamp].min()), str(test_frame[timestamp].max())],
        "metrics": metrics,
        "quality_gate": quality_gate,
        "thresholds": config["thresholds"],
        "model_path": str(model_path),
        "evaluation_path": str(evaluation_path),
    }
    if persist_artifacts:
        save_artifacts(pipeline, metadata, evaluation, config)
        if config.get("mlflow", {}).get("enabled"):
            log_mlflow(config, metadata, evaluation, model_path)
    else:
        metadata["evaluation"] = evaluation
    return metadata


def log_mlflow(config: dict, metadata: dict, evaluation: dict, model_path: Path) -> None:
    """Ghi tham số, metric và artifact vào MLflow khi được bật."""
    import mlflow

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=metadata["model_name"]):
        mlflow.log_params({"model_name": metadata["model_name"], "seed": config["project"]["random_state"]})
        scalar_metrics = {
            key: value
            for key, value in metadata["metrics"].items()
            if isinstance(value, (int, float)) and value is not None
        }
        mlflow.log_metrics(scalar_metrics)
        mlflow.log_dict(metadata["features"], "features.json")
        mlflow.log_dict(evaluation, "evaluation.json")
        mlflow.log_artifact(str(model_path))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Huấn luyện và chọn mô hình dự báo PM2.5 giờ kế tiếp")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Chạy toàn bộ đánh giá nhưng không ghi artifact")
    args = parser.parse_args()
    result = train(args.config, persist_artifacts=not args.dry_run)
    if args.dry_run:
        evaluation = result.pop("evaluation")
        result["dry_run"] = True
        result["backtest_summary"] = {
            name: {key: value for key, value in report.items() if key != "folds"}
            for name, report in evaluation["backtest"].items()
        }
        result["baseline_test_mae"] = {
            name: report["mae"] for name, report in evaluation["baselines"].items()
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
