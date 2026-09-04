"""Đọc, kiểm tra và lập báo cáo chất lượng dữ liệu quan trắc."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Đọc cấu hình YAML từ đường dẫn được cung cấp."""
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Kiểm tra sớm các khóa và giá trị cấu hình quan trọng."""
    required_sections = {"project", "data", "features", "split", "model", "thresholds", "artifacts"}
    missing_sections = sorted(required_sections - set(config or {}))
    if missing_sections:
        raise ValueError(f"Cấu hình thiếu section: {', '.join(missing_sections)}")

    test_fraction = config["split"].get("test_fraction")
    if not isinstance(test_fraction, int | float) or not 0 < test_fraction < 1:
        raise ValueError("split.test_fraction phải nằm trong khoảng (0, 1).")

    folds = config["split"].get("backtest_folds")
    minimum_periods = config["split"].get("minimum_train_periods")
    if not isinstance(folds, int) or folds < 2:
        raise ValueError("split.backtest_folds phải là số nguyên từ 2 trở lên.")
    if not isinstance(minimum_periods, int) or minimum_periods < 1:
        raise ValueError("split.minimum_train_periods phải là số nguyên dương.")

    lags = config["features"].get("lags", [])
    windows = config["features"].get("rolling_windows", [])
    if not lags or any(not isinstance(value, int) or value < 1 for value in lags):
        raise ValueError("features.lags phải chứa các số nguyên dương.")
    if not windows or any(not isinstance(value, int) or value < 1 for value in windows):
        raise ValueError("features.rolling_windows phải chứa các số nguyên dương.")

    low_max = config["thresholds"].get("low_max", config["thresholds"].get("good_max"))
    medium_max = config["thresholds"].get("medium_max", config["thresholds"].get("moderate_max"))
    if low_max is None or medium_max is None or low_max >= medium_max:
        raise ValueError("thresholds.low_max phải nhỏ hơn thresholds.medium_max.")



def resolve_data_path(configured_path: str | Path) -> Path:
    """Tìm CSV trong đường dẫn cấu hình, thư mục dự án hoặc /content (Colab)."""
    candidate = Path(configured_path).expanduser()
    project_root = Path(__file__).resolve().parents[1]
    candidates = [candidate, project_root / candidate, Path("/content") / candidate.name]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    searched = "\n- ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Không tìm thấy dữ liệu. Đã kiểm tra:\n- {searched}\n"
        "Hãy sửa data.path trong config hoặc tải CSV lên /content khi dùng Colab."
    )


def validate_schema(frame: pd.DataFrame, required_columns: list[str]) -> None:
    """Kiểm tra sự hiện diện của các cột bắt buộc trong bảng dữ liệu."""
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"CSV thiếu cột bắt buộc: {', '.join(missing)}")


def audit_air_quality(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Tạo báo cáo chất lượng dữ liệu không làm thay đổi dữ liệu đầu vào."""
    data_config = config["data"]
    timestamp = data_config["timestamp_column"]
    station = data_config["station_column"]
    target = data_config["target_column"]
    duplicate_mask = frame.duplicated([station, timestamp], keep=False)
    ordered = frame.sort_values([station, timestamp], kind="stable")
    gaps = ordered.groupby(station)[timestamp].diff().dropna()
    return {
        "rows": int(len(frame)),
        "stations": int(frame[station].nunique()),
        "period": [str(frame[timestamp].min()), str(frame[timestamp].max())],
        "duplicate_station_timestamps": int(duplicate_mask.sum()),
        "missing_by_column": {column: int(value) for column, value in frame.isna().sum().items()},
        "non_positive_target": int((frame[target] <= 0).sum()),
        "irregular_hourly_gaps": int((gaps != pd.to_timedelta(1, unit="h")).sum()),
    }


def load_air_quality(config: dict[str, Any]) -> pd.DataFrame:
    """Đọc và chuẩn hóa dữ liệu chất lượng không khí theo cấu hình."""
    data_config = config["data"]
    frame = pd.read_csv(resolve_data_path(data_config["path"]))
    validate_schema(frame, data_config["required_columns"])
    timestamp = data_config["timestamp_column"]
    station = data_config["station_column"]
    target = data_config["target_column"]

    # Tạo các cột tùy chọn còn thiếu để pipeline train và inference đồng nhất.
    for column in data_config.get("optional_columns", []):
        if column not in frame:
            frame[column] = pd.NA
    frame[timestamp] = pd.to_datetime(frame[timestamp], errors="coerce")
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    if frame[timestamp].isna().any():
        raise ValueError("Cột thời gian chứa giá trị không hợp lệ.")
    if frame[station].isna().any():
        raise ValueError("Cột trạm chứa giá trị thiếu.")
    if frame.duplicated([station, timestamp]).any():
        raise ValueError("Dữ liệu có station/timestamp trùng lặp.")
    if data_config.get("zero_as_missing", False):
        # Chỉ áp dụng quy tắc sentinel khi người dùng bật rõ ràng trong cấu hình.
        for column in ("TSP", target):
            if column in frame:
                frame.loc[frame[column] <= 0, column] = pd.NA
    return frame.sort_values([station, timestamp], kind="stable").reset_index(drop=True)
