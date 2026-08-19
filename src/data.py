from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file)


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
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"CSV thiếu cột bắt buộc: {', '.join(missing)}")


def load_air_quality(config: dict[str, Any]) -> pd.DataFrame:
    data_config = config["data"]
    frame = pd.read_csv(resolve_data_path(data_config["path"]))
    validate_schema(frame, data_config["required_columns"])
    timestamp = data_config["timestamp_column"]
    station = data_config["station_column"]
    target = data_config["target_column"]
    for column in data_config.get("optional_columns", []):
        if column not in frame:
            frame[column] = pd.NA
    frame[timestamp] = pd.to_datetime(frame[timestamp], errors="coerce")
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    if frame[timestamp].isna().any():
        raise ValueError("Cột thời gian chứa giá trị không hợp lệ.")
    if data_config.get("zero_as_missing", False):
        for column in ("TSP", target):
            if column in frame:
                frame.loc[frame[column] <= 0, column] = pd.NA
    return frame.sort_values([station, timestamp], kind="stable").reset_index(drop=True)
