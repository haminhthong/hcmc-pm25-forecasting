from __future__ import annotations

"""Khởi tạo các mô hình ứng viên từ tên cấu hình."""

from sklearn.base import RegressorMixin
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)


def build_model(
    name: str,
    random_state: int,
    params: dict | None = None,
) -> RegressorMixin:
    """Tạo model từ tên; ném lỗi rõ ràng khi tên không được hỗ trợ."""
    params = params or {}
    if name == "random_forest":
        defaults = {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 2, "n_jobs": -1}
        return RandomForestRegressor(random_state=random_state, **(defaults | params))
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=random_state)
    raise ValueError(f"Mô hình không được hỗ trợ: {name}")
