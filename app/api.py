from datetime import datetime
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.data import load_config
from src.predict import Predictor


class Observation(BaseModel):
    """Một quan trắc đầu vào của trạm với ràng buộc miền giá trị hợp lệ."""

    timestamp: datetime
    station: str = Field(min_length=1, max_length=100)
    PM25: float = Field(alias="PM2.5", ge=0, le=1000)
    TSP: float | None = Field(default=None, ge=0)
    NO2: float | None = Field(default=None, ge=0)
    SO2: float | None = Field(default=None, ge=0)
    CO: float | None = Field(default=None, ge=0)
    O3: float | None = Field(default=None, ge=0)
    temperature: float | None = Field(default=None, ge=-20, le=60)
    humidity: float | None = Field(default=None, ge=0, le=100)

    model_config = {"populate_by_name": True}


class PredictionRequest(BaseModel):
    """Chuỗi quan trắc dùng để tạo lag và dự báo (tối thiểu 25, tối đa 168 giờ)."""

    observations: list[Observation] = Field(min_length=25, max_length=168)


class Interval(BaseModel):
    """Khoảng dự báo Conformal Interval."""

    lower: float
    upper: float
    coverage: float = Field(default=0.9, ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Cấu trúc phản hồi chuẩn hóa của endpoint dự báo."""

    station: str
    forecast_origin: str
    forecast_for: str
    current_pm25: float
    predicted_pm25: float
    level: str
    interval: Interval
    model_version: str
    updated_at: str


class ErrorResponse(BaseModel):
    """Cấu trúc phản hồi lỗi chuẩn hóa."""

    code: str
    message: str


@lru_cache
def get_predictor() -> Predictor:
    """Nạp predictor một lần và tái sử dụng giữa các request."""
    return Predictor(load_config("configs/config.yaml"))


app = FastAPI(
    title="API dự báo PM2.5 TP.HCM",
    version="1.0.0",
    description="Hệ thống dự báo nồng độ PM2.5 giờ tiếp theo không rò rỉ dữ liệu.",
)


@app.get("/health")
def health():
    """Kiểm tra mô hình đã được nạp và sẵn sàng phục vụ."""
    try:
        predictor = get_predictor()
        if predictor.model is None:
            raise ValueError("Model is None")
        return {
            "status": "ready",
            "model_loaded": True,
        }
    except Exception as err:
        raise HTTPException(
            status_code=503,
            detail="Mô hình chưa sẵn sàng.",
        ) from err


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Kiểm tra đầu vào và trả dự báo PM2.5 giờ kế tiếp."""
    try:
        records = [item.model_dump(by_alias=True) for item in request.observations]
        # Chuyển timestamp datetime thành chuỗi ISO để pandas parser nhất quán
        for rec in records:
            if isinstance(rec.get("timestamp"), datetime):
                rec["timestamp"] = rec["timestamp"].isoformat()
        return get_predictor().predict(pd.DataFrame(records))
    except ValueError as error:
        return JSONResponse(
            status_code=400,
            content={"code": "INVALID_INPUT", "message": str(error)},
        )
    except FileNotFoundError:
        return JSONResponse(
            status_code=503,
            detail="Mô hình hoặc artifact chưa sẵn sàng.",
        )
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"code": "INTERNAL_ERROR", "message": "Lỗi nội bộ hệ thống."},
        )
