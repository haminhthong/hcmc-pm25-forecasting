"""FastAPI phục vụ dự báo PM2.5 giờ kế tiếp."""

from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data import load_config
from src.predict import Predictor


class Observation(BaseModel):
    """Một quan trắc đầu vào của trạm."""
    timestamp: str
    station: str
    PM25: float = Field(alias="PM2.5")
    TSP: float | None = None
    NO2: float | None = None
    SO2: float | None = None
    CO: float | None = None
    O3: float | None = None
    temperature: float | None = None
    humidity: float | None = None

    model_config = {"populate_by_name": True}


class PredictionRequest(BaseModel):
    """Chuỗi quan trắc dùng để tạo lag và dự báo."""
    observations: list[Observation] = Field(min_length=2)


class PredictionResponse(BaseModel):
    """Cấu trúc phản hồi ổn định của endpoint dự báo."""
    station: str
    current_pm25: float
    predicted_pm25: float
    level: str
    confidence: float | None
    updated_at: str


@lru_cache
def get_predictor() -> Predictor:
    """Nạp predictor một lần và tái sử dụng giữa các request."""
    return Predictor(load_config("configs/config.yaml"))


app = FastAPI(title="API dự báo PM2.5 TP.HCM", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Kiểm tra tiến trình API đang hoạt động."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> dict:
    """Kiểm tra đầu vào và trả dự báo PM2.5 giờ kế tiếp."""
    try:
        records = [item.model_dump(by_alias=True) for item in request.observations]
        return get_predictor().predict(pd.DataFrame(records))
    except (ValueError, KeyError, FileNotFoundError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
