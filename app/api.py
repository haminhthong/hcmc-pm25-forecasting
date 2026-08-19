from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data import load_config
from src.predict import Predictor


class Observation(BaseModel):
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
    observations: list[Observation] = Field(min_length=2)


@lru_cache
def get_predictor() -> Predictor:
    return Predictor(load_config("configs/config.yaml"))


app = FastAPI(title="API dự báo PM2.5 TP.HCM", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        records = [item.model_dump(by_alias=True) for item in request.observations]
        return get_predictor().predict(pd.DataFrame(records))
    except (ValueError, KeyError, FileNotFoundError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

