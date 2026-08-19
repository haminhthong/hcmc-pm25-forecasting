import pandas as pd
from fastapi.testclient import TestClient

from src.evaluate import VALID_LABELS, classify_pm25


def test_output_labels_are_valid():
    assert set(classify_pm25([5, 20, 50], 12, 35.5)) == set(VALID_LABELS)


def test_missing_o3_so2_can_be_imputed():
    from sklearn.impute import SimpleImputer
    values = pd.DataFrame({"O3": [None, 4.0], "SO2": [2.0, None]})
    assert SimpleImputer(strategy="median").fit_transform(values).shape == (2, 2)


def test_api_json_structure(monkeypatch):
    from app import api

    expected = {
        "station": "A", "current_pm25": 10.0, "predicted_pm25": 11.0,
        "level": "Tốt", "confidence": 0.9, "updated_at": "2024-01-01T00:00:00+00:00",
    }

    class FakePredictor:
        def predict(self, observations):
            return expected

    monkeypatch.setattr(api, "get_predictor", lambda: FakePredictor())
    client = TestClient(api.app)
    response = client.post("/predict", json={"observations": [
        {"timestamp": "2024-01-01T00:00:00", "station": "A", "PM2.5": 9},
        {"timestamp": "2024-01-01T01:00:00", "station": "A", "PM2.5": 10},
    ]})
    assert response.status_code == 200
    assert response.json() == expected
