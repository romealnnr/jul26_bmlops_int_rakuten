"""Point 18: each prediction carries its category name, the flag and the top 3."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import main as api_main
from src.models import predict as predict_module
from src.models.categories import PRDTYPECODE_LABELS, category_name


class _FakePipeline:
    """Stands in for the MLflow model: four classes, fixed probabilities."""

    classes_ = np.array([10, 40, 50, 60])

    def predict_proba(self, X):
        rows = [[0.05, 0.70, 0.15, 0.10], [0.30, 0.25, 0.20, 0.25]]
        return np.array(rows[: len(X)])


@pytest.fixture
def fake_model(monkeypatch):
    artifact = {
        "pipeline": _FakePipeline(),
        "version": "7",
        "feature_names": ["designation", "description", "productid", "imageid", "image_path"],
    }
    monkeypatch.setattr(predict_module, "load_artifact", lambda force_reload=False: artifact)


def test_category_names_cover_the_27_codes():
    assert len(PRDTYPECODE_LABELS) == 27
    assert category_name(2905) == "PC games"
    assert category_name(999) == "Unknown category 999"


def test_predict_returns_name_flag_and_top3(fake_model):
    records = [{"designation": "Manette"}, {"designation": "Objet"}]
    results = predict_module.predict(records, log_to_db=False)

    confident, unsure = results
    assert confident["prediction"] == 40
    assert confident["category_name"] == "Video games / imported CDs"
    assert confident["probability"] == pytest.approx(0.70)
    assert confident["flagged"] is False and confident["flag_reason"] is None
    assert [c["prediction"] for c in confident["top3"]] == [40, 50, 60]
    assert confident["top3"][0]["prediction"] == confident["prediction"]

    assert unsure["prediction"] == 10
    assert unsure["flagged"] is True
    assert unsure["flag_reason"] == "low_confidence"
    assert len(unsure["top3"]) == 3


def test_api_response_keeps_name_flag_and_top3(fake_model, monkeypatch):
    real_predict = predict_module.predict
    monkeypatch.setattr(
        predict_module,
        "predict",
        lambda records: real_predict(records, log_to_db=False),
    )
    client = TestClient(api_main.app)  # no context manager: startup (DB, MLflow) is skipped
    response = client.post("/predict", json={"records": [{"designation": "Manette"}]})

    assert response.status_code == 200
    item = response.json()["results"][0]
    assert item["category_name"] == "Video games / imported CDs"
    assert item["flagged"] is False
    assert item["flag_reason"] is None
    assert [c["category_name"] for c in item["top3"]] == [
        "Video games / imported CDs",
        "Console accessories",
        "Game consoles",
    ]
