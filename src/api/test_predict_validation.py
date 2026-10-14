"""/predict rejects invalid records (422) before any model or database is touched."""

from __future__ import annotations

import os

for _key, _value in {
    "POSTGRES_USER": "t",
    "POSTGRES_PASSWORD": "t",
    "POSTGRES_DB": "t",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "MODEL_NAME": "baseline_logreg",
    "TARGET_COLUMN": "prdtypecode",
    "TEST_SIZE": "0.2",
    "RANDOM_STATE": "42",
    "API_HOST": "0.0.0.0",
    "API_PORT": "8000",
    "LOG_LEVEL": "INFO",
    "MLFLOW_EXPERIMENT_NAME": "rakuten-test",
    "MLFLOW_REGISTERED_MODEL_NAME": "rakuten_baseline",
}.items():
    os.environ.setdefault(_key, _value)

import pydantic  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api import main as api_main  # noqa: E402
from src.api.schemas import MAX_RECORDS_PER_REQUEST  # noqa: E402
from src.models import predict as predict_module  # noqa: E402

GOOD = {"designation": "Console de jeu portable retro", "description": "400 jeux"}


@pytest.fixture
def client(monkeypatch):
    """No lifespan (no `with`), and predict() replaced: only validation is under test."""
    calls = []

    def fake_predict(records, **_kwargs):
        calls.append(records)
        result = {
            "prediction": 10, "category_name": "Used books", "top3": [],
            "probability": 0.9, "model_version": "1",
            "flagged": False, "flag_reason": None,
            "image_used": False, "warnings": [],
        }
        return [dict(result) for _ in records]

    monkeypatch.setattr(predict_module, "predict", fake_predict)
    test_client = TestClient(api_main.app)
    test_client.calls = calls
    return test_client


def test_valid_record_is_predicted(client):
    response = client.post("/predict", json={"records": [GOOD]})
    assert response.status_code == 200, response.text
    assert response.json()["count"] == 1
    assert client.calls[0][0]["designation"] == GOOD["designation"]


@pytest.mark.parametrize(
    "records",
    [
        [{"description": "no title"}],  # designation missing
        [{"designation": "   "}],  # designation blank
        [{**GOOD, "colour": "red"}],  # unknown field
        [{**GOOD, "productid": "abc"}],  # wrong type
        [],  # empty batch
        [GOOD] * (MAX_RECORDS_PER_REQUEST + 1),  # batch too large
    ],
)
def test_invalid_records_get_422_and_never_reach_the_model(client, records):
    response = client.post("/predict", json={"records": records})
    assert response.status_code == 422, response.text
    assert client.calls == []


def test_predict_function_validates_cli_and_dag_callers_too():
    # Validation runs before the model is loaded, so no MLflow is needed here.
    with pytest.raises(pydantic.ValidationError):
        predict_module.predict([{"description": "no title"}], log_to_db=False, log_to_mlflow=False)
