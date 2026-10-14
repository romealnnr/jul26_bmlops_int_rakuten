"""/health on a fresh deployment, before the first model is registered.

No PostgreSQL and no MLflow server needed: the registry is a throwaway SQLite
MLflow store, so the "model does not exist" error is MLflow's real one.
"""

from __future__ import annotations

import os

# Settings are read at import time; these defaults only apply when the
# environment (or .env) does not already provide the values.
for _key, _value in {
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_DB": "test",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "MODEL_NAME": "baseline",
    "TARGET_COLUMN": "prdtypecode",
    "TEST_SIZE": "0.2",
    "RANDOM_STATE": "42",
    "API_HOST": "0.0.0.0",
    "API_PORT": "8000",
    "LOG_LEVEL": "INFO",
    "MLFLOW_EXPERIMENT_NAME": "test",
    "MLFLOW_REGISTERED_MODEL_NAME": "health_test_model",
}.items():
    os.environ.setdefault(_key, _value)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src import config  # noqa: E402
from src.api import main  # noqa: E402
from src.models import predict as predict_module  # noqa: E402


@pytest.fixture
def empty_registry(tmp_path, monkeypatch):
    """An MLflow registry with no registered model, and no database."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(config.Settings, "mlflow_tracking_uri", property(lambda self: uri))
    monkeypatch.setattr(main, "ping", lambda: False)
    monkeypatch.setattr(predict_module, "_ARTIFACT", None)
    return uri


def test_load_artifact_without_registered_model_raises_model_not_found(empty_registry):
    with pytest.raises(predict_module.ModelNotFoundError):
        predict_module.load_artifact(force_reload=True)


def test_health_is_200_before_the_first_model(empty_registry):
    response = TestClient(main.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "down", "model_available": False}


def test_predict_before_the_first_model_is_409_not_500(empty_registry):
    response = TestClient(main.app).post("/predict", json={"records": [{"designation": "x"}]})
    assert response.status_code == 409


def test_health_is_200_when_the_registry_is_unreachable(monkeypatch):
    def unreachable():
        raise ConnectionError("registry down")

    monkeypatch.setattr(main, "ping", lambda: True)
    monkeypatch.setattr(predict_module, "model_info", unreachable)
    response = TestClient(main.app).get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_available"] is False
    assert body["model_error"] == "registry down"
