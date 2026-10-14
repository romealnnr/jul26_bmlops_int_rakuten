"""Predictions keep the true category (in /predict or later via /feedback).

The model and the database are stubbed, so these tests need neither Postgres
nor MLflow.
"""

from __future__ import annotations

import os

for _key, _value in {
    "POSTGRES_USER": "t",
    "POSTGRES_PASSWORD": "t",
    "POSTGRES_DB": "t",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "MODEL_NAME": "baseline_logreg",
    "TARGET_COLUMN": "target",
    "TEST_SIZE": "0.2",
    "RANDOM_STATE": "42",
    "API_HOST": "0.0.0.0",
    "API_PORT": "8000",
    "LOG_LEVEL": "INFO",
    "MLFLOW_EXPERIMENT_NAME": "rakuten-test",
    "MLFLOW_REGISTERED_MODEL_NAME": "rakuten_baseline",
}.items():
    os.environ.setdefault(_key, _value)

from contextlib import contextmanager  # noqa: E402

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.api import main as api_main  # noqa: E402
from src.models import predict as predict_module  # noqa: E402


def tiny_artifact() -> dict:
    pipeline = Pipeline(
        [
            ("text", ColumnTransformer([("designation", TfidfVectorizer(), "designation")])),
            ("classifier", LogisticRegression(max_iter=500)),
        ]
    )
    rows = [("livre roman poche", 10), ("piscine gonflable", 2583), ("carte pokemon", 1160)] * 4
    pipeline.fit(pd.DataFrame({"designation": [r[0] for r in rows]}), [r[1] for r in rows])
    return {
        "pipeline": pipeline,
        "model_name": "baseline_logreg",
        "version": "7",
        "trained_at": "2026-10-13T00:00:00+00:00",
        "feature_names": ["designation"],
        "metrics": {},
    }


@pytest.fixture
def stored(monkeypatch):
    added = []

    class Session:
        def add_all(self, objects):
            added.extend(objects)

        def flush(self):
            for number, row in enumerate(added, start=1):
                row.id = number

        def get(self, model, key):
            return next((row for row in added if row.id == key), None)

    @contextmanager
    def fake_get_session():
        yield Session()

    monkeypatch.setattr(predict_module, "get_session", fake_get_session)
    return added


@pytest.fixture
def client(monkeypatch):
    artifact = tiny_artifact()
    monkeypatch.setattr(predict_module, "load_artifact", lambda force_reload=False: artifact)
    # No MLflow server in tests: the per-batch MLflow summary is not what is tested here.
    monkeypatch.setattr(predict_module, "_log_batch_to_mlflow", lambda results: None)
    # Without the `with` block the lifespan (database init) does not run.
    return TestClient(api_main.app)


def test_predict_stores_true_label_outside_the_features(client, stored):
    response = client.post(
        "/predict", json={"records": [{"designation": "carte pokemon", "true_label": 1160}]}
    )
    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    assert result["true_label"] == 1160 and result["prediction_id"] == 1
    row = stored[0]
    assert row.true_label == 1160 and row.labeled_at is not None
    assert "true_label" not in row.features


def test_predict_without_true_label_stores_null(client, stored):
    response = client.post("/predict", json={"records": [{"designation": "carte pokemon"}]})
    assert response.status_code == 200, response.text
    assert response.json()["results"][0]["true_label"] is None
    assert stored[0].true_label is None and stored[0].labeled_at is None


@pytest.mark.parametrize("bad", [12345, "1160", True])
def test_predict_rejects_unknown_true_label(client, stored, bad):
    response = client.post(
        "/predict", json={"records": [{"designation": "carte pokemon", "true_label": bad}]}
    )
    assert response.status_code == 422
    assert stored == []


def test_feedback_round_trip(client, stored):
    predicted = client.post("/predict", json={"records": [{"designation": "carte pokemon"}]})
    prediction_id = predicted.json()["results"][0]["prediction_id"]
    response = client.post("/feedback", json={"prediction_id": prediction_id, "true_label": 1160})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["prediction_id"] == prediction_id and body["true_label"] == 1160
    assert body["correct"] == (body["prediction"] == 1160)
    assert stored[0].true_label == 1160 and stored[0].labeled_at is not None


@pytest.mark.parametrize(
    "payload",
    [
        {"prediction_id": 999, "true_label": 1160},
        {"prediction_id": 1, "true_label": 12345},
        {"prediction_id": 1},
        {"prediction_id": 1, "true_label": 1160, "extra": 1},
    ],
)
def test_feedback_rejects_unknown_ids_and_categories(client, stored, payload):
    client.post("/predict", json={"records": [{"designation": "carte pokemon"}]})
    assert client.post("/feedback", json=payload).status_code == 422


def test_feedback_without_model_returns_409(client, stored, monkeypatch):
    def no_model(force_reload=False):
        raise predict_module.ModelNotFoundError("no model")

    monkeypatch.setattr(predict_module, "load_artifact", no_model)
    response = client.post("/feedback", json={"prediction_id": 1, "true_label": 1160})
    assert response.status_code == 409
