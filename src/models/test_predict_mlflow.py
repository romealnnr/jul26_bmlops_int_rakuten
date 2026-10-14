"""predict() ne doit ouvrir aucun run MLflow (point 32).

Sans base ni serveur MLflow : le modèle servi est remplacé par un faux pipeline.
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

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.models import predict as predict_module  # noqa: E402


class FakePipeline:
    classes_ = np.array([10, 40])

    def predict(self, frame):
        return np.array([10] * len(frame))

    def predict_proba(self, frame):
        return np.array([[0.9, 0.1]] * len(frame))


@pytest.fixture
def served(monkeypatch):
    artifact = {"pipeline": FakePipeline(), "feature_names": ["designation"], "version": "7"}
    monkeypatch.setattr(predict_module, "load_artifact", lambda *a, **k: artifact)


def test_predict_creates_no_mlflow_run(served, monkeypatch):
    # Record calls instead of raising: an exception could be swallowed by a
    # try/except around the MLflow logging and hide the call.
    touched = []
    for name in ("set_tracking_uri", "set_experiment", "start_run", "log_metrics", "set_tags"):
        monkeypatch.setattr(
            predict_module.mlflow, name, lambda *a, _n=name, **k: touched.append(_n)
        )
    results = predict_module.predict([{"designation": "carte pokemon"}], log_to_db=False)
    assert results[0]["prediction"] == 10
    assert touched == []


def test_log_to_mlflow_is_accepted_but_ignored(served, monkeypatch):
    # Kept for existing callers: it warns and still opens no MLflow run.
    monkeypatch.setattr(
        predict_module.mlflow, "start_run", lambda *a, **k: pytest.fail("no MLflow run expected")
    )
    with pytest.warns(DeprecationWarning):
        predict_module.predict([{"designation": "x"}], log_to_db=False, log_to_mlflow=True)


def test_batch_summary_goes_to_separate_experiment(monkeypatch):
    calls = {}

    class Run:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(predict_module.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(
        predict_module.mlflow, "set_experiment", lambda name: calls.setdefault("exp", name)
    )
    monkeypatch.setattr(predict_module.mlflow, "start_run", lambda **k: Run())
    monkeypatch.setattr(predict_module.mlflow, "set_tags", lambda tags: None)
    monkeypatch.setattr(
        predict_module.mlflow, "log_metrics", lambda m: calls.setdefault("metrics", m)
    )

    predict_module.log_batch_to_mlflow(
        [{"flagged": True, "model_version": "7"}, {"flagged": False, "model_version": "7"}]
    )
    assert calls["exp"] == f"{predict_module.settings.mlflow_experiment_name}-predictions"
    assert calls["metrics"]["n_predictions"] == 2
    assert calls["metrics"]["n_flagged"] == 1
