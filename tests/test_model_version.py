"""/model must report the MLflow registry version, not the training timestamp.

No database and no MLflow server: the registry client and the model loader are
replaced by stubs.
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

from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

from src.models import predict as predict_module  # noqa: E402


class FakeClient:
    """Registry with version 7 behind @champion; its run carries a timestamp tag."""

    def get_model_version_by_alias(self, name, alias):
        return SimpleNamespace(version=7, run_id="run123")

    def get_run(self, run_id):
        return SimpleNamespace(
            data=SimpleNamespace(
                params={"model_name": "baseline_logreg"},
                tags={"version": "20261013220226"},
                metrics={"weighted_f1": 0.8},
            ),
            info=SimpleNamespace(start_time=1_760_000_000_000),
        )


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(predict_module, "_ARTIFACT", None)
    monkeypatch.setattr(predict_module, "MlflowClient", FakeClient)
    monkeypatch.setattr(predict_module.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(predict_module.mlflow.sklearn, "load_model", lambda uri: object())
    yield
    predict_module._ARTIFACT = None


def test_model_info_reports_registry_version_not_timestamp(registry):
    info = predict_module.model_info()
    assert info["version"] == "7"
    assert info["registered_model"] == "rakuten_baseline"
    assert info["run_id"] == "run123"


def test_predictions_are_tagged_with_registry_version(registry):
    assert predict_module.load_artifact()["version"] == "7"
