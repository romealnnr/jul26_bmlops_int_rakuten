"""Unit tests for point 24: registry aliases instead of legacy stages, and a clean
"no_new_data" result once continuous training has consumed every row.

No database and no MLflow server: a fake MlflowClient records alias calls."""
from __future__ import annotations

import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2

# Settings are read from the environment at import time; harmless test-only values
# so the modules import without a .env file (no connection is made).
for _key, _value in {
    "POSTGRES_USER": "t", "POSTGRES_PASSWORD": "t", "POSTGRES_DB": "t",
    "POSTGRES_HOST": "localhost", "POSTGRES_PORT": "5432",
    "MODEL_NAME": "rakuten_baseline", "TARGET_COLUMN": "prdtypecode", "TEST_SIZE": "0.2",
    "RANDOM_STATE": "42", "API_HOST": "0.0.0.0", "API_PORT": "8000", "LOG_LEVEL": "INFO",
    "MLFLOW_EXPERIMENT_NAME": "unit-tests", "MLFLOW_REGISTERED_MODEL_NAME": "rakuten_baseline",
}.items():
    os.environ.setdefault(_key, _value)

from src.models import predict, training  # noqa: E402

MODEL = "rakuten_baseline"


class FakeClient:
    """Minimal stand-in for mlflow.tracking.MlflowClient."""

    def __init__(self, champion=None, production=None, metrics=None):
        self.aliases: dict[str, str] = {}
        if champion is not None:
            self.aliases[training.CHAMPION_ALIAS] = str(champion)
        self.production = production  # version still in the legacy "Production" stage
        self.metrics = metrics or {}
        self.tags: dict[tuple[str, str], str] = {}
        self.stage_writes: list = []

    def get_model_version_by_alias(self, name, alias):
        if alias not in self.aliases:
            # What MLflow 3 answers for an unknown alias on a known model.
            raise MlflowException(
                f"Registered model alias {alias} not found.",
                error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
            )
        version = self.aliases[alias]
        return SimpleNamespace(version=version, run_id=f"run-{version}")

    def search_model_versions(self, filter_string=None):
        if self.production is None:
            return []
        return [
            SimpleNamespace(version="1", run_id="run-1", current_stage="Archived"),
            SimpleNamespace(
                version=str(self.production),
                run_id=f"run-{self.production}",
                current_stage="Production",
            ),
        ]

    def get_latest_versions(self, *args, **kwargs):
        raise AssertionError("get_latest_versions(stages=...) is deprecated and must not be used")

    def transition_model_version_stage(self, *args, **kwargs):
        self.stage_writes.append((args, kwargs))

    def set_registered_model_alias(self, name, alias, version):
        self.aliases[alias] = str(version)

    def set_model_version_tag(self, name, version, key, value):
        self.tags[(str(version), key)] = value

    def get_run(self, run_id):
        return SimpleNamespace(data=SimpleNamespace(metrics=self.metrics))


Y_TEST = [10, 20, 30]


def _promote(client, new_version, new_score, champion_score, monkeypatch):
    monkeypatch.setattr(training, "_score_champion_on", lambda *a, **k: champion_score)
    return training._compare_and_promote(
        client, MODEL, new_version, {training.PRIMARY_METRIC: new_score}, None, Y_TEST
    )


def test_first_training_sets_champion_alias_and_writes_no_stage(monkeypatch):
    client = FakeClient()
    result = _promote(client, "1", 0.7, None, monkeypatch)
    assert result["decision"] == "promoted"
    assert result["alias"] == training.CHAMPION_ALIAS
    assert client.aliases == {training.CHAMPION_ALIAS: "1"}
    assert client.stage_writes == []


def test_better_model_moves_champion(monkeypatch):
    client = FakeClient(champion=3, metrics={"weighted_f1": 0.6})
    result = _promote(client, "4", 0.8, 0.6, monkeypatch)
    assert result["decision"] == "promoted"
    assert client.aliases[training.CHAMPION_ALIAS] == "4"
    assert client.stage_writes == []


def test_worse_model_becomes_challenger_and_champion_stays(monkeypatch):
    client = FakeClient(champion=3, metrics={"weighted_f1": 0.8})
    result = _promote(client, "4", 0.5, 0.8, monkeypatch)
    assert result["decision"] == "rejected"
    assert client.aliases[training.CHAMPION_ALIAS] == "3"
    assert client.aliases[training.CHALLENGER_ALIAS] == "4"
    assert client.stage_writes == []


def test_legacy_production_version_is_migrated_to_champion():
    client = FakeClient(production=5, metrics={"weighted_f1": 0.7})
    version, metrics = training._get_champion_version(client, MODEL)
    assert version == "5"
    assert metrics == {"weighted_f1": 0.7}
    assert client.aliases[training.CHAMPION_ALIAS] == "5"


def test_no_champion_and_no_legacy_version():
    assert training._get_champion_version(FakeClient(), MODEL) == (None, None)


def test_continuous_training_returns_no_new_data_when_exhausted(monkeypatch):
    import src.data.db as db

    @contextmanager
    def fake_session():
        yield object()

    monkeypatch.setattr(training, "ensure_dirs", lambda: None)
    monkeypatch.setattr(training, "_setup_mlflow", lambda: None)
    monkeypatch.setattr(db, "get_session", fake_session)
    monkeypatch.setattr(training, "_get_cursor", lambda session: 84916)
    monkeypatch.setattr(training, "_has_rows_after", lambda session, cursor: False)

    def must_not_load(*args, **kwargs):
        raise AssertionError("no data should be loaded when the cursor is at the end")

    monkeypatch.setattr(training, "load_dataframe", must_not_load)

    result = training.train(continuous=True)
    assert result["status"] == "no_new_data"
    assert result["cursor"] == 84916


@pytest.mark.parametrize(
    ("client", "expected"),
    [
        (FakeClient(champion=7, production=5), "7"),  # alias wins
        (FakeClient(production=5), "5"),  # read-only fallback before the first new training
        (FakeClient(), None),
    ],
)
def test_api_serves_champion_alias(client, expected):
    version = predict._champion_version(client, MODEL)
    assert (version.version if version else None) == expected
    assert client.stage_writes == []
    if expected == "5":
        assert training.CHAMPION_ALIAS not in client.aliases  # serving never writes


def test_training_response_accepts_no_new_data():
    pytest.importorskip("fastapi")
    from src.api.main import TrainingResponse

    response = TrainingResponse(status="no_new_data", detail="nothing left", cursor=84916)
    assert response.status == "no_new_data"
    assert response.version is None


def test_no_new_data_keeps_served_model(monkeypatch):
    pytest.importorskip("fastapi")
    from src.api import main

    monkeypatch.setattr(
        main, "train", lambda **kwargs: {"status": "no_new_data", "detail": "x", "cursor": 1}
    )

    def must_not_reload(*args, **kwargs):
        raise AssertionError("the served model must not be reloaded when nothing was trained")

    monkeypatch.setattr(main.predict_module, "load_artifact", must_not_reload)
    # _record_training is what _train_and_record runs once it holds the lock.
    assert main._record_training(continuous=True)["status"] == "no_new_data"


class UnreachableClient(FakeClient):
    """Registry that answers with a server error instead of "not found"."""

    def get_model_version_by_alias(self, name, alias):
        raise MlflowException("connection refused", error_code=databricks_pb2.INTERNAL_ERROR)


def test_registry_error_is_not_read_as_missing_champion_in_predict():
    with pytest.raises(MlflowException):
        predict._champion_version(UnreachableClient(production=3), MODEL)


def test_registry_error_is_not_read_as_missing_champion_in_training():
    # Reading it as "no champion" would promote the new model as the first one.
    with pytest.raises(MlflowException):
        training._get_champion_version(UnreachableClient(production=3), MODEL)


@pytest.mark.parametrize(
    ("error_code", "message", "expected"),
    [
        ("RESOURCE_DOES_NOT_EXIST", "Registered Model with name=x not found", True),
        ("INVALID_PARAMETER_VALUE", "Registered model alias champion not found.", True),
        ("INVALID_PARAMETER_VALUE", "Invalid model name", False),
        ("INTERNAL_ERROR", "connection refused", False),
    ],
)
def test_is_not_found(error_code, message, expected):
    from src.models.registry_errors import is_not_found

    code = getattr(databricks_pb2, error_code)
    assert is_not_found(MlflowException(message, error_code=code)) is expected
    assert is_not_found(RuntimeError(message)) is False
