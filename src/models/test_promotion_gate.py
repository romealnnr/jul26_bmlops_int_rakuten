"""Unit tests of the promotion gate (point 21): no database, no MLflow server.

The gate must not promote on luck: a challenger replaces the @champion model only
when, on the same test rows the champion never trained on, the paired bootstrap
interval of the weighted-F1 gain is above 0 AND the gain reaches PROMOTION_MIN_GAIN.
"""

from __future__ import annotations

import os

# src.config reads these at import; a real .env (when present) takes precedence.
for _key, _value in {
    "POSTGRES_USER": "test", "POSTGRES_PASSWORD": "test", "POSTGRES_DB": "test",
    "POSTGRES_HOST": "localhost", "POSTGRES_PORT": "5432", "MODEL_NAME": "test_model",
    "TARGET_COLUMN": "prdtypecode", "TEST_SIZE": "0.2", "RANDOM_STATE": "42",
    "API_HOST": "0.0.0.0", "API_PORT": "8000", "LOG_LEVEL": "INFO",
    "MLFLOW_EXPERIMENT_NAME": "test", "MLFLOW_REGISTERED_MODEL_NAME": "test_model",
}.items():
    os.environ.setdefault(_key, _value)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from mlflow.protos import databricks_pb2  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from src.models import training  # noqa: E402

GATE = {"min_gain": 0.005, "n_bootstrap": 1000, "bootstrap_seed": 42,
        "first_min_f1": 0.2, "min_rows": 100}


@pytest.fixture(autouse=True)
def _clean_gate_env(monkeypatch):
    for key in ("PROMOTION_MIN_GAIN", "PROMOTION_BOOTSTRAP", "PROMOTION_BOOTSTRAP_SEED",
                "PROMOTION_FIRST_MIN_F1", "PROMOTION_MIN_ROWS"):
        monkeypatch.delenv(key, raising=False)


def _labels(n: int, n_classes: int = 5, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_classes, size=n)


def _with_errors(y: np.ndarray, n_wrong: int, n_classes: int = 5) -> np.ndarray:
    pred = y.copy()
    pred[:n_wrong] = (pred[:n_wrong] + 1) % n_classes
    return pred


# paired bootstrap -------------------------------------------------------------------

def test_point_scores_match_sklearn_weighted_f1():
    y = _labels(600)
    new, old = _with_errors(y, 60), _with_errors(y, 150)
    result = training.paired_bootstrap_gain(y, new, old, n_resamples=200)
    assert result["challenger_f1"] == pytest.approx(f1_score(y, new, average="weighted"))
    assert result["champion_f1"] == pytest.approx(f1_score(y, old, average="weighted"))
    assert result["n_rows"] == 600 and result["n_bootstrap"] == 200


def test_bootstrap_is_reproducible_with_the_same_seed():
    y = _labels(400)
    a = training.paired_bootstrap_gain(y, _with_errors(y, 40), _with_errors(y, 80), 300, 7)
    b = training.paired_bootstrap_gain(y, _with_errors(y, 40), _with_errors(y, 80), 300, 7)
    assert (a["ci_low"], a["ci_high"]) == (b["ci_low"], b["ci_high"])


def test_bootstrap_rejects_unpaired_inputs():
    with pytest.raises(ValueError):
        training.paired_bootstrap_gain([1, 2, 3], [1, 2], [1, 2, 3])


# decision rule ----------------------------------------------------------------------

def test_tiny_lucky_gain_is_rejected():
    """The point-21 case: one more correct row out of 2000 is noise, not a better model."""
    y = _labels(2000)
    old = _with_errors(y, 301)
    new = _with_errors(y, 300)
    paired = training.paired_bootstrap_gain(y, new, old)
    assert 0 < paired["gain"] < 0.005
    decision, reason, is_best = training.decide_promotion("3", paired["challenger_f1"],
                                                          paired, GATE)
    assert (decision, is_best) == ("rejected", False)
    assert "interval includes 0" in reason or "below the minimum" in reason


def test_identical_models_are_rejected():
    y = _labels(500)
    pred = _with_errors(y, 50)
    paired = training.paired_bootstrap_gain(y, pred, pred)
    assert paired["gain"] == 0 and paired["ci_low"] == 0
    assert training.decide_promotion("3", 0.9, paired, GATE)[0] == "rejected"


def test_clear_gain_is_promoted():
    y = _labels(1000)
    paired = training.paired_bootstrap_gain(y, _with_errors(y, 50), _with_errors(y, 250))
    assert paired["ci_low"] > 0
    decision, reason, is_best = training.decide_promotion("3", 0.95, paired, GATE)
    assert (decision, is_best) == ("promoted", True) and "Promoted" in reason


def test_real_but_small_gain_below_minimum_is_rejected():
    y = _labels(1000)
    paired = training.paired_bootstrap_gain(y, _with_errors(y, 50), _with_errors(y, 250))
    gate = {**GATE, "min_gain": 0.5}
    decision, reason, _ = training.decide_promotion("3", 0.95, paired, gate)
    assert decision == "rejected" and "gain below the minimum" in reason


def test_first_model_needs_the_floor():
    assert training.decide_promotion(None, 0.5, None, GATE)[0] == "promoted"
    assert training.decide_promotion(None, 0.1, None, GATE)[0] == "rejected"


def test_no_paired_comparison_is_never_a_promotion():
    decision, reason, is_best = training.decide_promotion("3", 0.99, None, GATE, "x")
    assert (decision, is_best) == ("rejected", False) and "not auto-promoted" in reason


def test_settings_from_environment_and_validation(monkeypatch):
    monkeypatch.setenv("PROMOTION_MIN_GAIN", "0.01")
    assert training._promotion_settings()["min_gain"] == 0.01
    monkeypatch.setenv("PROMOTION_BOOTSTRAP", "10")
    with pytest.raises(ValueError):
        training._promotion_settings()


# _compare_and_promote with a fake registry ------------------------------------------

class _Run:
    def __init__(self, metrics, params):
        self.data = type("D", (), {"metrics": metrics, "params": params})()


class FakeClient:
    """Registry with an optional @champion version (alias API, no stages)."""

    def __init__(self, champion=None, params=None, fail=False):
        self.champion, self.params, self.fail = champion, params or {}, fail
        self.tags, self.aliases = {}, {}

    def get_model_version_by_alias(self, name, alias):
        if self.fail:
            raise MlflowException("registry unreachable", error_code=databricks_pb2.INTERNAL_ERROR)
        if self.champion is None:
            raise MlflowException(
                f"Registered model alias {alias} not found.",
                error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
            )
        return type("V", (), {"version": self.champion, "run_id": "r1"})()

    def search_model_versions(self, filter_string=None):
        if self.fail:
            raise RuntimeError("registry unreachable")
        return []

    def get_run(self, run_id):
        return _Run({"weighted_f1": 0.80}, self.params)

    def set_model_version_tag(self, name, version, key, value):
        self.tags[key] = value

    def set_registered_model_alias(self, name, alias, version):
        self.aliases[alias] = version


def _batch(n=400, first_id=1001):
    y = pd.Series(_labels(n), index=range(10, 10 + n))
    X = pd.DataFrame({"designation": ["x"] * n}, index=y.index)
    ids = np.arange(first_id, first_id + n)
    return X, y, ids


def test_compare_promotes_on_clear_gain_and_only_scores_unseen_rows(monkeypatch):
    X, y, ids = _batch()
    seen = {}

    def fake_champion(model_name, X_eval):
        seen["rows"] = len(X_eval)
        return _with_errors(y.loc[X_eval.index].to_numpy(), 120)

    monkeypatch.setattr(training, "_score_champion_on", fake_champion)
    client = FakeClient(champion="3", params={"window_max_id": "1100"})
    result = training._compare_and_promote(
        client, "m", "4", {"weighted_f1": 0.95}, X, y, _with_errors(y.to_numpy(), 5), ids,
    )
    assert seen["rows"] == 300  # ids 1101..1400: rows the champion never trained on
    assert result["decision"] == "promoted" and client.aliases[training.CHAMPION_ALIAS] == "4"
    assert result["comparison_basis"] == "paired_bootstrap_unseen_rows"
    assert result["evaluated_on_rows"] == 300 and result["gain_ci_low"] > 0


def test_compare_rejects_when_champion_window_unknown(monkeypatch):
    X, y, ids = _batch()
    monkeypatch.setattr(training, "_score_champion_on", lambda *a: pytest.fail("scored"))
    client = FakeClient(champion="3", params={})
    result = training._compare_and_promote(
        client, "m", "4", {"weighted_f1": 0.99}, X, y, y.to_numpy(), ids,
    )
    assert result["comparison_basis"] == "champion_window_unknown"
    assert result["decision"] == "rejected" and client.aliases[training.CHALLENGER_ALIAS] == "4"


def test_compare_rejects_when_too_few_unseen_rows(monkeypatch):
    X, y, ids = _batch()
    monkeypatch.setattr(training, "_score_champion_on", lambda *a: pytest.fail("scored"))
    client = FakeClient(champion="3", params={"window_max_id": "1350"})
    result = training._compare_and_promote(
        client, "m", "4", {"weighted_f1": 0.99}, X, y, y.to_numpy(), ids,
    )
    assert result["comparison_basis"] == "too_few_unseen_rows"
    assert result["decision"] == "rejected" and result["evaluated_on_rows"] == 50


def test_compare_rejects_when_champion_not_scorable(monkeypatch):
    X, y, ids = _batch()
    monkeypatch.setattr(training, "_score_champion_on", lambda *a: None)
    client = FakeClient(champion="3", params={"window_max_id": "1000"})
    result = training._compare_and_promote(
        client, "m", "4", {"weighted_f1": 0.99}, X, y, y.to_numpy(), ids,
    )
    assert result["comparison_basis"] == "champion_not_scorable"
    assert result["decision"] == "rejected"


def test_compare_first_model_and_unreadable_registry():
    X, y, ids = _batch()
    client = FakeClient(champion=None)
    result = training._compare_and_promote(
        client, "m", "1", {"weighted_f1": 0.6}, X, y, y.to_numpy(), ids,
    )
    assert result["decision"] == "promoted" and result["comparison_basis"] == "no_previous_model"
    # An unreadable registry is an error, never "no champion yet".
    with pytest.raises(MlflowException):
        training._compare_and_promote(
            FakeClient(fail=True), "m", "2", {"weighted_f1": 0.6}, X, y, y.to_numpy(), ids,
        )


def test_invalid_setting_fails_training_before_any_work(monkeypatch):
    monkeypatch.setenv("PROMOTION_MIN_GAIN", "-1")
    monkeypatch.setattr(training, "ensure_dirs", lambda: None)
    monkeypatch.setattr(training, "load_dataframe", lambda **k: pytest.fail("data loaded"))
    monkeypatch.setattr(training, "_setup_mlflow", lambda: pytest.fail("MLflow touched"))
    with pytest.raises(ValueError, match="PROMOTION_MIN_GAIN"):
        training.train()
