"""Training-outcome metrics: the gauge the training-failure alert reads (point 30)."""

from __future__ import annotations

from prometheus_client import REGISTRY

from src.monitoring import metrics as mon


def _value(name: str, labels: dict[str, str] | None = None) -> float:
    value = REGISTRY.get_sample_value(name, labels or {})
    return 0.0 if value is None else value


def test_failure_sets_gauge_until_next_success() -> None:
    failed_before = _value("rakuten_training_runs_total", {"outcome": "failed"})
    ok_before = _value("rakuten_training_runs_total", {"outcome": "succeeded"})

    mon.observe_training_outcome(succeeded=False)
    assert _value("rakuten_last_training_failed") == 1.0
    assert _value("rakuten_training_runs_total", {"outcome": "failed"}) == failed_before + 1

    # Stays at 1: nothing else clears it, so the alert does not resolve by itself.
    assert _value("rakuten_last_training_failed") == 1.0

    mon.observe_training_outcome(succeeded=True)
    assert _value("rakuten_last_training_failed") == 0.0
    assert _value("rakuten_training_runs_total", {"outcome": "succeeded"}) == ok_before + 1
