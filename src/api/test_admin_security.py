"""API key on /training and /drift/compute, and one training at a time.

No database or model is needed: training and the lock are replaced by fakes.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.data.db import TrainingInProgressError

KEY = "unit-test-key-123"
SUMMARY = {
    "model_name": "m",
    "version": "v1",
    "trained_at": "2026-01-01T00:00:00",
    "n_train": 8,
    "n_test": 2,
    "n_features": 3,
    "metrics": {},
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_KEY", KEY)

    @contextmanager
    def free_lock():
        yield

    monkeypatch.setattr(main, "training_lock", free_lock)
    monkeypatch.setattr(main, "_record_training", lambda **kwargs: SUMMARY)
    # No `with`: the lifespan (database, model loading) is not started.
    return TestClient(main.app)


@pytest.mark.parametrize("path", ["/training", "/training/trigger", "/drift/compute"])
def test_admin_endpoints_refuse_missing_or_wrong_key(client, path):
    assert client.post(path, json={}).status_code == 401
    assert client.post(path, json={}, headers={"X-API-Key": "wrong"}).status_code == 401


@pytest.mark.parametrize("path", ["/training", "/training/trigger", "/drift/compute"])
def test_admin_endpoints_fail_closed_without_server_key(client, monkeypatch, path):
    monkeypatch.delenv("API_KEY")
    assert client.post(path, json={}, headers={"X-API-Key": KEY}).status_code == 503


def test_training_with_key_runs(client):
    response = client.post("/training", json={"limit": 10}, headers={"X-API-Key": KEY})
    assert response.status_code == 200
    assert response.json()["version"] == "v1"


def test_bearer_token_is_accepted(client):
    response = client.post("/training", json={}, headers={"Authorization": f"Bearer {KEY}"})
    assert response.status_code == 200


def test_second_training_gets_409(client, monkeypatch):
    @contextmanager
    def busy_lock():
        raise TrainingInProgressError()
        yield  # pragma: no cover

    monkeypatch.setattr(main, "training_lock", busy_lock)
    response = client.post("/training", json={}, headers={"X-API-Key": KEY})
    assert response.status_code == 409
    assert "already in progress" in response.json()["detail"]


def test_read_only_endpoints_stay_open(client, monkeypatch):
    monkeypatch.setattr(main.predict_module, "model_info", lambda: {"version": "v1"})
    assert client.get("/model").status_code == 200
