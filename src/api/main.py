"""API d'inférence : endpoints /training et /predict."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from src.api.schemas import PredictRequest
from src.config import ensure_dirs, setup_logging
from src.data.db import init_db, ping
from src.models import predict as predict_module
from src.models.predict import ModelNotFoundError
from src.models.training import train
from src.monitoring import metrics as mon
from src.monitoring.drift import compute_drift

logger = logging.getLogger(__name__)


def _refresh_model_metrics() -> None:
    """Point the model gauges at whatever is actually loadable right now.

    Called at startup and after every training run. `model_info()` is the
    honest question to ask: /health only checks that the artifact file
    exists, and a file that exists but cannot be loaded would report a
    healthy model while every prediction fails.
    """
    try:
        mon.observe_model(predict_module.model_info())
    except ModelNotFoundError:
        mon.observe_model(None)
    except Exception:  # noqa: BLE001
        # An artifact that is present but unreadable is not a loaded model.
        logger.exception("Impossible de lire les métadonnées du modèle")
        mon.observe_model(None)


def _training_in_progress() -> bool:
    """Is a training run executing right now?

    Reads the gauge through the collector's own output rather than the
    private `_value`, so it keeps working if prometheus_client changes its
    internals. The monitoring metric doubles as the lock: the number that
    tells a dashboard the trainer is busy is the same number that stops a
    second run being started.
    """
    try:
        return any(
            sample.value >= 1
            for metric in mon.TRAINING_IN_PROGRESS.collect()
            for sample in metric.samples
        )
    except Exception:  # noqa: BLE001
        # Never let a monitoring read decide that training is impossible.
        return False


def _train_and_record(**kwargs: Any) -> dict[str, Any]:
    """Run a training and keep the metrics honest, wherever it was started from.

    Both the manual endpoint and the alert webhook go through here, so a run
    triggered by Grafana is counted, timed and reflected in the model gauges
    exactly like one a person asked for. Raises on failure; each caller
    decides what to do with that.
    """
    mon.TRAINING_IN_PROGRESS.set(1)
    started = time.perf_counter()
    try:
        summary = train(**kwargs)
    except Exception:
        mon.TRAINING_RUNS.labels(outcome="failed").inc()
        raise
    finally:
        # Cleared here and nowhere else: a run that crashes must not leave the
        # gauge stuck at 1, which would both lie to the dashboard and lock the
        # webhook out permanently. Duration is recorded for failures too - how
        # long a run lasts before it breaks is the difference between a bad
        # argument and an exhausted machine.
        mon.TRAINING_IN_PROGRESS.set(0)
        mon.TRAINING_DURATION.observe(time.perf_counter() - started)

    mon.TRAINING_RUNS.labels(outcome="succeeded").inc()
    predict_module.load_artifact(force_reload=True)
    _refresh_model_metrics()
    return summary


def _alert_triggered_training() -> None:
    """The background job the drift webhook starts.

    Exceptions are swallowed on purpose: this runs after the HTTP response has
    already gone back to Grafana, so there is nobody left to return an error
    to. The failure is recorded by _train_and_record in
    rakuten_training_runs_total{outcome="failed"} and in the log, which is
    where anyone would look for it.
    """
    try:
        summary = _train_and_record(continuous=True)
        logger.info(
            "Alert-triggered training finished: version %s, %s",
            summary.get("version"), summary.get("comparison", {}).get("decision"),
        )
    except Exception:  # noqa: BLE001
        logger.exception("Alert-triggered training failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    ensure_dirs()
    if ping():
        init_db()
    else:
        logger.warning("PostgreSQL injoignable au démarrage")

    # Model gauges start out telling the truth rather than starting at zero:
    # an API that comes up without a model should say so on its first scrape,
    # not on its first failed prediction.
    _refresh_model_metrics()

    # The drift summary - age, share, column count - is computed at scrape
    # time, straight from the database. A background task updating it would
    # die with the thing it is meant to be watching, and an in-process gauge
    # would be known only to whichever replica served the drift run.
    mon.register_drift_collectors()

    yield


app = FastAPI(
    title="MLOps Phase 1 — Inference API",
    description="Entraînement et inférence du modèle baseline.",
    version="0.1.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


# --------------------------------------------------------------------------- schémas
class TrainingRequest(BaseModel):
    test_size: float | None = Field(default=None, gt=0, lt=1)
    random_state: int | None = None
    limit: int | None = Field(default=None, gt=0, description="Limiter à N lignes (test rapide)")
    continuous: bool = Field(
        default=False,
        description="Mode versioning continu : traite la prochaine tranche depuis pipeline_cursor",
    )


class TrainingResponse(BaseModel):
    status: str
    model_name: str
    version: str
    trained_at: str
    n_train: int
    n_test: int
    n_features: int
    metrics: dict[str, Any]
    mlflow_run_id: str | None = None
    mlflow_model_version: str | None = None
    comparison: dict[str, Any] | None = None


class CategoryScore(BaseModel):
    prediction: int
    category_name: str
    probability: float


class PredictionItem(BaseModel):
    prediction: int
    category_name: str
    probability: float
    model_version: str
    flagged: bool
    flag_reason: str | None = None
    top3: list[CategoryScore]
    image_used: bool
    warnings: list[str] = Field(default_factory=list)
    true_label: int | None = Field(
        default=None, description="True category if the record gave one (optional)"
    )
    prediction_id: int | None = Field(
        default=None, description="Row id in predictions; null if it could not be stored"
    )


class PredictResponse(BaseModel):
    count: int
    flag_threshold: float
    results: list[PredictionItem]


class FeedbackRequest(BaseModel):
    """True category of an earlier prediction, sent once it is known."""

    model_config = {"extra": "forbid"}

    prediction_id: int = Field(gt=0)
    true_label: int = Field(strict=True, description="One of the categories the model knows")


class FeedbackResponse(BaseModel):
    prediction_id: int
    prediction: int
    true_label: int
    correct: bool
    labeled_at: datetime


# --------------------------------------------------------------------------- endpoints
@app.get("/health", tags=["monitoring"])
def health() -> dict[str, Any]:
    """État de l'API, de la base et du modèle (vérifié via le MLflow Model Registry)."""
    try:
        predict_module.model_info()
        model_available = True
    except ModelNotFoundError:
        model_available = False
    return {
        "status": "ok",
        "database": "up" if ping() else "down",
        "model_available": model_available,
    }


@app.post("/training", response_model=TrainingResponse, tags=["model"])
async def run_training(request: TrainingRequest) -> TrainingResponse:
    """Réentraîne le modèle baseline à partir des données en base."""
    try:
        summary = await run_in_threadpool(
            _train_and_record,
            test_size=request.test_size,
            random_state=request.random_state,
            limit=request.limit,
            continuous=request.continuous,
        )
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Échec de l'entraînement")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return TrainingResponse(status="trained", **summary)


@app.post("/training/trigger", tags=["monitoring"])
async def training_trigger(background: BackgroundTasks) -> dict[str, Any]:
    """Entry point for Grafana's webhook contact point (Phase 3 of the brief).

    Why this exists instead of pointing the webhook at /training:

    Grafana sends its own alert payload, which says nothing about how to
    train. TrainingRequest would accept it and read every field as its
    default - meaning `continuous: false`, a full retrain over every row,
    more than an hour - each time the alert fired. The decision is fixed here
    instead: a drift alert means "process the next slice", never "retrain on
    everything".

    Why it answers immediately:

    Grafana's webhook times out in seconds and retries what it believes
    failed. A training run takes minutes. Answering synchronously would turn
    one alert into a queue of identical runs, so the work is handed to a
    background task and the call returns at once.

    The request body is ignored entirely - whatever Grafana sends, this does
    one specific thing.
    """
    if _training_in_progress():
        # The third brake, after `for: 10m` on the rule and `repeat_interval`
        # on the notification policy. Drift does not clear the moment a model
        # is retrained - it clears when the next drift report is computed on a
        # new window - so the alert stays firing through the run that is
        # already dealing with it.
        logger.info("Alert-triggered training skipped: a run is already in progress")
        return {"status": "skipped", "reason": "a training run is already in progress"}

    background.add_task(_alert_triggered_training)
    logger.info("Alert-triggered training accepted (continuous mode)")
    return {"status": "accepted", "mode": "continuous"}


@app.post("/predict", response_model=PredictResponse, tags=["model"])
async def run_predict(request: PredictRequest) -> PredictResponse:
    """Prédit la classe pour 1 à 1000 observations.

    Les enregistrements sont validés avant l'inférence : 422 si designation est
    absente ou vide, ou si un champ inconnu est envoyé.
    """
    records = [record.model_dump() for record in request.records]
    try:
        results = await run_in_threadpool(predict_module.predict, records)
    except ModelNotFoundError as exc:
        # The model went away between startup and now; say so immediately
        # rather than waiting for the next training run to correct the gauge.
        mon.observe_model(None)
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Échec de l'inférence")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    # Instrumented here, at the HTTP boundary, rather than inside predict():
    # prometheus_client counters live in the memory of the process that
    # increments them, and this is the only process that serves /metrics.
    # Counting inside predict() would silently record nothing whenever it is
    # called from a DAG or the CLI.
    mon.observe_predictions(results)

    return PredictResponse(
        count=len(results),
        flag_threshold=predict_module.FLAG_THRESHOLD,
        results=results,
    )


@app.post("/feedback", response_model=FeedbackResponse, tags=["model"])
def submit_feedback(request: FeedbackRequest) -> dict[str, Any]:
    """Store the true category of an earlier prediction (by its prediction_id).

    The label goes to predictions.true_label only; it is never used as a model
    input. Unknown prediction ids and unknown categories return 422; 409 when no
    model is loaded (the known categories are read from the served model).
    """
    try:
        return predict_module.record_feedback(request.prediction_id, request.true_label)
    except ModelNotFoundError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except (LookupError, ValueError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@app.get("/model", tags=["model"])
def model_metadata() -> dict[str, Any]:
    """Métadonnées et métriques du modèle actuellement servi."""
    try:
        return predict_module.model_info()
    except ModelNotFoundError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@app.post("/drift/compute", tags=["monitoring"])
async def run_drift_computation() -> dict[str, Any]:
    """Calcule le data drift entre la baseline et le dernier batch traité,
    et persiste les métriques en base pour Grafana."""
    from src.data.db import get_session
    from src.models.training import _get_cursor

    with get_session() as session:
        current_max_id = _get_cursor(session)

    if current_max_id == 0:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="Aucun entraînement continu n'a encore tourné (curseur à 0).",
        )

    try:
        result = await run_in_threadpool(compute_drift, current_max_id)
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Échec du calcul de drift")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    # Nothing is pushed to Prometheus here on purpose. The drift gauges read
    # the drift_metrics table when Prometheus scrapes them, so the numbers are
    # correct for every replica - including the ones that never served this
    # request - and they stay correct if this process restarts.
    return result
