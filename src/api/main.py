"""API d'inférence : endpoints /training et /predict."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from src.config import ensure_dirs, settings, setup_logging
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

    # The drift report's age is computed at scrape time, straight from the
    # database. A background task updating it would die with the thing it is
    # meant to be watching.
    mon.register_drift_age_collector()

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


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "records": [
                    {
                        "designation": "Console de jeu portable retro",
                        "description": "400 jeux inclus, écran couleur",
                        "productid": 123456,
                        "imageid": 987654,
                        "image_path": None,
                    }
                ]
            }
        }
    }

class PredictionItem(BaseModel):
    prediction: int
    probability: float
    model_version: str


class PredictResponse(BaseModel):
    count: int
    results: list[PredictionItem]


# --------------------------------------------------------------------------- endpoints
@app.get("/health", tags=["monitoring"])
def health() -> dict[str, Any]:
    """État de l'API, de la base et du modèle."""
    model_available = settings.model_path.exists()
    return {
        "status": "ok",
        "database": "up" if ping() else "down",
        "model_available": model_available,
    }


@app.post("/training", response_model=TrainingResponse, tags=["model"])
async def run_training(request: TrainingRequest) -> TrainingResponse:
    """Réentraîne le modèle baseline à partir des données en base."""
    # Set before the call and cleared in `finally`: a training run that
    # crashes must not leave the gauge stuck at 1 forever, which would hide
    # every subsequent run behind an apparently busy trainer.
    mon.TRAINING_IN_PROGRESS.set(1)
    started = time.perf_counter()
    try:
        summary = await run_in_threadpool(
            train,
            test_size=request.test_size,
            random_state=request.random_state,
            limit=request.limit,
            continuous=request.continuous,
        )
    except RuntimeError as exc:
        mon.TRAINING_RUNS.labels(outcome="failed").inc()
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        mon.TRAINING_RUNS.labels(outcome="failed").inc()
        logger.exception("Échec de l'entraînement")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    finally:
        mon.TRAINING_IN_PROGRESS.set(0)
        # Recorded for failures too. How long a run takes before it breaks is
        # the difference between a bad argument and an exhausted machine.
        mon.TRAINING_DURATION.observe(time.perf_counter() - started)

    mon.TRAINING_RUNS.labels(outcome="succeeded").inc()

    predict_module.load_artifact(force_reload=True)
    _refresh_model_metrics()

    return TrainingResponse(status="trained", **summary)


@app.post("/predict", response_model=PredictResponse, tags=["model"])
async def run_predict(request: PredictRequest) -> PredictResponse:
    """Prédit la classe pour une ou plusieurs observations."""
    try:
        results = await run_in_threadpool(predict_module.predict, request.records)
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

    return PredictResponse(count=len(results), results=results)


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

    # Only the summary goes to Prometheus; the per-column scores stay in
    # drift_metrics, which is the right store for them.
    mon.observe_drift(result["metrics"])

    return result
