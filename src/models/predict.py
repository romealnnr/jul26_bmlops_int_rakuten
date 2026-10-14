"""Inférence à partir du modèle entraîné.

Usage :
    python -m src.models.predict --sample
    python -m src.models.predict --input payload.json [--mlflow]

predict() never opens an MLflow run: the API calls it once per /predict request,
and a run per request would flood the training experiment. A batch summary is
logged only on demand from the CLI (--mlflow), in a separate experiment
"<MLFLOW_EXPERIMENT_NAME>-predictions".
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings as py_warnings
from datetime import UTC
from datetime import datetime as dt
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from PIL import Image

from src.config import RAW_DATA_DIR, settings, setup_logging
from src.data.db import Prediction, get_session
from src.features.build_features import get_feature_columns, load_dataframe, prepare_frame
from src.models.categories import category_name
from src.models.registry_errors import is_not_found

logger = logging.getLogger(__name__)

_ARTIFACT: dict[str, Any] | None = None

FLAG_THRESHOLD = 0.6  # en dessous de ce score de confiance, la prédiction est flaggée


CHAMPION_ALIAS = "champion"  # same alias as src.models.training.CHAMPION_ALIAS


class ModelNotFoundError(RuntimeError):
    """Aucun modèle @champion n'est disponible dans le MLflow Model Registry."""


def _champion_version(client: MlflowClient, model_name: str):
    """Registry version behind the @champion alias.

    Read-only fallback for a registry created before the alias change: the
    latest version still in the legacy "Production" stage is served until the
    next training run sets the alias. Returns None when neither exists.
    Any other registry error (unreachable, server error) is raised, so /health
    reports it instead of claiming that no model exists."""
    try:
        return client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    except Exception as exc:
        if not is_not_found(exc):
            raise
    legacy = [
        v
        for v in client.search_model_versions(f"name='{model_name}'")
        if v.current_stage == "Production"
    ]
    return max(legacy, key=lambda v: int(v.version)) if legacy else None


def load_artifact(force_reload: bool = False) -> dict[str, Any]:
    """Charge le modèle @champion depuis le MLflow Model Registry, avec cache
    mémoire. Contrairement à un fichier local, ceci garantit que l'API sert
    toujours exactement le modèle promu par _compare_and_promote, jamais un
    modèle rejeté (@challenger) lors d'un entraînement."""
    global _ARTIFACT

    if _ARTIFACT is not None and not force_reload:
        return _ARTIFACT

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    model_name = settings.mlflow_registered_model_name
    version_info = _champion_version(client, model_name)
    if version_info is None:
        raise ModelNotFoundError(
            f"Aucun modèle @{CHAMPION_ALIAS} pour '{model_name}'. "
            "Lancez d'abord : python -m src.models.training"
        )

    run = client.get_run(version_info.run_id)
    # Pinned to the resolved version, so the metadata and the weights always match.
    model_uri = f"models:/{model_name}/{version_info.version}"
    pipeline = mlflow.sklearn.load_model(model_uri)

    _ARTIFACT = {
        "pipeline": pipeline,
        "model_name": run.data.params.get("model_name", settings.model_name),
        "version": run.data.tags.get("version", str(version_info.version)),
        "mlflow_model_version": str(version_info.version),
        "mlflow_run_id": version_info.run_id,
        "trained_at": dt.fromtimestamp(run.info.start_time / 1000, tz=UTC).isoformat(),
        "feature_names": get_feature_columns(None),
        "metrics": dict(run.data.metrics),
    }
    logger.info(
        "Modèle chargé depuis MLflow Model Registry (version %s, run %s)",
        version_info.version, version_info.run_id,
    )
    return _ARTIFACT


def model_info() -> dict[str, Any]:
    """Métadonnées du modèle courant, sans les objets sklearn."""
    artifact = load_artifact()
    return {
        "model_name": artifact["model_name"],
        "version": artifact["version"],
        "trained_at": artifact["trained_at"],
        "n_features": len(artifact["feature_names"]),
        "metrics": artifact["metrics"],
    }


def log_batch_to_mlflow(results: list[dict[str, Any]]) -> None:
    """CLI uniquement : journalise un résumé du batch dans MLflow.

    Expérience séparée "<MLFLOW_EXPERIMENT_NAME>-predictions" pour que l'expérience
    d'entraînement ne contienne que des runs d'entraînement.
    """
    n_total = len(results)
    n_flagged = sum(1 for r in results if r["flagged"])
    flag_rate = (n_flagged / n_total) if n_total else 0.0

    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(f"{settings.mlflow_experiment_name}-predictions")

        with mlflow.start_run(run_name="prediction_batch"):
            mlflow.set_tags(
                {
                    "run_type": "prediction_batch",
                    "has_flagged_predictions": "true" if n_flagged > 0 else "false",
                    "model_version": results[0]["model_version"] if results else "unknown",
                }
            )
            mlflow.log_metrics(
                {
                    "n_predictions": n_total,
                    "n_flagged": n_flagged,
                    "flag_rate": flag_rate,
                }
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Journalisation MLflow du batch impossible : %s", exc)


def _validate(records: list[dict]) -> list[dict[str, Any]]:
    """Valide chaque observation avec le schéma de l'API (CLI et DAG compris).

    Lève pydantic.ValidationError (sous-classe de ValueError) si designation est
    absente ou vide, ou si un champ inconnu est présent.
    """
    from src.api.schemas import ProductRecord

    return [ProductRecord.model_validate(record).model_dump() for record in records]


def _check_image(image_path: str | None) -> tuple[str | None, str | None]:
    """Return (usable absolute path, None) or (None, warning).

    Only existing, readable image files inside the raw data folder are used. A
    relative path is read relative to that folder. Anything else is reported, and
    the model then sees no image (the image transformer encodes it as zeros).
    """
    if image_path is None or not str(image_path).strip():
        return None, "no image_path: prediction uses text only"

    raw_root = RAW_DATA_DIR.resolve()
    candidate = Path(image_path)
    if not candidate.is_absolute():
        candidate = raw_root / candidate
    resolved = candidate.resolve()

    if not resolved.is_relative_to(raw_root):
        return None, (
            f"image_path outside the raw data folder: {image_path}; prediction uses text only"
        )
    if not resolved.is_file():
        return None, f"image_path not found: {image_path}; prediction uses text only"
    try:
        with Image.open(resolved) as image:
            image.verify()
    except Exception:  # noqa: BLE001
        return None, f"image_path unreadable: {image_path}; prediction uses text only"
    return str(resolved), None


def _known_categories(artifact: dict[str, Any]) -> set[int]:
    """Categories (prdtypecode) the served model can predict."""
    return {int(code) for code in artifact["pipeline"].classes_}


def _true_label(value: Any, known: set[int]) -> int | None:
    """Optional true category; when given it must be one the model knows."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value not in known:
        raise ValueError(f"unknown true_label {value!r}; expected one of {sorted(known)}")
    return value


def predict(
    records: list[dict], log_to_db: bool = True, log_to_mlflow: bool | None = None
) -> list[dict[str, Any]]:
    """Prédit pour une liste d'observations et journalise le résultat en base.

    Aucun run MLflow n'est créé ici (voir log_batch_to_mlflow pour la CLI).
    log_to_mlflow is deprecated and ignored; it is still accepted so existing
    callers keep working.
    """
    if log_to_mlflow is not None:
        py_warnings.warn(
            "predict(log_to_mlflow=...) is ignored: predict() no longer opens MLflow runs. "
            "Use log_batch_to_mlflow(results) for a batch summary.",
            DeprecationWarning,
            stacklevel=2,
        )
    if not records:
        raise ValueError("Aucune observation fournie")
    records = _validate(records)

    artifact = load_artifact()
    # Validated before the model runs, so a bad label never produces a stored row.
    known = _known_categories(artifact)
    true_labels = [_true_label(record.get("true_label"), known) for record in records]
    pipeline = artifact["pipeline"]
    feature_names: list[str] = artifact["feature_names"]

    # A missing or unusable image_path is never silently replaced by a blank image:
    # only checked paths reach the model, and each result says whether its image was used.
    model_rows, image_used, warnings_per_record = [], [], []
    for record in records:
        usable_path, image_warning = _check_image(record.get("image_path"))
        model_rows.append({**record, "image_path": usable_path})
        image_used.append(usable_path is not None)
        warnings_per_record.append([image_warning] if image_warning else [])

    X = prepare_frame(model_rows, feature_names)
    probabilities_matrix = pipeline.predict_proba(X)
    # Column i of predict_proba is the class pipeline.classes_[i].
    classes = [int(c) for c in pipeline.classes_]

    results = []
    for proba_row, used, warnings, true_label in zip(
        probabilities_matrix, image_used, warnings_per_record, true_labels, strict=True
    ):
        # The three most likely categories, best first. top3[0] is the prediction.
        ranked = sorted(range(len(classes)), key=lambda i: proba_row[i], reverse=True)[:3]
        top3 = [
            {
                "prediction": classes[i],
                "category_name": category_name(classes[i]),
                "probability": round(float(proba_row[i]), 6),
            }
            for i in ranked
        ]
        best = top3[0]
        flagged = best["probability"] < FLAG_THRESHOLD
        results.append(
            {
                "prediction": best["prediction"],
                "category_name": best["category_name"],
                "probability": best["probability"],
                "model_version": artifact["version"],
                "flagged": flagged,
                "flag_reason": "low_confidence" if flagged else None,
                "top3": top3,
                "image_used": used,
                "warnings": warnings,
                "true_label": true_label,
                "prediction_id": None,
            }
        )

    n_without_image = image_used.count(False)
    if n_without_image:
        logger.warning("%d/%d record(s) predicted without an image", n_without_image, len(results))

    n_flagged = sum(1 for r in results if r["flagged"])
    if n_flagged:
        logger.info("%d/%d prédiction(s) flaggée(s) (confiance < %.2f)", 
                    n_flagged, 
                    len(results), 
                    FLAG_THRESHOLD)

    if log_to_db:
        _store(records, results, artifact["version"])

    return results


def _store(records: list[dict], results: list[dict[str, Any]], version: str) -> None:
    """Write one predictions row per result and copy the row id into the result."""
    now = dt.now(UTC)
    try:
        with get_session() as session:
            rows = [
                Prediction(
                    model_version=version,
                    # The label is stored in its own column, never as a model input.
                    features={k: v for k, v in record.items() if k != "true_label"},
                    prediction=result["prediction"],
                    probability=result["probability"],
                    flagged=result["flagged"],
                    flag_reason=result["flag_reason"],
                    true_label=result["true_label"],
                    labeled_at=now if result["true_label"] is not None else None,
                )
                for record, result in zip(records, results, strict=True)
            ]
            session.add_all(rows)
            session.flush()  # assigns the ids
            for row, result in zip(rows, results, strict=True):
                result["prediction_id"] = row.id
    except Exception as exc:  # noqa: BLE001
        for result in results:
            result["prediction_id"] = None
        logger.warning("Journalisation des prédictions impossible : %s", exc)


def record_feedback(prediction_id: int, true_label: int) -> dict[str, Any]:
    """Set the true category of a stored prediction.

    Raises ValueError for a category the served model does not know and
    LookupError for an unknown prediction id.
    """
    true_label = _true_label(true_label, _known_categories(load_artifact()))
    with get_session() as session:
        row = session.get(Prediction, prediction_id)
        if row is None:
            raise LookupError(f"Unknown prediction_id: {prediction_id}")
        row.true_label = true_label
        row.labeled_at = dt.now(UTC)
        return {
            "prediction_id": row.id,
            "prediction": row.prediction,
            "true_label": row.true_label,
            "correct": row.prediction == row.true_label,
            "labeled_at": row.labeled_at,
        }


def sample_record() -> dict[str, Any]:
    """Construit une observation d'exemple à partir de la première ligne en base."""
    artifact = load_artifact()
    df = load_dataframe(limit=1)
    row = df[artifact["feature_names"]].iloc[0]
    # Pas de cast global en float : designation/description/image_path sont du texte.
    return row.where(pd.notna(row), None).to_dict()

def main() -> int:
    parser = argparse.ArgumentParser(description="Inférence du modèle baseline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Fichier JSON (objet ou liste d'objets)")
    group.add_argument("--sample", action="store_true", help="Utilise une ligne de la base")
    parser.add_argument("--no-log", action="store_true", help="Ne journalise pas en base")
    # Deprecated: predictions are no longer logged to MLflow by default, so there is
    # nothing to switch off. Still accepted so existing scripts keep working.
    parser.add_argument("--no-mlflow", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Journalise un résumé du batch dans l'expérience <MLFLOW_EXPERIMENT_NAME>-predictions",
    )
    args = parser.parse_args()

    setup_logging()
    if args.no_mlflow:
        logger.warning("--no-mlflow is deprecated and has no effect (use --mlflow to log).")

    if args.sample:
        records = [sample_record()]
    else:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else [payload]

    results = predict(records, log_to_db=not args.no_log)
    if args.mlflow:
        log_batch_to_mlflow(results)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
