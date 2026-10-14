"""Inférence à partir du modèle entraîné.

Usage :
    python -m src.models.predict --sample
    python -m src.models.predict --input payload.json
"""

from __future__ import annotations

import argparse
import json
import logging
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

logger = logging.getLogger(__name__)

_ARTIFACT: dict[str, Any] | None = None

FLAG_THRESHOLD = 0.6  # en dessous de ce score de confiance, la prédiction est flaggée


class ModelNotFoundError(RuntimeError):
    """Aucun modèle en Production n'est disponible dans le MLflow Model Registry."""


def load_artifact(force_reload: bool = False) -> dict[str, Any]:
    """Charge le modèle actuellement en Production depuis le MLflow Model Registry,
    avec cache mémoire. Contrairement à un fichier local, ceci garantit que l'API
    sert toujours exactement le modèle que MLflow considère comme Production —
    jamais un modèle rejeté par _compare_and_promote lors d'un entraînement."""
    global _ARTIFACT

    if _ARTIFACT is not None and not force_reload:
        return _ARTIFACT

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    versions = client.get_latest_versions(settings.mlflow_registered_model_name,
                                          stages=["Production"])
    if not versions:
        raise ModelNotFoundError(
            f"Aucun modèle en Production pour '{settings.mlflow_registered_model_name}'. "
            "Lancez d'abord : python -m src.models.training"
        )

    version_info = versions[0]
    run = client.get_run(version_info.run_id)
    model_uri = f"models:/{settings.mlflow_registered_model_name}/Production"
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


def _log_batch_to_mlflow(results: list[dict[str, Any]]) -> None:
    """Journalise un résumé du batch de prédictions dans MLflow (monitoring)."""
    n_total = len(results)
    n_flagged = sum(1 for r in results if r["flagged"])
    flag_rate = (n_flagged / n_total) if n_total else 0.0

    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

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


def predict(
    records: list[dict], log_to_db: bool = True, log_to_mlflow: bool = True
) -> list[dict[str, Any]]:
    """Prédit pour une liste d'observations et journalise le résultat."""
    if not records:
        raise ValueError("Aucune observation fournie")
    records = _validate(records)

    artifact = load_artifact()
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
    for proba_row, used, warnings in zip(
        probabilities_matrix, image_used, warnings_per_record, strict=True
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
        try:
            with get_session() as session:
                session.add_all(
                    [
                        Prediction(
                            model_version=artifact["version"],
                            features=record,
                            prediction=result["prediction"],
                            probability=result["probability"],
                            flagged=result["flagged"],
                            flag_reason=result["flag_reason"],
                        )
                        for record, result in zip(records, results, strict=True)
                    ]
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Journalisation des prédictions impossible : %s", exc)

    if log_to_mlflow:
        _log_batch_to_mlflow(results)

    return results

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
    parser.add_argument("--no-mlflow", action="store_true", help="Ne journalise pas dans MLflow")
    args = parser.parse_args()

    setup_logging()

    if args.sample:
        records = [sample_record()]
    else:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else [payload]

    results = predict(records, log_to_db=not args.no_log, log_to_mlflow=not args.no_mlflow)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
