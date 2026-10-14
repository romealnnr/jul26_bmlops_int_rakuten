"""Training and evaluation of the Rakuten baseline model, with MLflow tracking."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.config import RAW_DATA_DIR, ensure_dirs, settings, setup_logging
from src.features.build_features import (
    build_preprocessor,
    get_feature_columns,
    load_dataframe,
    split_train_test,
    split_xy,
)

logger = logging.getLogger(__name__)

PRIMARY_METRIC = "weighted_f1"  # metric used to differentiate the models (cf. OBJECTIVES.md)


def build_pipeline(feature_names: list[str], random_state: int) -> Pipeline:
    """Pipeline Rakuten : prétraitement texte + numérique + image + régression logistique."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_names)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def evaluate(pipeline: Pipeline, X_test, y_test) -> dict[str, Any]:
    """Métriques d'évaluation multiclasse (Rakuten a ~27 classes, pas 2)."""
    y_pred = pipeline.predict(X_test)

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "weighted_f1": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "weighted_precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "weighted_recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }


def _dataset_fingerprint() -> str:
    """Fingerprint of the raw source files: tracks which version of the data produced
        which model (minimal data versioning, without a dedicated tool)."""
    hasher = hashlib.sha256()
    for filename in ("X_train_update.csv", "Y_train_CVw08PX.csv"):
        path = RAW_DATA_DIR / filename
        if path.exists():
            hasher.update(path.read_bytes())
    return hasher.hexdigest()[:12]


def _batches_for_rows(row_ids) -> list[int]:
    """Which data batches the rows used by this run came from.

    load_dataframe selects a contiguous range of ids, so the range is enough to
    find their batches without sending tens of thousands of ids to the database.
    Rows ingested before lineage existed carry no batch and are left out, which
    is why an empty list is a meaningful answer rather than an error.
    """
    from sqlalchemy import select

    from src.data.db import RawSample, get_session

    low, high = int(row_ids.min()), int(row_ids.max())
    with get_session() as session:
        found = session.execute(
            select(RawSample.batch_id).where(RawSample.id.between(low, high)).distinct()
        ).all()
    return sorted({int(batch) for (batch,) in found if batch is not None})


def _data_version_tags(batch_ids: list[int]) -> dict[str, str]:
    """MLflow tags that make the data behind this run recoverable.

    A batch id is only meaningful inside this database, so the DVC hashes the
    batches were recorded with travel alongside it. Paired with git_commit they
    are an address rather than a label: `dvc pull` resolves them back to the
    exact bytes the model was trained on.

    Falls back to the locally computed fingerprint when no batch is found, so a
    run on pre-lineage rows still records something instead of nothing.
    """
    from sqlalchemy import select

    from src.data.db import DataBatch, get_session

    if not batch_ids:
        return {
            "data_versioning": "local_fingerprint",
            "data_fingerprint": _dataset_fingerprint(),
        }

    with get_session() as session:
        batches = session.execute(
            select(DataBatch.sources).where(DataBatch.id.in_(batch_ids))
        ).all()

    sources: dict[str, str] = {}
    for (batch_sources,) in batches:
        sources.update(batch_sources or {})

    tags = {
        "data_versioning": "dvc",
        "data_batches": ",".join(str(batch) for batch in batch_ids),
    }
    for name, md5 in sorted(sources.items()):
        tags[f"data_dvc_{name.replace('.', '_')}"] = str(md5)
    return tags


def _setup_mlflow() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


def _get_production_version(client: MlflowClient, model_name: str):
    """Returns (version, metrics) of the model currently in Production, or (None, None)."""
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
    except Exception:
        return None, None

    if not versions:
        return None, None

    version = versions[0]
    run = client.get_run(version.run_id)
    return version.version, run.data.metrics


def _score_champion_on(model_name: str, X_test, y_test) -> float | None:
    """Score the Production model on the test set the challenger was just judged on.

    This is the whole point of the change. Every training run splits its own
    batch and reports a metric measured on its own test rows, so comparing a
    challenger's number against the champion's stored number compares two
    exams sat on different papers.

    It matters most in exactly the situation the system is built to handle.
    When the data drifts, a model trained on the new distribution faces harder
    - or simply different - test rows, so it can score lower than a champion
    that was graded on older, easier ones, and be rejected for having adapted.
    The retraining loop would then spin without ever changing the served model.

    Returns None when the champion cannot be loaded or scored; the caller then
    falls back to the stored metric and says so, rather than silently promoting.
    """
    try:
        champion = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Champion model could not be loaded for comparison: %s", exc)
        return None

    try:
        y_pred = champion.predict(X_test)
    except Exception as exc:  # noqa: BLE001
        # A champion that cannot predict on current data is not a valid
        # comparison - but it is also a finding worth seeing in the logs.
        logger.warning("Champion model could not predict on the current test set: %s", exc)
        return None

    return round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)


def _compare_and_promote(
    client: MlflowClient,
    model_name: str,
    new_version: str,
    new_metrics: dict[str, Any],
    X_test,
    y_test,
) -> dict[str, Any]:
    """Compare challenger and champion on the SAME test set, then promote or reject.

    Both models are scored on the test rows of the batch that was just
    processed - that is, on current data. The question the registry answers
    becomes "which model handles today's data better?" instead of "who got a
    higher mark on their own paper?".

    The challenger was trained on rows from this same batch, so it does have
    an advantage here. That is intended: when the data has moved, the model
    that matches the new distribution is the one that should win, and that is
    precisely what this comparison rewards.

    ASSUMPTION, stated here on purpose. A production system would also hold
    the older data to account: a model that handles today's batch well but has
    forgotten everything before it is not obviously the better model, and the
    honest evaluation set would mix the recent rows with a sample of the
    history. We judge on the current batch alone, which keeps a training cycle
    at minutes instead of hours and suits a pipeline whose stated job is to
    follow the newest slice. It is a deliberate simplification, not an
    oversight: in a setting where the past still matters, this test set is the
    first thing that should change.

    A 'best_model' tag is set on each version for quick reading without
    querying the stages."""
    prev_version, prev_metrics = _get_production_version(client, model_name)
    new_score = new_metrics[PRIMARY_METRIC]

    # What the champion reported when it was trained, kept only so the two
    # numbers can be read side by side. It is NOT what the decision uses.
    prev_logged_score = prev_metrics.get(PRIMARY_METRIC) if prev_metrics else None
    prev_score = None
    basis = "no_previous_model"

    if prev_version is not None:
        prev_score = _score_champion_on(model_name, X_test, y_test)
        if prev_score is not None:
            basis = "same_test_set"
        else:
            # Fall back rather than promote blindly, and record that we did.
            prev_score = prev_logged_score
            basis = "champion_logged_metric_fallback"

    if prev_version is None:
        decision, reason, is_best = "promoted", "No model in Production: first promotion.", True
    elif prev_score is None:
        decision = "promoted"
        reason = "Champion could not be evaluated and reported no stored metric."
        is_best = True
    elif new_score > prev_score:
        decision = "promoted"
        reason = (
            f"New {PRIMARY_METRIC}={new_score:.4f} > champion {prev_score:.4f} "
            f"on the same test set ({len(y_test)} current rows)."
        )
        is_best = True
    else:
        decision = "rejected"
        reason = (
            f"New {PRIMARY_METRIC}={new_score:.4f} <= champion {prev_score:.4f} "
            f"on the same test set ({len(y_test)} current rows)."
        )
        is_best = False

    client.set_model_version_tag(model_name, new_version, "best_model", "true" if is_best else "false")
    client.set_model_version_tag(model_name, new_version, PRIMARY_METRIC, str(new_score))
    client.set_model_version_tag(model_name, new_version, "comparison_basis", basis)

    # transition_model_version_stage is marked as “legacy” in recent MLflow versions
    # (replaced by aliases) but remains functional and widely used.
    if is_best:
        client.transition_model_version_stage(
            name=model_name, version=new_version, stage="Production", archive_existing_versions=True,
        )
    else:
        client.transition_model_version_stage(
            name=model_name, version=new_version, stage="Staging", archive_existing_versions=False,
        )

    logger.info("Comparaison MLflow : %s (version %s) — %s", decision, new_version, reason)

    return {
        "decision": decision,
        "reason": reason,
        "new_version": new_version,
        "new_score": new_score,
        "previous_version": prev_version,
        # Champion re-scored on this batch's test rows - the number that decided.
        "previous_score": prev_score,
        # Champion's own number from its own training run, for reference only.
        "previous_score_logged": prev_logged_score,
        "comparison_basis": basis,
        "evaluated_on_rows": int(len(y_test)),
        "metric": PRIMARY_METRIC,
    }



CONTINUOUS_PIPELINE_NAME = "rakuten-continuous-training"
CONTINUOUS_BATCH_SIZE = 3000


def _get_cursor(session, pipeline_name: str = CONTINUOUS_PIPELINE_NAME) -> int:
    """Retourne last_processed_id pour ce pipeline (0 si absent, ligne créée à la volée)."""
    from src.data.db import PipelineCursor

    cursor = session.query(PipelineCursor).filter_by(pipeline_name=pipeline_name).first()
    if cursor is None:
        cursor = PipelineCursor(pipeline_name=pipeline_name, last_processed_id=0)
        session.add(cursor)
        session.commit()
        return 0
    return cursor.last_processed_id


def _update_cursor(session, new_id: int, pipeline_name: str = CONTINUOUS_PIPELINE_NAME) -> None:
    """Updates last_processed_id after a successful training."""
    from datetime import datetime, timezone

    from src.data.db import PipelineCursor

    cursor = session.query(PipelineCursor).filter_by(pipeline_name=pipeline_name).first()
    if cursor is None:
        cursor = PipelineCursor(pipeline_name=pipeline_name)
        session.add(cursor)
    cursor.last_processed_id = new_id
    cursor.last_run_at = datetime.now(timezone.utc)
    session.commit()


def train(
    test_size: float | None = None,
    random_state: int | None = None,
    limit: int | None = None,
    continuous: bool = False,
) -> dict[str, Any]:
    """Trains the Rakuten model, logs it in MLflow (Tracking + Registry), 
    compares it to the current Production version, 
    and persists the local artifact (compatibility with predict.py)."""
    ensure_dirs()
    _setup_mlflow()

    test_size = test_size if test_size is not None else settings.test_size
    random_state = random_state if random_state is not None else settings.random_state

    min_id = None
    batch_max_id = None
    if continuous:
        from src.data.db import get_session

        with get_session() as session:
            min_id = _get_cursor(session)
        df = load_dataframe(limit=CONTINUOUS_BATCH_SIZE, min_id=min_id)
        batch_max_id = int(df["_id"].max())
        logger.info(
            "Continuous mode: batch (id > %d), %d loaded rows, max id = %d",
            min_id, len(df), batch_max_id,
        )
    else:
        df = load_dataframe(limit=limit)

    feature_names = get_feature_columns(df)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size, random_state)

    logger.info("Training on %d line (%d en test)", len(X_train), len(X_test))
    batch_ids = _batches_for_rows(df["_id"])

    with mlflow.start_run() as run:
        pipeline = build_pipeline(feature_names, random_state)
        pipeline.fit(X_train, y_train)

        metrics = evaluate(pipeline, X_test, y_test)
        version = datetime.now(UTC).strftime("%Y%m%d%H%M%S")

        mlflow.log_params({
            "model_name": settings.model_name,
            "test_size": test_size,
            "random_state": random_state,
            "limit": limit,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": len(feature_names),
            "classifier": "LogisticRegression",
        })
        mlflow.set_tags({
            "target_column": settings.target_column,
            "data_source": "postgresql:raw_samples",
            "git_commit": os.getenv("GIT_COMMIT", "unknown"),
            "version": version,
            **_data_version_tags(batch_ids),
        })
        mlflow.log_metrics({
            "accuracy": metrics["accuracy"],
            "weighted_f1": metrics["weighted_f1"],
            "weighted_precision": metrics["weighted_precision"],
            "weighted_recall": metrics["weighted_recall"],
        })
        
        if os.getenv("MLFLOW_LOG_ARTIFACTS", "true").lower() != "false":
             mlflow.log_dict(metrics["report"], "classification_report.json")
             mlflow.log_dict({"confusion_matrix": metrics["confusion_matrix"]}, "confusion_matrix.json")
        
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            serialization_format="cloudpickle"
        )
        run_id = run.info.run_id

    # Registration in the Model Registry (outside the with block, run already closed)
    registered = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=settings.mlflow_registered_model_name)
    new_model_version = registered.version

    client = MlflowClient()
    comparison = _compare_and_promote(
        client, settings.mlflow_registered_model_name, new_model_version, metrics, X_test, y_test,
    )
    new_model_version = str(new_model_version)

    if continuous and batch_max_id is not None:
        from src.data.db import get_session

        with get_session() as session:
            _update_cursor(session, batch_max_id)
        logger.info("Curseur mis à jour : last_processed_id = %d", batch_max_id)

    # NOTE : The model is no longer persisted locally as a .joblib file. predict.py
    # now always loads the “Production” version from the MLflow Model
    # Registry, which ensures that a model rejected by _compare_and_promote
    # (above) can never be served by mistake — before this fix, each training
    # run overwrote the local file, even when the model was rejected.
    trained_at = datetime.now(UTC).isoformat()
    n_train = int(len(X_train))
    n_test = int(len(X_test))

    summary = {
        "model_name": settings.model_name,
        "version": version,
        "trained_at": trained_at,
        "n_train": n_train,
        "n_test": n_test,
        "n_features": len(feature_names),
        "metrics": metrics,
        "mlflow_run_id": run_id,
        "mlflow_model_version": new_model_version,
        "data_batches": batch_ids,
        "comparison": comparison,
    }
    settings.metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Métriques enregistrées : %s", settings.metrics_path)
    logger.info(
        "accuracy=%.4f | weighted_f1=%.4f | décision=%s",
        metrics["accuracy"], metrics["weighted_f1"], comparison["decision"],
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Entraînement du modèle Rakuten")
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit to N rows (quick test)")
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Continuous versioning mode: processes the next 3000 rows from the pipeline_cursor",
    )
    args = parser.parse_args()

    limit = args.limit if args.limit else None  # 0 ou None → no limit

    setup_logging()
    train(test_size=args.test_size, random_state=args.random_state, limit=limit, continuous=args.continuous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())