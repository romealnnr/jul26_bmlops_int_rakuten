"""Training and evaluation of the Rakuten baseline model, with MLflow tracking.

The model registry uses aliases, not the legacy stages (deprecated in MLflow 3):
- @champion: the version the API serves (promoted);
- @challenger: the latest version that was trained but rejected.
"""
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
import numpy as np
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
from src.models.registry_errors import is_not_found

logger = logging.getLogger(__name__)

CHAMPION_ALIAS = "champion"  # registry alias of the model the API serves
CHALLENGER_ALIAS = "challenger"  # registry alias of the latest rejected model
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


def evaluate(pipeline: Pipeline, X_test, y_test, y_pred=None) -> dict[str, Any]:
    """Métriques d'évaluation multiclasse (Rakuten a ~27 classes, pas 2).
    y_pred: predictions already computed on X_test (reused by the promotion gate)."""
    if y_pred is None:
        y_pred = pipeline.predict(X_test)

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "weighted_f1": round(float(f1_score(y_test, 
                                            y_pred, 
                                            average="weighted", 
                                            zero_division=0)), 4),
        "weighted_precision": round(float(precision_score(y_test, 
                                                          y_pred, 
                                                          average="weighted", 
                                                          zero_division=0)), 4),
        "weighted_recall": round(float(recall_score(y_test, 
                                                    y_pred, 
                                                    average="weighted", 
                                                    zero_division=0)), 4),
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


def _setup_mlflow() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


def _get_champion_version(client: MlflowClient, model_name: str):
    """Return (version, metrics, params) of the @champion model, or (None, None, None).

    One-time migration: if no alias exists yet but a version is still in the old
    "Production" stage, the alias is set on that version. Stages are only read
    here, never written.

    Only "model or alias not found" means "no champion yet". Any other registry
    error is raised: reading it as "no champion" would promote the new model
    as if it were the first one."""
    try:
        version = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    except Exception as exc:
        if not is_not_found(exc):
            raise
        version = None

    if version is None:
        legacy = [
            v
            for v in client.search_model_versions(f"name='{model_name}'")
            if v.current_stage == "Production"
        ]
        if not legacy:
            return None, None, None
        version = max(legacy, key=lambda v: int(v.version))
        client.set_registered_model_alias(model_name, CHAMPION_ALIAS, version.version)
        logger.info("Migration: @%s set on Production version %s", CHAMPION_ALIAS, version.version)

    run = client.get_run(version.run_id)
    return str(version.version), run.data.metrics, run.data.params


# Promotion gate (point 21) -------------------------------------------------------
# One paired comparison: the @champion model and the new model
# (challenger) predict the SAME test rows, restricted to rows the champion never
# trained on (id above the champion's run param window_max_id). A paired bootstrap
# over those rows gives the weighted-F1 gain with a 95% interval. The challenger is
# promoted only if the lower bound is above 0 AND the gain is at least
# PROMOTION_MIN_GAIN, so a +0.0001 difference is no longer enough.
# Settings (environment, read at each training):
#   PROMOTION_MIN_GAIN        minimum point gain in weighted F1 (default 0.005)
#   PROMOTION_BOOTSTRAP       number of bootstrap resamples (default 1000)
#   PROMOTION_BOOTSTRAP_SEED  seed of the resampling (default 42)
#   PROMOTION_FIRST_MIN_F1    floor for the very first promotion (default 0.2)
#   PROMOTION_MIN_ROWS        fewest paired rows the decision may rest on (default 100)
DEFAULT_PROMOTION_MIN_GAIN = 0.005
DEFAULT_PROMOTION_BOOTSTRAP = 1000
DEFAULT_PROMOTION_BOOTSTRAP_SEED = 42
DEFAULT_PROMOTION_FIRST_MIN_F1 = 0.2
DEFAULT_PROMOTION_MIN_ROWS = 100
# A gain exactly equal to a threshold must not be lost to float rounding.
FLOAT_TOLERANCE = 1e-9


def _promotion_settings() -> dict[str, Any]:
    """Promotion gate settings from the environment (see the block comment above)."""
    config = {
        "min_gain": float(os.getenv("PROMOTION_MIN_GAIN", DEFAULT_PROMOTION_MIN_GAIN)),
        "n_bootstrap": int(os.getenv("PROMOTION_BOOTSTRAP", DEFAULT_PROMOTION_BOOTSTRAP)),
        "bootstrap_seed": int(
            os.getenv("PROMOTION_BOOTSTRAP_SEED", DEFAULT_PROMOTION_BOOTSTRAP_SEED)
        ),
        "first_min_f1": float(
            os.getenv("PROMOTION_FIRST_MIN_F1", DEFAULT_PROMOTION_FIRST_MIN_F1)
        ),
        "min_rows": int(os.getenv("PROMOTION_MIN_ROWS", DEFAULT_PROMOTION_MIN_ROWS)),
    }
    if min(config["min_gain"], config["first_min_f1"]) < 0:
        raise ValueError("PROMOTION_MIN_GAIN and PROMOTION_FIRST_MIN_F1 must be >= 0")
    if config["n_bootstrap"] < 100:
        raise ValueError("PROMOTION_BOOTSTRAP must be >= 100")
    if config["min_rows"] < 1:
        raise ValueError("PROMOTION_MIN_ROWS must be >= 1")
    return config


def _weighted_f1_rows(weights: np.ndarray, truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Weighted F1 (sklearn average="weighted", zero_division=0) for each row of weights.

    weights is (m, n): how many times each of the n test rows is drawn in each of m
    resamples; truth and pred are (n, k) one-hot matrices over the k labels."""
    support = weights @ truth
    predicted = weights @ pred
    true_pos = weights @ (truth * pred)
    denominator = support + predicted
    f1 = np.divide(
        2 * true_pos, denominator, out=np.zeros_like(true_pos), where=denominator > 0
    )
    return (f1 * support).sum(axis=1) / support.sum(axis=1)


def paired_bootstrap_gain(
    y_true, pred_challenger, pred_champion, n_resamples: int = DEFAULT_PROMOTION_BOOTSTRAP,
    seed: int = DEFAULT_PROMOTION_BOOTSTRAP_SEED, chunk: int = 100,
) -> dict[str, float]:
    """Weighted-F1 gain (challenger minus champion) on the same rows, with a paired
    bootstrap 95% percentile interval.

    Each resample draws n row indices with replacement and scores BOTH models on those
    same rows, so the interval reflects the paired difference. Vectorised: one matrix
    product per chunk of resamples."""
    y_true = np.asarray(y_true)
    pred_challenger = np.asarray(pred_challenger)
    pred_champion = np.asarray(pred_champion)
    n = len(y_true)
    if n == 0 or len(pred_challenger) != n or len(pred_champion) != n:
        raise ValueError("paired bootstrap needs non-empty, equal-length predictions")
    _, codes = np.unique(
        np.concatenate([y_true, pred_challenger, pred_champion]).astype(str),
        return_inverse=True,
    )
    eye = np.eye(int(codes.max()) + 1)
    truth, new, old = eye[codes[:n]], eye[codes[n:2 * n]], eye[codes[2 * n:]]

    ones = np.ones((1, n))
    challenger_f1 = float(_weighted_f1_rows(ones, truth, new)[0])
    champion_f1 = float(_weighted_f1_rows(ones, truth, old)[0])

    rng = np.random.default_rng(seed)
    gains = []
    for start in range(0, n_resamples, chunk):
        m = min(chunk, n_resamples - start)
        draws = rng.integers(0, n, size=(m, n))
        flat = (draws + (np.arange(m) * n)[:, None]).ravel()
        weights = np.bincount(flat, minlength=m * n).reshape(m, n).astype(float)
        gains.append(
            _weighted_f1_rows(weights, truth, new) - _weighted_f1_rows(weights, truth, old)
        )
    ci_low, ci_high = np.percentile(np.concatenate(gains), [2.5, 97.5])
    return {
        "challenger_f1": challenger_f1,
        "champion_f1": champion_f1,
        "gain": challenger_f1 - champion_f1,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_bootstrap": int(n_resamples),
        "n_rows": int(n),
    }


def decide_promotion(
    prev_version: str | None,
    new_score: float,
    paired: dict[str, float] | None,
    gate: dict[str, Any],
    no_pair_reason: str | None = None,
) -> tuple[str, str, bool]:
    """The promotion rule: (decision, reason, is_best).

    - No @champion model yet: promoted if the challenger's weighted F1 reaches
      PROMOTION_FIRST_MIN_F1, otherwise rejected.
    - No paired comparison possible (champion not scorable, its training window
      unknown, or too few rows it never trained on): rejected. Its logged metric comes
      from other rows and is not comparable, so it never decides.
    - Otherwise promoted iff the bootstrap lower bound of the gain is > 0 AND the point
      gain is >= PROMOTION_MIN_GAIN."""
    if prev_version is None:
        floor = gate["first_min_f1"]
        if new_score >= floor - FLOAT_TOLERANCE:
            return (
                "promoted",
                f"No @champion yet: first promotion ({PRIMARY_METRIC} "
                f"{new_score:.4f} >= floor {floor:.4f}).",
                True,
            )
        return (
            "rejected",
            f"No @champion yet, but {PRIMARY_METRIC} {new_score:.4f} is below the "
            f"first-promotion floor PROMOTION_FIRST_MIN_F1={floor:.4f}: not served.",
            False,
        )
    if paired is None:
        return (
            "rejected",
            f"No paired comparison with @champion version {prev_version}: "
            f"{no_pair_reason or 'champion not scorable'}. Its logged metric comes from "
            "other rows and is not comparable, so the challenger is not auto-promoted. "
            "Inspect, then promote by hand if wanted.",
            False,
        )
    gain, low, high = paired["gain"], paired["ci_low"], paired["ci_high"]
    detail = (
        f"{PRIMARY_METRIC} new {paired['challenger_f1']:.4f} vs champion "
        f"{paired['champion_f1']:.4f} on the same {paired['n_rows']} unseen test rows, "
        f"gain {gain:+.4f} (95% bootstrap interval {low:+.4f} to {high:+.4f}, "
        f"{paired['n_bootstrap']} resamples), minimum gain {gate['min_gain']:.4f}"
    )
    if low > 0 and gain >= gate["min_gain"] - FLOAT_TOLERANCE:
        return "promoted", f"Promoted: {detail}.", True
    why = "interval includes 0" if low <= 0 else "gain below the minimum"
    return "rejected", f"Rejected ({why}): {detail}.", False


def _score_champion_on(model_name: str, X_test) -> np.ndarray | None:
    """Per-row predictions of the @champion model on the rows the challenger is judged on.

    This is the whole point of the change. Every training run splits its own
    batch and reports a metric measured on its own test rows, so comparing a
    challenger's number against the champion's stored number compares two
    exams sat on different papers.

    It matters most in exactly the situation the system is built to handle.
    When the data drifts, a model trained on the new distribution faces harder
    - or simply different - test rows, so it can score lower than a champion
    that was graded on older, easier ones, and be rejected for having adapted.
    The retraining loop would then spin without ever changing the served model.

    Returns None when the champion cannot be loaded or cannot predict; the caller
    then rejects the challenger and says so (never promote blindly).
    """
    try:
        champion = mlflow.sklearn.load_model(f"models:/{model_name}@{CHAMPION_ALIAS}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Champion model could not be loaded for comparison: %s", exc)
        return None

    try:
        return np.asarray(champion.predict(X_test))
    except Exception as exc:  # noqa: BLE001
        # A champion that cannot predict on current data is not a valid
        # comparison - but it is also a finding worth seeing in the logs.
        logger.warning("Champion model could not predict on the current test set: %s", exc)
        return None


def _window_max_id(params: dict[str, str] | None) -> int | None:
    """Highest raw_samples id the champion trained on (run param window_max_id), or
    None when unknown (runs logged before this change)."""
    value = (params or {}).get("window_max_id")
    try:
        return int(value) if value not in (None, "", "None") else None
    except ValueError:
        return None


def _compare_and_promote(
    client: MlflowClient,
    model_name: str,
    new_version: str,
    new_metrics: dict[str, Any],
    X_test,
    y_test,
    y_pred,
    test_ids,
) -> dict[str, Any]:
    """Compare challenger and champion on the SAME unseen test rows, then promote or reject.

    Both models predict the test rows of the batch that was just processed - that
    is, current data - restricted to the rows whose id is above the champion's
    training window (run param window_max_id), so the champion is never graded on
    rows it trained on. The decision is the paired bootstrap rule of
    decide_promotion: a gain must be real (interval above 0) and large enough
    (PROMOTION_MIN_GAIN), not a lucky +0.0001.

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

    Promotion moves the registry aliases: @champion if promoted, @challenger if
    rejected (legacy stages are no longer written). A 'best_model' tag is also
    set on each version so the decision is visible in the MLflow UI."""
    gate = _promotion_settings()
    prev_version, prev_metrics, prev_params = _get_champion_version(client, model_name)
    new_score = new_metrics[PRIMARY_METRIC]

    # What the champion reported when it was trained, kept only so the two
    # numbers can be read side by side. It is NOT what the decision uses.
    prev_logged_score = prev_metrics.get(PRIMARY_METRIC) if prev_metrics else None
    paired = None
    no_pair_reason = None
    n_unseen = None
    basis = "no_previous_model"

    if prev_version is not None:
        window = _window_max_id(prev_params)
        if window is None:
            basis = "champion_window_unknown"
            no_pair_reason = (
                "its training window (run param window_max_id) is unknown, so the rows "
                "it trained on cannot be excluded"
            )
        else:
            unseen = np.asarray(test_ids) > window
            n_unseen = int(unseen.sum())
            if n_unseen < gate["min_rows"]:
                basis = "too_few_unseen_rows"
                no_pair_reason = (
                    f"only {n_unseen} test rows have an id above its training window "
                    f"(window_max_id={window}), PROMOTION_MIN_ROWS={gate['min_rows']} "
                    "needed. A champion trained on the full dataset has seen almost every "
                    "row: compare on new data (--continuous after new rows) or promote by hand"
                )
            else:
                champion_pred = _score_champion_on(model_name, X_test.loc[unseen])
                if champion_pred is None:
                    basis = "champion_not_scorable"
                    no_pair_reason = "it could not be loaded or could not predict"
                else:
                    basis = "paired_bootstrap_unseen_rows"
                    paired = paired_bootstrap_gain(
                        np.asarray(y_test)[unseen], np.asarray(y_pred)[unseen],
                        champion_pred, gate["n_bootstrap"], gate["bootstrap_seed"],
                    )

    decision, reason, is_best = decide_promotion(
        prev_version, new_score, paired, gate, no_pair_reason,
    )

    client.set_model_version_tag(model_name, 
                                 new_version, 
                                 "best_model", 
                                 "true" if is_best else "false")
    client.set_model_version_tag(model_name, new_version, PRIMARY_METRIC, str(new_score))
    client.set_model_version_tag(model_name, new_version, "comparison_basis", basis)
    client.set_model_version_tag(
        model_name, new_version, "promotion_min_gain", str(gate["min_gain"])
    )
    if paired is not None:
        for key in ("gain", "ci_low", "ci_high"):
            client.set_model_version_tag(
                model_name, new_version, f"promotion_{key}", f"{paired[key]:.6f}"
            )

    # Promotion moves the @champion alias; the API serves whatever @champion points to.
    alias = CHAMPION_ALIAS if is_best else CHALLENGER_ALIAS
    client.set_registered_model_alias(model_name, alias, new_version)

    logger.info("Comparaison MLflow : %s (version %s) — %s", decision, new_version, reason)

    return {
        "decision": decision,
        "reason": reason,
        "new_version": new_version,
        "new_score": new_score,
        "previous_version": prev_version,
        # Champion's weighted F1 on the paired unseen rows - the number that decided.
        "previous_score": paired["champion_f1"] if paired else None,
        # Champion's own number from its own training run, for reference only.
        "previous_score_logged": prev_logged_score,
        "comparison_basis": basis,
        "evaluated_on_rows": paired["n_rows"] if paired else n_unseen,
        "metric": PRIMARY_METRIC,
        "alias": alias,
        "gain": paired["gain"] if paired else None,
        "gain_ci_low": paired["ci_low"] if paired else None,
        "gain_ci_high": paired["ci_high"] if paired else None,
        "min_gain": gate["min_gain"],
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
    from datetime import datetime

    from src.data.db import PipelineCursor

    cursor = session.query(PipelineCursor).filter_by(pipeline_name=pipeline_name).first()
    if cursor is None:
        cursor = PipelineCursor(pipeline_name=pipeline_name)
        session.add(cursor)
    cursor.last_processed_id = new_id
    cursor.last_run_at = datetime.now(UTC)
    session.commit()


def _has_rows_after(session, cursor: int) -> bool:
    """Is there at least one raw_samples row after the cursor?"""
    from sqlalchemy import select

    from src.data.db import RawSample

    return session.scalar(select(RawSample.id).where(RawSample.id > cursor).limit(1)) is not None


def _no_new_data(cursor: int) -> dict[str, Any]:
    """Result of a continuous run when every row has already been consumed."""
    detail = (
        f"No new rows after id {cursor} in raw_samples: nothing to train on. "
        "The current @champion is kept."
    )
    logger.info(detail)
    return {"status": "no_new_data", "detail": detail, "cursor": cursor}


def train(
    test_size: float | None = None,
    random_state: int | None = None,
    limit: int | None = None,
    continuous: bool = False,
) -> dict[str, Any]:
    """Trains the Rakuten model, logs it in MLflow (Tracking + Registry), 
    compares it to the current @champion version
    and moves the registry aliases.

    In continuous mode, returns {"status": "no_new_data", ...} instead of raising
    once the cursor has reached the end of the data, so the DAG can skip cleanly."""
    ensure_dirs()
    # Read the promotion settings first, so an invalid PROMOTION_* value fails the
    # request before any training or registration happens.
    _promotion_settings()
    _setup_mlflow()

    test_size = test_size if test_size is not None else settings.test_size
    random_state = random_state if random_state is not None else settings.random_state

    min_id = None
    batch_max_id = None
    if continuous:
        from src.data.db import get_session

        with get_session() as session:
            min_id = _get_cursor(session)
            has_rows = _has_rows_after(session, min_id)
        if not has_rows:
            return _no_new_data(min_id)
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

    with mlflow.start_run() as run:
        pipeline = build_pipeline(feature_names, random_state)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        metrics = evaluate(pipeline, X_test, y_test, y_pred)
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
            # Highest raw_samples id this model trained on: the promotion gate only
            # compares a later challenger with it on rows above this id.
            "window_max_id": int(df["_id"].max()),
        })
        mlflow.set_tags({
            "target_column": settings.target_column,
            "data_source": "postgresql:raw_samples",
            "data_fingerprint": _dataset_fingerprint(),
            "git_commit": os.getenv("GIT_COMMIT", "unknown"),
            "version": version,
        })
        mlflow.log_metrics({
            "accuracy": metrics["accuracy"],
            "weighted_f1": metrics["weighted_f1"],
            "weighted_precision": metrics["weighted_precision"],
            "weighted_recall": metrics["weighted_recall"],
        })
        
        if os.getenv("MLFLOW_LOG_ARTIFACTS", "true").lower() != "false":
             mlflow.log_dict(metrics["report"], "classification_report.json")
             mlflow.log_dict({"confusion_matrix": metrics["confusion_matrix"]}, 
                             "confusion_matrix.json")
        
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            serialization_format="cloudpickle"
        )
        run_id = run.info.run_id

    # Registration in the Model Registry (outside the with block, run already closed)
    registered = mlflow.register_model(model_uri=f"runs:/{run_id}/model", 
                                       name=settings.mlflow_registered_model_name)
    new_model_version = registered.version

    client = MlflowClient()
    comparison = _compare_and_promote(
        client, settings.mlflow_registered_model_name, new_model_version, metrics,
        X_test, y_test, y_pred, df.loc[X_test.index, "_id"].to_numpy(),
    )
    new_model_version = str(new_model_version)

    if continuous and batch_max_id is not None:
        from src.data.db import get_session

        with get_session() as session:
            _update_cursor(session, batch_max_id)
        logger.info("Curseur mis à jour : last_processed_id = %d", batch_max_id)

    # NOTE : The model is no longer persisted locally as a .joblib file. predict.py
    # now always loads the @champion version from the MLflow Model
    # Registry, which ensures that a model rejected by _compare_and_promote
    # (above) can never be served by mistake — before this fix, each training
    # run overwrote the local file, even when the model was rejected.
    trained_at = datetime.now(UTC).isoformat()
    n_train = int(len(X_train))
    n_test = int(len(X_test))

    summary = {
        "model_name": settings.model_name,
        # Registry version number, the same number /model and the MLflow UI show.
        # The timestamp tag set on the run is kept as version_tag.
        "version": new_model_version,
        "version_tag": version,
        "trained_at": trained_at,
        "n_train": n_train,
        "n_test": n_test,
        "n_features": len(feature_names),
        "metrics": metrics,
        "mlflow_run_id": run_id,
        "mlflow_model_version": new_model_version,
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
    train(test_size=args.test_size, 
          random_state=args.random_state, 
          limit=limit, 
          continuous=args.continuous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())