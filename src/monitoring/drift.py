"""Computes the data drift between the very first data slice (fixed baseline) and 
a given batch, using Evidently. Results are stored in the database 
to be visualized in Grafana."""

from __future__ import annotations

import logging

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from sqlalchemy import desc, select

from src.data.db import DriftMetric, Prediction, RawSample, get_session

logger = logging.getLogger(__name__)

BASELINE_MAX_ID = 3000  # the very first slice serves as a fixed reference
DRIFT_WINDOW = 3000  # size of the training "current" batch to compare
PREDICTION_WINDOW = 500  # number of most recent real /predict calls to compare

# Columns shared between raw_samples.features and predictions.features, usable
# for drift comparison. "target" is intentionally excluded: raw_samples has a
# true label, predictions only has a model prediction (no ground truth) — they
# are not the same thing and must never be compared as if they were.
FEATURE_COLUMNS = ("designation", "description", "productid", "imageid")


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    # Type coercion, consistent with split_xy() in build_features.py: without this,
    # Evidently may incorrectly infer a column's type (text vs numeric) if the
    # types are inconsistent between reference and current (e.g., productid as str
    # in one batch and int in the other), which causes the text-drift computation to fail.
    for col in ("designation", "description"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    for col in ("productid", "imageid"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _load_training_batch(session, min_id: int, max_id: int) -> pd.DataFrame:
    """Loads a slice of the training dataset (raw_samples), features + target."""
    rows = session.execute(
        select(RawSample.features, RawSample.target)
        .where(RawSample.id > min_id, RawSample.id <= max_id)
    ).all()
    df = pd.DataFrame([{**features, "target": target} for features, target in rows])
    return _coerce_types(df)


def _load_recent_predictions(session, limit: int) -> pd.DataFrame:
    """Loads the features of the most recent real /predict calls — i.e. what
    users actually send to the model in production, as opposed to the static
    training dataset. This is the only source that can reveal real-world drift."""
    rows = session.execute(
        select(Prediction.features)
        .order_by(desc(Prediction.id))
        .limit(limit)
    ).all()
    df = pd.DataFrame([features for (features,) in rows])
    return _coerce_types(df)


def compute_drift(current_max_id: int) -> dict:
    """Compares the baseline training data (id ≤ BASELINE_MAX_ID) with the most
    recent real prediction requests (table `predictions`), and persists each
    column metric in the database.

    current_max_id is kept as a parameter for logging/traceability (it reflects
    how far the continuous-training pipeline has progressed at the time this
    drift check runs), but it no longer determines which rows are compared:
    the "current" side is always the latest real /predict traffic, not another
    slice of the training set."""
    with get_session() as session:
        reference = _load_training_batch(session, 0, BASELINE_MAX_ID)
        current = _load_recent_predictions(session, PREDICTION_WINDOW)

    if reference.empty:
        raise RuntimeError("Pas assez de données d'entraînement pour calculer le drift.")
    if current.empty:
        raise RuntimeError(
            "Aucune prédiction réelle enregistrée dans la table `predictions` : "
            "impossible de calculer le drift tant que /predict n'a jamais été appelé."
        )

    # Only compare columns that exist on both sides (predictions has no "target").
    shared_cols = [c for c in FEATURE_COLUMNS if c in reference.columns and c in current.columns]
    reference = reference[shared_cols]
    current = current[shared_cols]

    report = Report([DataDriftPreset()])
    snapshot = report.run(current_data=current, reference_data=reference)
    result = snapshot.dict()

    reference_desc = f"training id<=~{BASELINE_MAX_ID}"
    current_desc = f"last {len(current)} real /predict calls (as of training id~{current_max_id})"

    saved = []
    with get_session() as session:
        for metric in result["metrics"]:
            config = metric.get("config", {})
            if config.get("type") != "evidently:metric_v2:ValueDrift":
                continue  # only the per‑column drift is kept, not the global aggregate

            column_name = config.get("column", "unknown")
            method = config.get("method", "unknown")
            threshold = config.get("threshold")
            score = metric.get("value")

            row = DriftMetric(
                reference_desc=reference_desc,
                current_desc=current_desc,
                column_name=column_name,
                method=method,
                score=float(score),
                threshold=float(threshold) if threshold is not None else None,
            )
            session.add(row)
            saved.append({"column": column_name, "method": method, "score": score, "threshold": threshold})
        session.commit()

    logger.info("Drift calculé (%s vs %s) : %d colonnes", reference_desc, current_desc, len(saved))
    return {"reference": reference_desc, "current": current_desc, "metrics": saved}
