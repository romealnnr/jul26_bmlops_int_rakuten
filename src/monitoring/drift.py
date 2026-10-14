"""Computes the data drift between the very first data slice (fixed baseline) and 
a given batch, using Evidently. Results are stored in the database 
to be visualized in Grafana."""

from __future__ import annotations

import logging

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from sqlalchemy import select

from src.data.db import DriftMetric, RawSample, get_session

logger = logging.getLogger(__name__)

BASELINE_MAX_ID = 3000  # the very first slice serves as a fixed reference
DRIFT_WINDOW = 3000  # size of the “current” batch to compare


def _load_batch(session, min_id: int, max_id: int) -> pd.DataFrame:
    rows = session.execute(
        select(RawSample.features, RawSample.target)
        .where(RawSample.id > min_id, RawSample.id <= max_id)
    ).all()
    df = pd.DataFrame([{**features, "target": target} for features, target in rows])

    # Type coercion, consistent with split_xy() in build_features.py: without this,
    # Evidently may incorrectly infer a column’s type (text vs numeric) if the
    # types are inconsistent between reference and current (e.g., productid as str
    # in one batch and int in the other), which causes the text‑drift computation to fail.
    for col in ("designation", "description"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    for col in ("productid", "imageid"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def compute_drift(current_max_id: int) -> dict:
    """Compares the baseline (id ≤ BASELINE_MAX_ID) with the batch ending at
        current_max_id, and persists each column metric in the database."""
    current_min_id = max(current_max_id - DRIFT_WINDOW, BASELINE_MAX_ID)

    with get_session() as session:
        reference = _load_batch(session, 0, BASELINE_MAX_ID)
        current = _load_batch(session, current_min_id, current_max_id)

    if reference.empty or current.empty:
        raise RuntimeError("Pas assez de données pour calculer le drift.")

    # Columns that are irrelevant for statistical drift are removed
    # (file paths, technical identifiers with no business meaning).
    drop_cols = [c for c in ("image_path",) if c in reference.columns]
    reference = reference.drop(columns=drop_cols, errors="ignore")
    current = current.drop(columns=drop_cols, errors="ignore")

    report = Report([DataDriftPreset()])
    snapshot = report.run(current_data=current, reference_data=reference)
    result = snapshot.dict()

    reference_desc = f"id<=~{BASELINE_MAX_ID}"
    current_desc = f"id={current_min_id}-{current_max_id}"

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
            saved.append({"column": column_name, 
                          "method": method, 
                          "score": score, 
                          "threshold": threshold})
        session.commit()

    logger.info("Drift calculé (%s vs %s) : %d colonnes", reference_desc, current_desc, len(saved))
    return {"reference": reference_desc, "current": current_desc, "metrics": saved}
