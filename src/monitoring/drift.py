"""Calcul du data drift entre la toute première tranche de données (baseline
fixe) et un batch donné, via Evidently. Résultats persistés en base pour
être visualisés dans Grafana."""

from __future__ import annotations

import logging

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from sqlalchemy import select

from src.data.db import DriftMetric, RawSample, get_session

logger = logging.getLogger(__name__)

BASELINE_MAX_ID = 3000  # la toute première tranche sert de référence fixe
DRIFT_WINDOW = 3000  # taille du batch "current" à comparer


def _load_batch(session, min_id: int, max_id: int) -> pd.DataFrame:
    rows = session.execute(
        select(RawSample.features, RawSample.target)
        .where(RawSample.id > min_id, RawSample.id <= max_id)
    ).all()
    return pd.DataFrame([{**features, "target": target} for features, target in rows])


def compute_drift(current_max_id: int) -> dict:
    """Compare la baseline (id <= BASELINE_MAX_ID) au batch se terminant à
    current_max_id, et persiste chaque métrique de colonne en base."""
    current_min_id = max(current_max_id - DRIFT_WINDOW, BASELINE_MAX_ID)

    with get_session() as session:
        reference = _load_batch(session, 0, BASELINE_MAX_ID)
        current = _load_batch(session, current_min_id, current_max_id)

    if reference.empty or current.empty:
        raise RuntimeError("Pas assez de données pour calculer le drift.")

    # Les colonnes non pertinentes pour le drift statistique sont retirées
    # (chemins de fichiers, identifiants techniques sans signification métier).
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
                continue  # on ne garde que le drift par colonne, pas l'agrégat global

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
