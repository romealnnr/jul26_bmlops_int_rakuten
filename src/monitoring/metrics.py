"""Prometheus metrics for the ML system.

The FastAPI instrumentator already covers the HTTP layer: throughput,
latency, status codes. That answers "is the web service healthy?". This
module covers the model layer: what the API is serving, what it predicts,
and whether the training and drift loops are still alive.

These names are a contract between the code that sets them and the
dashboards and alert rules that read them. Renaming one breaks panels and
alerts silently - they simply show "No data" - so treat them as an API.

The `rakuten_` prefix separates them from the generic server metrics, so a
dashboard query can never pick one up by accident.

Two deliberate omissions:

  * No ratios. A flag *rate* is not stored; the flagged and total counters
    are, and the rate is computed in PromQL. A pre-divided ratio cannot be
    re-aggregated across instances or time windows afterwards.

  * No per-column drift scores. Those stay in the drift_metrics table, which
    is the right store for them: exact, auditable, and cheap to keep. Only
    the summary - how much drifted, and how old the report is - comes here,
    because those are the two things you alert on.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# --------------------------------------------------------------------------
# The served model
# --------------------------------------------------------------------------

MODEL_LOADED = Gauge(
    "rakuten_model_loaded",
    "1 when a model artifact is loaded and able to serve, 0 otherwise.",
)

MODEL_INFO = Gauge(
    "rakuten_model_info",
    "Identity of the model currently served. The value is always 1; the "
    "information is in the labels. This is the standard Prometheus way to "
    "expose strings, which cannot be metric values.",
    ["model_name", "version"],
)

MODEL_TRAINED_AT = Gauge(
    "rakuten_model_trained_at_timestamp_seconds",
    "When the served model was trained, as a Unix timestamp. Subtracting it "
    "from now() gives the model's age, which is what a dashboard should show.",
)

# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

PREDICTIONS = Counter(
    "rakuten_predictions_total",
    "Predictions served, by predicted class. The distribution over time is "
    "the cheapest early warning there is: when it shifts without the input "
    "volume changing, something upstream has changed.",
    ["predicted_class"],
)

PREDICTIONS_FLAGGED = Counter(
    "rakuten_predictions_flagged_total",
    "Predictions flagged as low confidence (below FLAG_THRESHOLD).",
)

PREDICTION_CONFIDENCE = Histogram(
    "rakuten_prediction_confidence",
    "Confidence of the winning class. A model that is drifting usually gets "
    "less sure before it gets wrong, and this is visible before any label "
    "arrives to prove it.",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

TRAINING_RUNS = Counter(
    "rakuten_training_runs_total",
    "Completed training runs, by outcome.",
    ["outcome"],  # succeeded | failed
)

TRAINING_DURATION = Histogram(
    "rakuten_training_duration_seconds",
    "Wall-clock duration of a training run.",
    buckets=(30, 60, 120, 300, 600, 1200, 1800, 3600),
)

TRAINING_IN_PROGRESS = Gauge(
    "rakuten_training_in_progress",
    "1 while a training run is executing, 0 otherwise. Also useful for "
    "reading the other graphs: latency rises during training because the "
    "same container is doing both.",
)

# --------------------------------------------------------------------------
# Drift (summary only - the detail lives in the drift_metrics table)
# --------------------------------------------------------------------------

DRIFT_SHARE = Gauge(
    "rakuten_drift_share_of_drifted_columns",
    "Share of monitored columns above their threshold in the latest drift "
    "report, between 0 and 1.",
)

DRIFT_COLUMNS_CHECKED = Gauge(
    "rakuten_drift_columns_checked",
    "Number of columns in the latest drift report. Needed to read the share: "
    "0.5 of two columns and 0.5 of forty are different statements.",
)

DRIFT_REPORT_AGE = Gauge(
    "rakuten_drift_report_age_seconds",
    "Age of the most recent drift report. This is the metric that watches "
    "the watcher: if the drift job dies, the drift panel keeps displaying "
    "its last values and looks healthy, while this number climbs. Silence "
    "and 'no drift' are indistinguishable without it.",
)


def _as_timestamp(value) -> float:
    """Unix timestamp from whatever trained_at happens to be.

    The artifact stores it as an ISO string today, but it has been a
    datetime before; accepting both costs three lines and avoids a metric
    that breaks on a change nobody thought was breaking.
    """
    from datetime import datetime, timezone

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        parsed = value

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def observe_predictions(results: list[dict]) -> None:
    """Record one batch of predictions.

    Takes the list predict() already builds, so the call site does not need
    to know anything about Prometheus.
    """
    for result in results:
        PREDICTIONS.labels(predicted_class=str(result["prediction"])).inc()
        PREDICTION_CONFIDENCE.observe(result["probability"])
        if result["flagged"]:
            PREDICTIONS_FLAGGED.inc()


def observe_model(info: dict | None) -> None:
    """Record which model is being served, or that none is.

    `info` is what model_info() returns, or None when no model is available.
    """
    if info is None:
        MODEL_LOADED.set(0)
        MODEL_INFO.clear()
        return

    MODEL_LOADED.set(1)

    # clear() first: without it, a reload leaves the previous version's label
    # set behind as a second series, and a dashboard showing "the current
    # model" would show two.
    MODEL_INFO.clear()
    MODEL_INFO.labels(
        model_name=str(info.get("model_name", "unknown")),
        version=str(info.get("version", "unknown")),
    ).set(1)

    trained_at = info.get("trained_at")
    if trained_at is not None:
        MODEL_TRAINED_AT.set(_as_timestamp(trained_at))


def observe_drift(metrics: list[dict]) -> None:
    """Record the summary of a drift report.

    `metrics` is the list compute_drift() returns: one dict per column with
    `score` and `threshold`.
    """
    DRIFT_COLUMNS_CHECKED.set(len(metrics))

    if not metrics:
        DRIFT_SHARE.set(0.0)
        return

    drifted = sum(
        1
        for m in metrics
        if m.get("threshold") is not None and m["score"] > m["threshold"]
    )
    DRIFT_SHARE.set(drifted / len(metrics))


def register_drift_age_collector() -> None:
    """Make DRIFT_REPORT_AGE answer from the database at scrape time.

    The age has to be right even when nothing is running - especially then,
    since a dead drift job is exactly what this measures. A background task
    that updates it would die with the thing it is watching, so instead the
    value is computed when Prometheus asks, straight from the table.

    Call once at application startup.
    """
    from datetime import datetime, timezone

    from sqlalchemy import func, select

    from src.data.db import DriftMetric, get_session

    def _age_seconds() -> float:
        try:
            with get_session() as session:
                latest = session.execute(
                    select(func.max(DriftMetric.run_at))
                ).scalar_one_or_none()
        except Exception:  # noqa: BLE001
            # A database that cannot be reached is its own alert elsewhere;
            # it must not take the whole /metrics endpoint down with it.
            return -1.0

        if latest is None:
            # No report has ever been produced. -1 rather than 0, so a panel
            # cannot read "brand new" when the truth is "never ran".
            return -1.0

        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - latest).total_seconds()

    DRIFT_REPORT_AGE.set_function(_age_seconds)
