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

Three deliberate choices:

  * No ratios. A flag *rate* is not stored; the flagged and total counters
    are, and the rate is computed in PromQL. A pre-divided ratio cannot be
    re-aggregated afterwards - which stops being academic the moment the API
    runs behind a load balancer, because averaging percentages taken over
    different request volumes gives the wrong number.

  * No per-column drift scores. Those stay in the drift_metrics table, which
    is the right store for them: exact, auditable, and cheap to keep. Only
    the summary - how much drifted, how many columns, and how old the report
    is - comes here, because those are the three things you alert on.

  * The drift summary is read from the database at scrape time, not set when
    a drift run finishes. Counters and in-process gauges belong to the
    process that incremented them; with several API replicas, only the one
    that happened to serve /drift/compute would know the answer and the rest
    would report zero forever. Reading the table means every replica gives
    the same answer, and the number of replicas stops mattering.
"""

from __future__ import annotations

import time
from datetime import UTC

from prometheus_client import Counter, Gauge, Histogram

# --------------------------------------------------------------------------
# The served model
# --------------------------------------------------------------------------

MODEL_LOADED = Gauge(
    "rakuten_model_loaded",
    "1 when a model artifact is loaded and able to serve, 0 otherwise. "
    "Per replica: aggregate with min() to ask whether ALL replicas can serve.",
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

LAST_TRAINING_FAILED = Gauge(
    "rakuten_last_training_failed",
    "1 when the most recent training run of this API process failed, 0 after "
    "a successful run (and at start-up). The training-failure alert reads this "
    "instead of a one-hour counter window: the DAG runs every 2 days, so a "
    "failure must stay visible until the next run succeeds, not for one hour. "
    "Per process: it starts at 0 after a restart.",
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
    "report, between 0 and 1. -1 when no report has ever been produced, -2 "
    "when the drift table could not be read.",
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
    "and 'no drift' are indistinguishable without it. -1 when no report has "
    "ever been produced, -2 when the drift table could not be read.",
)


def _as_timestamp(value) -> float:
    """Unix timestamp from whatever trained_at happens to be.

    The artifact stores it as an ISO string today, but it has been a
    datetime before; accepting both costs three lines and avoids a metric
    that breaks on a change nobody thought was breaking.
    """
    from datetime import datetime

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        parsed = value

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
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


def observe_training_outcome(succeeded: bool) -> None:
    """Record the outcome of one training run.

    Increments rakuten_training_runs_total{outcome} and sets
    rakuten_last_training_failed, so the counter (history) and the gauge
    (current state, used by the alert) can never disagree.
    """
    TRAINING_RUNS.labels(outcome="succeeded" if succeeded else "failed").inc()
    LAST_TRAINING_FAILED.set(0 if succeeded else 1)


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


# --------------------------------------------------------------------------
# Drift, read from the database at scrape time
# --------------------------------------------------------------------------

# Two sentinels, because two very different situations would otherwise look
# identical. 0 is not available for either: a panel must not read "brand new"
# or "no drift" when the truth is "never ran" or "cannot see the database".
NEVER = -1.0       # no drift report has ever been produced
UNREADABLE = -2.0  # the drift table could not be read

# Prometheus asks each gauge separately, so without this the three drift
# gauges would run three identical queries per scrape. One scrape interval is
# 15s; five seconds of cache collapses them into one query without ever
# serving a value from a previous scrape.
_CACHE_TTL_SECONDS = 5.0
_cache: tuple[float, tuple[float, float, float]] | None = None


def _latest_drift() -> tuple[float, float, float]:
    """(age_seconds, drifted_share, columns_checked) for the newest report.

    Returns (NEVER, NEVER, 0) when no report has ever been produced, and
    (UNREADABLE, UNREADABLE, 0) when the table cannot be read. Distinguishing
    them matters for the same reason this module exists: "nobody has measured
    drift yet" and "the monitoring cannot see its own data" call for different
    actions, and a single sentinel would make them indistinguishable - the
    failure mode these metrics are here to prevent.

    Either way the exception is swallowed. An unreachable database is its own
    alert elsewhere; what this function must never do is take the whole
    /metrics endpoint down with it.
    """
    from datetime import datetime

    from sqlalchemy import func, select

    from src.data.db import DriftMetric, get_session

    global _cache

    now = time.monotonic()
    if _cache is not None and now - _cache[0] < _CACHE_TTL_SECONDS:
        return _cache[1]

    result = (NEVER, NEVER, 0.0)
    try:  # noqa: SIM105
        with get_session() as session:
            # run_at comes from the database's now(), which in Postgres is
            # the transaction start time - so every row written by one drift
            # run shares the same value and identifies that run exactly.
            latest = session.execute(
                select(func.max(DriftMetric.run_at))
            ).scalar_one_or_none()

            if latest is not None:
                rows = session.execute(
                    select(DriftMetric.score, DriftMetric.threshold).where(
                        DriftMetric.run_at == latest
                    )
                ).all()

                if latest.tzinfo is None:
                    latest = latest.replace(tzinfo=UTC)
                age = (datetime.now(UTC) - latest).total_seconds()

                total = len(rows)
                drifted = sum(
                    1 for score, threshold in rows
                    if threshold is not None and score > threshold
                )
                share = (drifted / total) if total else 0.0
                result = (age, share, float(total))
    except Exception:  # noqa: BLE001
        result = (UNREADABLE, UNREADABLE, 0.0)

    _cache = (now, result)
    return result


def register_drift_collectors() -> None:
    """Make the three drift gauges answer from the database at scrape time.

    The age in particular has to be right even when nothing is running -
    especially then, since a dead drift job is exactly what it measures. A
    background task that updated it would die with the thing it is watching.

    Call once at application startup.
    """
    DRIFT_REPORT_AGE.set_function(lambda: _latest_drift()[0])
    DRIFT_SHARE.set_function(lambda: _latest_drift()[1])
    DRIFT_COLUMNS_CHECKED.set_function(lambda: _latest_drift()[2])
