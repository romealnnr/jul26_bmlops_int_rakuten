# Monitoring regression checks

Owner: Ben (monitoring workstream)

## What this is for

The monitoring stack is not self-announcing: when it breaks, it does not throw an
error, it just shows a flat line or a zero. A dashboard that says `0` looks exactly
like a dashboard that says "nothing is wrong". That is the whole reason this file
exists.

Run the **fast suite** (layers 0-4, about three minutes) after every merge into
`master`, and before pushing anything of your own that touches `src/api/main.py`,
`src/models/`, `src/monitoring/`, the `Dockerfile`, `docker-compose*.yml` or anything
under `monitoring/`.

Run the **full suite** (layers 0-7, about twenty minutes, because it trains a model)
before the defence, and after any change to training, promotion or the alert path.

Every command below was executed against a live stack. The "expected" lines are
observed output, not guesses.

## How to start the stack

The monitoring overlay is a second compose file. Both are needed, in this order:

```
docker compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
```

If you start with `docker compose up -d` alone, Prometheus comes up without its
retention and lifecycle flags and Grafana comes up without the Postgres credentials
it needs for the drift datasource. Everything will *look* fine. Layer 3 is what
catches it.

Shell: the commands are written for Windows `cmd`. On bash, replace `findstr /c:"x"`
with `grep -F "x"` and `findstr /i "x y"` with `grep -iE "x|y"`.

---

# FAST SUITE

## Layer 0 — the stack is standing

```
docker compose ps
```

Expected: `api`, `grafana`, `prometheus`, `postgres`, `mlflow` all `Up`. A container
in `Restarting` is a failure, not a transient state — read its logs before going on.

```
curl -s http://127.0.0.1:8000/health
```

Expected: a JSON body reporting the service as healthy.

**If `api` is unhealthy or restarting:** check `docker compose logs api --tail 50`
first. The two failure modes we have actually hit are a permissions error on
`/home/appuser/.cache/torch` (fixed in the `Dockerfile`; if it returns, somebody
reverted that chunk) and a model artifact that cannot be loaded from the Registry.

## Layer 1 — the metric contract exists

This is the one that matters most, because the contract is what every dashboard and
every alert rule is written against. If a metric disappears, nothing errors: the
panels go empty and the alerts silently stop firing.

```
curl -s http://127.0.0.1:8000/metrics > metrics_check.txt
findstr /c:"rakuten_" metrics_check.txt
```

Expected: all ten metric families present.

| Metric | Type | Notes |
|---|---|---|
| `rakuten_model_loaded` | gauge | 1 when a model is serving |
| `rakuten_model_info` | gauge | labels `model_name`, `version` |
| `rakuten_model_trained_at_timestamp_seconds` | gauge | |
| `rakuten_predictions_total` | counter | label `predicted_class` |
| `rakuten_predictions_flagged_total` | counter | low-confidence predictions |
| `rakuten_prediction_confidence` | histogram | |
| `rakuten_training_runs_total` | counter | label `outcome` = succeeded / failed |
| `rakuten_last_training_failed` | gauge | 1 after a failed run until the next success; the training-failure alert reads this |
| `rakuten_training_duration_seconds` | histogram | |
| `rakuten_training_in_progress` | gauge | the webhook lock reads this |
| `rakuten_drift_share_of_drifted_columns` | gauge | read from the DB at scrape time |
| `rakuten_drift_columns_checked` | gauge | |
| `rakuten_drift_report_age_seconds` | gauge | |

Then clean up: `del metrics_check.txt`

**The contract rule:** these names are referenced from `rules.yml` and from three
dashboard JSON files. Renaming one, or dropping a label, breaks all of them at once
and breaks them *quietly*. If a metric has to change, the dashboards and the rules
change in the same commit.

## Layer 2 — Prometheus is actually scraping

A metric that exists on `/metrics` but is not being scraped is invisible to every
alert.

```
curl -s http://127.0.0.1:9090/api/v1/targets
```

Expected: the API target with `"health":"up"`. A target in `"down"` with a
connection-refused error usually means the API container restarted and Prometheus has
not retried yet; wait one scrape interval and re-check.

```
curl -s "http://127.0.0.1:9090/api/v1/query?query=rakuten_model_loaded"
```

Expected: a result array with `"value"` containing `"1"`. An **empty** result array is
the failure signal: Prometheus knows the target but has no such series, meaning layer
1 and layer 2 disagree.

## Layer 3 — Grafana provisioning loaded

Four things are provisioned from files. All four must report
`"provenance":"file"` — that is what proves Grafana read *our* files and not a stale
copy in its own database.

```
curl -s -u admin:admin http://127.0.0.1:3000/api/v1/provisioning/alert-rules > rules_check.json
findstr /c:"rakuten-" rules_check.json
```

Expected: four rules — `rakuten-model-not-loaded`, `rakuten-drift-above-threshold`,
`rakuten-drift-monitor-stale`, `rakuten-training-failures` — all with the same
`folderUID` and all with `"provenance":"file"`. Check the `"updated"` timestamp: it
must be from the current container start, otherwise you are looking at rules Grafana
kept from an earlier run while this run's provisioning failed.

```
curl -s -u admin:admin http://127.0.0.1:3000/api/v1/provisioning/contact-points
```

Expected: `rakuten-retrain`, type `webhook`, `url` =
`http://api:8000/training/trigger`, `httpMethod` `POST`, `"provenance":"file"`. The
other entry named `email receiver` with an empty uid is Grafana's own default row;
leave it alone.

```
curl -s -u admin:admin http://127.0.0.1:3000/api/v1/provisioning/policies
```

Expected: root `"receiver":"grafana-default-email"`, one child route to
`rakuten-retrain` matching `["alertname","=","Data drift above threshold"]`,
`"provenance":"file"`.

**An empty response here is the alarm.** Provisioning alerting is all-or-nothing: if
the policy tree is rejected, Grafana aborts the entire alerting block and the four
rules and the contact point stop being refreshed too. The cause is always visible in
the log:

```
docker compose logs grafana --tail 40 | findstr /i "provision alerting"
```

Expected: `starting to provision alerting` followed by `finished to provision
alerting`, with no `level=error`.

Two traps we have paid for:

- The policy's **root** receiver is validated against the *provisioned* contact points
  plus Grafana's reserved `grafana-default-email`. Pointing it at the database row
  named `email receiver` fails with `receiver 'email receiver' does not exist`, even
  though that name appears in the contact-points API.
- The root receiver must stay a harmless sink. It receives *every* alert, so if the
  root were `rakuten-retrain`, then "API is serving without a model" would also
  trigger a retrain. The filtering belongs in the child route.

Then clean up: `del rules_check.json`

```
curl -s -u admin:admin http://127.0.0.1:3000/api/v1/datasources
```

Expected: the Prometheus datasource with uid `prometheus_ds` — the rules reference
that uid literally, so renaming it breaks all four — and the Postgres datasource with
`"user":"rakuten"` and a non-empty `"database"`. If those two come back empty, the
monitoring overlay was not applied (see "How to start the stack") and the Data Drift
dashboard will render with no rows.

## Layer 4 — the drift path

Drift is not pushed into Prometheus by anything. The three drift gauges are
`set_function` collectors that read the `drift_metrics` table **at scrape time**, with
a five-second cache. That design is deliberate: it is what lets several API replicas
report the same drift figure instead of each holding a private copy.

```
curl -s http://127.0.0.1:8000/metrics | findstr /c:"rakuten_drift"
```

Expected: a `rakuten_drift_report_age_seconds` in the thousands of seconds (its real
age), `rakuten_drift_share_of_drifted_columns` between 0 and 1, and
`rakuten_drift_columns_checked` matching the number of monitored columns.

Two sentinel values are not errors, they are answers:

- `-1` — no drift report has ever been produced. Honest "never run".
- `-2` — the drift table could not be read. The database is down or the
  credentials changed.

A `0` share with a plausible age is a healthy "no drift detected". A `0` share with an
age of `-1` means the monitor has never run, which is a completely different
situation; the dashboard tile maps both sentinels to text for exactly that reason.

**Optional negative test** (proves the path is live rather than cached):
`docker compose stop postgres`, scrape again and expect `-2`, then
`docker compose start postgres` and expect the real values back within the cache TTL.

---

# FULL SUITE

Everything above, plus:

## Layer 5 — the webhook, end to end

This proves the brief's requirement: drift alert → Grafana → retraining.

The lock first. Send two POSTs back to back, with no pause:

```
curl -s -X POST http://127.0.0.1:8000/training/trigger
curl -s -X POST http://127.0.0.1:8000/training/trigger
```

Expected, in this exact order:

```
{"status":"accepted","mode":"continuous"}
{"status":"skipped","reason":"a training run is already in progress"}
```

The second response is the point of the test. Grafana re-sends a firing alert on its
`repeat_interval`, so without the lock a drift condition that stays true would stack
retraining runs on top of each other. `_training_in_progress()` reads the
`rakuten_training_in_progress` gauge, which `_train_and_record` sets to 1 before
training and clears to 0 in a `finally` block, so the lock also releases when training
*fails*.

While it runs:

```
curl -s http://127.0.0.1:8000/metrics | findstr /c:"rakuten_training_in_progress"
```

Expected `1.0` during the run, `0.0` after it finishes.

And after:

```
curl -s http://127.0.0.1:8000/metrics | findstr /c:"rakuten_training_runs_total"
```

Expected: `outcome="succeeded"` incremented by one and `outcome="failed"` unchanged.
`rakuten_last_training_failed` (same command, that name) reads `0.0` after a successful
run and `1.0` after a failed one, until the next run succeeds.

**Note on the endpoint:** the webhook posts to `/training/trigger`, not to `/training`.
That separation is not cosmetic. `/training` takes a request body, and Grafana's
webhook payload contains none of those fields; pydantic ignores unknown fields, so
Grafana posting to `/training` would read as all-defaults and silently launch a *full*
retrain instead of a continuous one. `/training/trigger` takes no body, answers
immediately, and does the work in a background task so Grafana's webhook does not
time out.

## Layer 6 — the promotion gate

The gate decides whether a freshly trained model replaces the one in production. It
must compare the challenger and the champion **on the same test set**.

```
docker compose logs api --tail 200 | findstr /i "comparison"
```

Expected: a `comparison_basis` of `same_test_set`, with both `previous_score` (the
champion re-scored on the current batch) and `previous_score_logged` (what the
champion recorded at its own training time) present and usually different.

`comparison_basis` has three legal values:

- `same_test_set` — the healthy case.
- `champion_logged_metric_fallback` — the champion could not be loaded or could not
  predict, so the gate fell back to its logged metric. Degraded, but explicit.
- `no_previous_model` — first ever training. Promotes unconditionally.

**Why this matters, and it is worth being able to say it out loud:** the original gate
compared the challenger's score on the *current* batch against the champion's score as
logged on the *batch the champion was trained on*. Those are two different test sets,
so the comparison was meaningless, and because evaluation batches vary in difficulty
it rejected good models and could have accepted bad ones. We measured it: challenger
`0.6316` against the champion's logged `0.6848` would have been rejected, while
re-scoring the champion on the same batch gave `0.6218` — the challenger was in fact
better and is now correctly promoted.

**Stated assumption:** we judge a model on the current batch alone. A production
system would evaluate on a held-out set that mixes in a sample of the full history, so
that a model is never promoted for being good at only the newest data. We say so
explicitly in the docstring of `_compare_and_promote` rather than pretending
otherwise.

## Layer 7 — CI and packaging

```
ruff check src
```

Expected: `All checks passed!`. This is the CI gate (`.github/workflows/python-app.yml`,
the Lint step); a failure here fails the build for everybody, not just for you.

Then confirm `src.monitoring` is listed in the `packages` table in `pyproject.toml`.
If it is missing, the module works from a source checkout and from the Docker image
but is absent from an installed wheel, so every metric silently disappears in any
install-based deployment.

---

## Quick triage

| Symptom | Most likely cause | Layer |
|---|---|---|
| Dashboard panel empty | metric renamed or label dropped | 1 |
| Panel empty but `/metrics` has the series | Prometheus target down, or `prometheus_ds` uid renamed | 2, 3 |
| Alert rules missing from the UI | alerting provisioning aborted on the policy tree | 3 |
| Data Drift dashboard shows no rows | monitoring overlay not applied, Postgres datasource credentials empty | 3 |
| Drift age is `-1` | the drift job has never run | 4 |
| Drift values are `-2` | Postgres unreachable from the API | 4 |
| Retrain never happens on drift | contact point or child route missing | 3, 5 |
| Retrains stacking up | the in-progress gauge is not being set | 5 |
| Good models being rejected | promotion gate comparing across different test sets | 6 |
