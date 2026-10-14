"""DAG de versioning continu : entraîne le modèle Rakuten sur la prochaine
tranche de données toutes les 2 jours, via l'API FastAPI existante."""

from __future__ import annotations

from datetime import datetime, timedelta

import requests
from airflow.sdk import dag, task

TRAINING_URL = "http://api:8000/training"
DRIFT_URL = "http://api:8000/drift/compute"


@dag(
    dag_id="rakuten_continuous_training",
    description="Versioning continu du modèle Rakuten (tranches de 3000 lignes) + calcul de drift",
    schedule=timedelta(days=2),
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={
        # A transient API or database hiccup should not fail the whole run:
        # tasks are retried twice, the first time after about 5 minutes and
        # then with a growing delay (the training task overrides this with
        # retries=0). The training request waits up to 3 hours (below), so a
        # slow run is not cut off client-side.
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "retry_exponential_backoff": True,
        "execution_timeout": timedelta(hours=3),
    },
    tags=["rakuten", "mlops", "continuous-training"],
)
def rakuten_continuous_training():
    # No retry here: a retry while the API is still training would start a second
    # training run. A failed training request is retried at the next scheduled run.
    @task(retries=0)
    def trigger_continuous_training():
        response = requests.post(TRAINING_URL, json={"continuous": True}, timeout=3 * 3600)
        response.raise_for_status()
        result = response.json()
        print(f"Entraînement terminé : {result.get('comparison')}")
        return result

    @task
    def trigger_drift_computation(training_result: dict):
        response = requests.post(DRIFT_URL, timeout=600)
        response.raise_for_status()
        result = response.json()
        print(f"Drift calculé : {len(result.get('metrics', []))} colonnes analysées")
        return result

    trigger_drift_computation(trigger_continuous_training())


rakuten_continuous_training()
