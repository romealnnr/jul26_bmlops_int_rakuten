"""DAG de versioning continu : entraîne le modèle Rakuten sur la prochaine
tranche de données toutes les 2 jours, via l'API FastAPI existante."""

from __future__ import annotations

from datetime import datetime, timedelta

import requests
from airflow.sdk import dag, task

API_URL = "http://api:8000/training"


@dag(
    dag_id="rakuten_continuous_training",
    description="Versioning continu du modèle Rakuten (tranches de 3000 lignes)",
    schedule=timedelta(days=2),
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["rakuten", "mlops", "continuous-training"],
)
def rakuten_continuous_training():
    @task
    def trigger_continuous_training():
        response = requests.post(API_URL, json={"continuous": True}, timeout=1800)
        response.raise_for_status()
        result = response.json()
        print(f"Entraînement terminé : {result.get('comparison')}")
        return result

    trigger_continuous_training()


rakuten_continuous_training()
