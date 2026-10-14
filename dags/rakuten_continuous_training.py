"""DAG de versioning continu : entraîne le modèle Rakuten sur la prochaine
tranche de données toutes les 2 jours, via l'API FastAPI existante.

Quand toutes les lignes ont été consommées, /training renvoie le statut
"no_new_data" : le calcul de drift est alors sauté et le run reste vert, au lieu
d'échouer à chaque exécution une fois le jeu de données épuisé."""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import requests
from airflow.sdk import dag, task

TRAINING_URL = "http://api:8000/training"
DRIFT_URL = "http://api:8000/drift/compute"
# /training and /drift/compute require the API key (API_KEY in .env, passed by docker-compose).
HEADERS = {"X-API-Key": os.environ.get("API_KEY", "").strip()}


@dag(
    dag_id="rakuten_continuous_training",
    description="Versioning continu du modèle Rakuten (tranches de 3000 lignes) + calcul de drift",
    schedule=timedelta(days=2),
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["rakuten", "mlops", "continuous-training"],
)
def rakuten_continuous_training():
    @task
    def trigger_continuous_training():
        response = requests.post(
            TRAINING_URL, json={"continuous": True}, headers=HEADERS, timeout=1800
        )
        response.raise_for_status()
        result = response.json()
        if result.get("status") == "no_new_data":
            print(f"Pas de nouvelles données : {result.get('detail')}")
        else:
            print(f"Entraînement terminé : {result.get('comparison')}")
        return result

    @task.short_circuit
    def model_was_trained(training_result: dict) -> bool:
        """False (drift sauté, run vert) quand rien n'a été entraîné."""
        return training_result.get("status", "trained") == "trained"

    @task
    def trigger_drift_computation():
        response = requests.post(DRIFT_URL, headers=HEADERS, timeout=600)
        response.raise_for_status()
        result = response.json()
        print(f"Drift calculé : {len(result.get('metrics', []))} colonnes analysées")
        return result

    model_was_trained(trigger_continuous_training()) >> trigger_drift_computation()


rakuten_continuous_training()
