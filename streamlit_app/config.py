import os

API_INTERNAL_URL = os.getenv("API_INTERNAL_URL", "http://api:8000")
PROMETHEUS_INTERNAL_URL = os.getenv("PROMETHEUS_INTERNAL_URL", "http://prometheus:9090")

DASHBOARDS = {
    "Grafana": {"url": os.getenv("GRAFANA_URL", "http://localhost:3000"), "embed":True},
    "Prometheus": {"url": os.getenv("PROMETHEUS_URL", "http://localhost:9090"), "embed": True},
    "Airflow": {"url": os.getenv("AIRFLOW_URL", "http://localhost:8081"),  "embed": True},
    "FastAPI Docs": {"url": os.getenv("API_DOCS_URL", "http://localhost:8000/docs"), "embed": True},
    "DagsHub": {"url": os.getenv("DAGSHUB_URL", ""), "embed": False},
}


FLAG_THRESHOLD = float(os.getenv("FLAG_THRESHOLD", "0.6"))