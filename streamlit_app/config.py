import os
from pathlib import Path

RAW_DIR = Path(os.getenv("RAW_DIR", "/data/raw"))
# The raw data folder as seen from INSIDE the api container; image paths sent to
# /predict are built from this root (trailing "/" stripped to avoid "//").
API_IMAGE_ROOT = os.getenv("API_IMAGE_ROOT", "/app/src/data/raw").rstrip("/")
Y_TRAIN_FILE = "Y_train_CVw08PX.csv"
SPLITS = ("train", "test")

API_INTERNAL_URL = os.getenv("API_INTERNAL_URL", "http://api:8000")
PROMETHEUS_INTERNAL_URL = os.getenv("PROMETHEUS_INTERNAL_URL", "http://prometheus:9090")


def _configured_url(name: str, default: str = "") -> str:
    """Return a usable URL from the environment, or "" when unset or still a placeholder."""
    url = os.getenv(name, default).strip()
    # Template values such as http://dagshub.com/<user>/<repo> are not real links.
    if "<" in url or ">" in url:
        return ""
    return url


DASHBOARDS = {
    "Grafana": {
        "url": os.getenv("GRAFANA_URL", "http://localhost:3000"),
        "embed": True,
        "help": (
            "Dashboards are readable without a login (anonymous Viewer role); sign in as "
            "admin (password GRAFANA_ADMIN_PASSWORD) to edit. If the frame stays blank, "
            "use the button above."
        ),
    },
    "Prometheus": {"url": os.getenv("PROMETHEUS_URL", "http://localhost:9090"), "embed": True},
    "Airflow": {"url": os.getenv("AIRFLOW_URL", "http://localhost:8081"),  "embed": True},
    "FastAPI Docs": {"url": os.getenv("API_DOCS_URL", "http://localhost:8000/docs"), "embed": True},
    "DagsHub": {"url": _configured_url("DAGSHUB_URL"), "embed": False},
    "MLflow": {
        "url": os.getenv("MLFLOW_URL", "http://localhost:5000"),
        "embed": True,
        "help": "Experiments, runs and registered models (mlflow-ui service, port 5000).",
    },
}



PRDTYPECODE_LABELS = {
    10:   "Used books",
    40:   "Video games / imported CDs",
    50:   "Console accessories",
    60:   "Game consoles",
    1140: "Figurines / merchandise",
    1160: "Collectible cards",
    1180: "Figurines / role-playing games",
    1280: "Toys / plush toys",
    1281: "Children's board games",
    1300: "Scale models / drones",
    1301: "Baby clothing / children's accessories",
    1302: "Outdoor games / leisure",
    1320: "Baby and childcare products",
    1560: "Furniture / home interior",
    1920: "Household linen",
    1940: "Food",
    2060: "Decoration",
    2220: "Pet supplies",
    2280: "Used magazines",
    2403: "Books / comics / sheet music",
    2462: "Used video games",
    2522: "Stationery",
    2582: "Garden furniture",
    2583: "Pool / spa",
    2585: "Tools / gardening",
    2705: "New books",
    2905: "PC games",
}