"""Configuration centralisée : chemins du projet et variables d'environnement."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "src" / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
IMAGE_TRAIN_DIR = RAW_DATA_DIR / "image_train" / "image_train"
IMAGE_TEST_DIR = RAW_DATA_DIR / "image_test" / "image_test"


class Settings(BaseSettings):
    """Paramètres lus depuis .env (ou l'environnement du conteneur)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # PostgreSQL
    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str
    postgres_port: int

    # Modèle
    model_name: str
    target_column: str
    test_size: float
    random_state: int

    # API
    api_host: str
    api_port: int
    log_level: str
    
    # MLflow
    # mlflow_tracking_uri: str
    mlflow_experiment_name: str
    mlflow_registered_model_name: str
    
    @property
    def mlflow_tracking_uri(self) -> str:
        return self.database_url
    
    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def model_path(self) -> Path:
        return MODELS_DIR / f"{self.model_name}.joblib"

    @property
    def metrics_path(self) -> Path:
        return REPORTS_DIR / f"{self.model_name}_metrics.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def setup_logging() -> None:
    """Configuration de logs uniforme pour scripts et API."""
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_dirs() -> None:
    for directory in (RAW_DATA_DIR, MODELS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
