"""Couche base de données PostgreSQL : moteur, session et modèles ORM."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from src.config import settings

logger = logging.getLogger(__name__)

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    """Base déclarative commune à toutes les tables."""


class DataBatch(Base):
    """One ingestion of raw source files into raw_samples.

    PostgreSQL holds a working copy of the data; the files it was built from are
    versioned with DVC. This table is the link between the two, so that any row
    in raw_samples can be traced back to the exact bytes it came from.

    `sources` maps each input file to the md5 DVC recorded for it, and
    `git_commit` says which commit held those pointers. Together they are an
    address rather than a label: `dvc pull` resolves them back to the data.
    """

    __tablename__ = "data_batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sources: Mapped[dict] = mapped_column(JSONB, nullable=False)
    git_commit: Mapped[str | None] = mapped_column(String(40), nullable=True)
    n_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    note: Mapped[str | None] = mapped_column(String(200), nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<DataBatch id={self.id} rows={self.n_rows}>"


class RawSample(Base):
    """Une observation brute du jeu de données.

    Les variables explicatives sont stockées en JSONB : le schéma reste
    valable quel que soit le nombre de colonnes du dataset.
    """

    __tablename__ = "raw_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    features: Mapped[dict] = mapped_column(JSONB, nullable=False)
    target: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False, default="sklearn")
    # Nullable on purpose: rows ingested before data lineage existed have no
    # batch, and the column has to be addable to a populated table.
    batch_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("data_batches.id"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<RawSample id={self.id} target={self.target}>"


class Prediction(Base):
    """Journal des prédictions servies par l'API ou le CLI."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    features: Mapped[dict] = mapped_column(JSONB, nullable=False)
    prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    flagged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    flag_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<Prediction id={self.id} value={self.prediction}>"


class PipelineCursor(Base):
    """Position d'avancement du versioning continu (une ligne par pipeline)."""

    __tablename__ = "pipeline_cursor"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pipeline_name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    last_processed_id: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<PipelineCursor {self.pipeline_name}={self.last_processed_id}>"


class DriftMetric(Base):
    """Résultat d'un calcul de drift (Evidently) entre une baseline et un batch."""

    __tablename__ = "drift_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    reference_desc: Mapped[str] = mapped_column(String(200), nullable=False)
    current_desc: Mapped[str] = mapped_column(String(200), nullable=False)
    column_name: Mapped[str] = mapped_column(String(100), nullable=False)
    method: Mapped[str] = mapped_column(String(100), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)

    def __repr__(self) -> str:
        return f"<DriftMetric {self.column_name}={self.score}>"


def _ensure_batch_id_column() -> None:
    """Add raw_samples.batch_id to databases created before lineage existed.

    create_all() creates missing tables but never alters existing ones, and this
    project has no migration tool. This is the smallest honest substitute: one
    idempotent statement, run at startup, that costs nothing once the column is
    there. A longer-lived project would use Alembic instead.
    """
    statement = text(
        "ALTER TABLE raw_samples "
        "ADD COLUMN IF NOT EXISTS batch_id INTEGER REFERENCES data_batches(id)"
    )
    try:
        with engine.begin() as connection:
            connection.execute(statement)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not ensure raw_samples.batch_id exists: %s", exc)


def init_db() -> None:
    """Crée les tables manquantes (idempotent)."""
    Base.metadata.create_all(bind=engine)
    _ensure_batch_id_column()
    logger.info("Schéma vérifié : %s", ", ".join(Base.metadata.tables))


def drop_db() -> None:
    """Supprime toutes les tables du projet."""
    Base.metadata.drop_all(bind=engine)
    logger.warning("Tables supprimées")


@contextmanager
def get_session() -> Iterator[Session]:
    """Session transactionnelle : commit en sortie, rollback en cas d'erreur."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """Dépendance FastAPI."""
    with get_session() as session:
        yield session


def ping() -> bool:
    """Teste la connexion au serveur PostgreSQL."""
    try:
        with engine.connect() as conn:
            conn.execute(select(1))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Connexion PostgreSQL impossible : %s", exc)
        return False


def count_raw_samples() -> int:
    with get_session() as session:
        return session.scalar(select(func.count()).select_from(RawSample)) or 0
    