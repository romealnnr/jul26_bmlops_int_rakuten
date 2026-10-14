"""Tests de la couche base de données."""

from __future__ import annotations

import pytest
from sqlalchemy import inspect, select, text

from src.data.db import Prediction, RawSample, engine, get_session, init_db, ping

# Les 27 codes produit Rakuten (prdtypecode) de Y_train_CVw08PX.csv.
# Toute cible ingérée doit être l'un d'eux.
# fmt: off
RAKUTEN_PRDTYPECODES = frozenset(
    {
        10, 40, 50, 60, 1140, 1160, 1180, 1280, 1281, 1300, 1301, 1302, 1320, 1560,
        1920, 1940, 2060, 2220, 2280, 2403, 2462, 2522, 2582, 2583, 2585, 2705, 2905,
    }
)
# fmt: on


@pytest.fixture(scope="module", autouse=True)
def database() -> None:
    """Ignore le module entier si PostgreSQL n'est pas joignable."""
    if not ping():
        pytest.skip("PostgreSQL indisponible", allow_module_level=True)
    init_db()


def test_connection_is_alive() -> None:
    with engine.connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar_one() == 1


def test_tables_exist() -> None:
    tables = set(inspect(engine).get_table_names())
    assert {"raw_samples", "predictions"} <= tables


def test_raw_samples_are_populated() -> None:
    with get_session() as session:
        sample = session.scalars(select(RawSample).limit(1)).first()
        targets = set(session.scalars(select(RawSample.target).distinct()).all())

    if sample is None:
        pytest.skip("Table vide — lancez : python -m src.data.make_dataset")

    assert isinstance(sample.features, dict)
    assert len(sample.features) > 0
    unexpected = targets - RAKUTEN_PRDTYPECODES
    assert not unexpected, f"cibles hors des 27 codes Rakuten : {sorted(unexpected)}"


def test_prediction_roundtrip() -> None:
    with get_session() as session:
        row = Prediction(
            model_version="test",
            features={"f1": 1.0},
            prediction=1,
            probability=0.9,
        )
        session.add(row)
        session.flush()
        created_id = row.id

    with get_session() as session:
        stored = session.get(Prediction, created_id)
        assert stored is not None
        assert stored.prediction == 1
        session.delete(stored)
