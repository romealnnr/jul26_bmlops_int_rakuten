"""Script one-shot : charge le jeu de données source et le stocke dans PostgreSQL.

Usage :
    python -m src.data.make_dataset
    python -m src.data.make_dataset --force
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
from sqlalchemy import insert

from src.config import IMAGE_TRAIN_DIR, RAW_DATA_DIR, ensure_dirs, setup_logging
from src.data.db import (
    DataBatch,
    RawSample,
    count_raw_samples,
    drop_db,
    get_session,
    init_db,
    ping,
)
from src.data.versioning import UNTRACKED, collect_sources, current_git_commit

logger = logging.getLogger(__name__)

SOURCE_NAME = "rakuten"


def load_source_dataframe() -> pd.DataFrame:
    """Récupère les features et la cible Rakuten (2 fichiers séparés) et les fusionne."""
    X = pd.read_csv(RAW_DATA_DIR / "X_train_update.csv", index_col=0)
    y = pd.read_csv(RAW_DATA_DIR / "Y_train_CVw08PX.csv", index_col=0)
    df = X.join(y)  # jointure sur l'index -> ajoute la colonne prdtypecode

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df["designation"] = df["designation"].fillna("")
    df["description"] = df["description"].fillna("")

    logger.info("Dataset source chargé : %d lignes, %d colonnes", *df.shape)
    return df


def save_raw_copy(df: pd.DataFrame) -> None:
    """Conserve une copie brute fusionnée sur disque (traçabilité)."""
    ensure_dirs()
    path = RAW_DATA_DIR / "rakuten_merged.csv"
    df.to_csv(path, index=False)
    logger.info("Copie brute écrite : %s", path)


def resolve_image_path(imageid, productid) -> str | None:
    """Reconstruit le chemin d'image attendu et vérifie qu'il existe sur disque."""
    if pd.isna(imageid):
        return None
    filename = f"image_{int(imageid)}_product_{int(productid)}.jpg"
    path = IMAGE_TRAIN_DIR / filename
    return str(path) if path.exists() else None


def dataframe_to_rows(df: pd.DataFrame) -> list[dict]:
    records = []
    n_with_image = 0
    for _, row in df.iterrows():
        image_path = resolve_image_path(row["imageid"], row["productid"])
        if image_path is not None:
            n_with_image += 1

        features = {
            "designation": row["designation"],
            "description": row["description"],
            "productid": int(row["productid"]),
            "imageid": int(row["imageid"]) if pd.notna(row["imageid"]) else None,
            "image_path": image_path,  # stocké dans le JSONB, pas de migration nécessaire
        }
        target = int(row["prdtypecode"])
        records.append({
            "features": features,
            "target": target,
            "source": SOURCE_NAME,
        })

    logger.info("Images trouvées et associées : %d/%d", n_with_image, len(records))
    if n_with_image == 0:
        logger.warning(
            "Aucune image trouvée dans %s — vérifie IMAGE_TRAIN_DIR dans src/config.py",
            IMAGE_TRAIN_DIR,
        )
    return records


def record_batch(n_rows: int) -> int:
    """Record which version of the raw files this ingestion is loading.

    Written before the rows, so that every row inserted afterwards can point at
    it. The hashes come from the DVC pointers, which makes the batch a
    resolvable address back to the exact bytes rather than just a timestamp.
    """
    sources = collect_sources()
    untracked = [name for name, value in sources.items() if value == UNTRACKED]
    if untracked:
        logger.warning(
            "No DVC pointer found for %s. This ingestion will be recorded, but those "
            "inputs will not be traceable back to a version. Run `dvc pull` first.",
            ", ".join(untracked),
        )

    with get_session() as session:
        batch = DataBatch(
            sources=sources,
            git_commit=current_git_commit(),
            n_rows=n_rows,
            note=f"ingestion of {SOURCE_NAME}",
        )
        session.add(batch)
        session.flush()  # assigns the primary key before the transaction closes
        batch_id = int(batch.id)

    logger.info("Lot de données enregistré : id=%d, %d lignes, sources=%s",
                batch_id, n_rows, sources)
    return batch_id


def ingest(rows: list[dict], batch_id: int | None = None, chunk_size: int = 500) -> int:
    """Insère les lignes par lots, en les rattachant à leur lot de données."""
    if batch_id is not None:
        for record in rows:
            record["batch_id"] = batch_id

    inserted = 0
    with get_session() as session:
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start : start + chunk_size]
            session.execute(insert(RawSample), chunk)
            inserted += len(chunk)
            logger.info("Insérées : %d/%d", inserted, len(rows))
    return inserted


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingestion one-shot vers PostgreSQL")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Vide les tables et réinsère les données",
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    args = parser.parse_args()

    setup_logging()

    if not ping():
        logger.error("Base inaccessible — démarrez-la avec : docker compose up -d db")
        return 1

    if args.force:
        drop_db()

    init_db()

    existing = count_raw_samples()
    if existing and not args.force:
        logger.info("%d lignes déjà présentes, rien à faire (--force pour réinitialiser)", existing)
        return 0

    df = load_source_dataframe()
    save_raw_copy(df)
    rows = dataframe_to_rows(df)
    batch_id = record_batch(len(rows))
    inserted = ingest(rows, batch_id=batch_id, chunk_size=args.chunk_size)

    logger.info("Ingestion terminée : %d lignes dans raw_samples (lot %d)", inserted, batch_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())