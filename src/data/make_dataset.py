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

from src.config import RAW_DATA_DIR, IMAGE_TRAIN_DIR, ensure_dirs, settings, setup_logging
from src.data.db import RawSample, count_raw_samples, drop_db, get_session, init_db, ping

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


def ingest(rows: list[dict], chunk_size: int = 500) -> int:
    """Insère les lignes par lots."""
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
    parser.add_argument("--force", action="store_true", help="Vide les tables et réinsère les données")
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
    inserted = ingest(dataframe_to_rows(df))

    logger.info("Ingestion terminée : %d lignes dans raw_samples", inserted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())