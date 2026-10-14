"""One-off: attach rows ingested before data lineage existed to a batch.

raw_samples was populated before data_batches existed, so those rows carry no
batch_id. They were loaded from the files the current .dvc pointers describe -
the md5 recorded in the pointer matches the file on disk - so we record one
batch for them and point the orphan rows at it.

The batch is marked as a backfill rather than as an ingestion, because that is
what it is: the link is asserted after the fact instead of recorded as it
happened. Anyone reading the table later should be able to tell the difference.

Safe to run repeatedly: it does nothing once no orphan rows are left.

Usage:
    python scripts/backfill_data_batches.py
"""

from __future__ import annotations

import logging

from sqlalchemy import func, select, update

from src.config import setup_logging
from src.data.db import DataBatch, RawSample, get_session
from src.data.versioning import collect_sources, current_git_commit

logger = logging.getLogger(__name__)

NOTE = "backfill: rows ingested before lineage tracking existed"


def main() -> int:
    setup_logging()

    with get_session() as session:
        orphans = session.scalar(
            select(func.count()).select_from(RawSample).where(RawSample.batch_id.is_(None))
        )
        orphans = int(orphans or 0)

        if not orphans:
            logger.info("Every row already belongs to a batch; nothing to do.")
            return 0

        batch = DataBatch(
            sources=collect_sources(),
            git_commit=current_git_commit(),
            n_rows=orphans,
            note=NOTE,
        )
        session.add(batch)
        session.flush()
        batch_id = int(batch.id)

        session.execute(
            update(RawSample).where(RawSample.batch_id.is_(None)).values(batch_id=batch_id)
        )

    logger.info("Attached %d rows to batch %d", orphans, batch_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())