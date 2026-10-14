"""Data provenance: read the DVC pointers that identify the raw data.

The raw files are versioned with DVC and stored on a remote; PostgreSQL only
ever holds a working copy of them. These helpers read the `.dvc` pointer files
so that an ingestion can record *which* version of the data it loaded.

A `.dvc` file is plain YAML holding the md5 of the data it points at, so we
parse it directly instead of importing DVC. That keeps DVC a development and CI
dependency: the API image has no reason to carry it, and what matters here is
the hash, not the tool that produced it. Paired with the git commit, that hash
is an address rather than a label, because `dvc pull` resolves it back to those
exact bytes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

from src.config import RAW_DATA_DIR

logger = logging.getLogger(__name__)

# The raw inputs of the ingestion, keyed by the name they are recorded under.
DATA_POINTERS: dict[str, Path] = {
    "X_train_update.csv": RAW_DATA_DIR / "X_train_update.csv.dvc",
    "Y_train_CVw08PX.csv": RAW_DATA_DIR / "Y_train_CVw08PX.csv.dvc",
    "image_train": RAW_DATA_DIR / "image_train" / "image_train.dvc",
}

UNTRACKED = "untracked"  # no usable pointer: this input came from outside DVC


def read_pointer(pointer: Path) -> str | None:
    """Return the md5 recorded in a .dvc file, or None if it cannot be read."""
    try:
        document = yaml.safe_load(pointer.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("DVC pointer %s could not be read: %s", pointer.name, exc)
        return None

    outs = document.get("outs") or []
    if not outs:
        logger.warning("DVC pointer %s has no outs entry", pointer.name)
        return None

    md5 = outs[0].get("md5")
    return str(md5) if md5 else None


def collect_sources() -> dict[str, str]:
    """Map each raw input to the md5 that DVC records for it.

    An input whose pointer is missing is reported as UNTRACKED rather than
    omitted. An ingestion that ran without DVC is a fact worth recording, and a
    silently shorter dictionary would hide it.
    """
    sources: dict[str, str] = {}
    for name, pointer in DATA_POINTERS.items():
        md5 = read_pointer(pointer) if pointer.exists() else None
        sources[name] = md5 if md5 is not None else UNTRACKED
    return sources


def current_git_commit() -> str | None:
    """The commit this code was built from, as the Docker image records it.

    The .dvc pointers live in git, so the commit is what makes a set of hashes
    reproducible: it says which pointers were in force when the data was read.
    Returns None when the image was built without the GIT_COMMIT build argument.
    """
    commit = os.getenv("GIT_COMMIT", "").strip()
    if not commit or commit == "unknown":
        return None
    return commit