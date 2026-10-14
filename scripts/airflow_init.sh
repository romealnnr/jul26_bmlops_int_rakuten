#!/usr/bin/env bash
# One-shot Airflow initialisation, run by the `airflow-init` compose service.
#
# 1. Creates the Airflow metadata database in Postgres if it is missing. This
#    works on a brand-new volume and on one that existed before; an init script
#    in docker/initdb (PR #12 adds one) may already have created it, which is fine.
#    Host, user, password and database name are read from
#    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN, so no extra .env key is needed.
# 2. Creates or upgrades the Airflow schema (`airflow db migrate`).
#
# Both steps are idempotent: running the script again changes nothing.
set -euo pipefail

: "${AIRFLOW__DATABASE__SQL_ALCHEMY_CONN:?AIRFLOW__DATABASE__SQL_ALCHEMY_CONN is not set}"

echo "[airflow-init] checking the Airflow metadata database"
python - <<'PY'
import os

import psycopg2
from psycopg2 import sql
from sqlalchemy.engine import make_url

url = make_url(os.environ["AIRFLOW__DATABASE__SQL_ALCHEMY_CONN"])
name = url.database
# Connect to the always-present "postgres" maintenance database, because the
# target database may not exist yet.
conn = psycopg2.connect(
    host=url.host,
    port=url.port or 5432,
    user=url.username,
    password=url.password,
    dbname="postgres",
)
conn.autocommit = True  # CREATE DATABASE cannot run inside a transaction
with conn.cursor() as cur:
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,))
    if cur.fetchone():
        print(f"[airflow-init] database '{name}' already exists")
    else:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        print(f"[airflow-init] database '{name}' created")
conn.close()
PY

echo "[airflow-init] airflow db migrate"
airflow db migrate

echo "[airflow-init] done"
