#!/usr/bin/env bash
# Starts the Airflow api-server (web UI on http://localhost:8081) with a fixed
# login taken from .env, instead of a random password printed in the log.
#
# Airflow 3 uses SimpleAuthManager: the user list comes from
# AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_USERS and the passwords from the JSON file
# AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_PASSWORDS_FILE (both set in docker-compose.yml).
# This script writes that file from AIRFLOW_ADMIN_USER / AIRFLOW_ADMIN_PASSWORD
# on every start, so the login always follows the current .env values.
set -euo pipefail

: "${AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_PASSWORDS_FILE:?passwords file path is not set}"
export AIRFLOW_ADMIN_USER="${AIRFLOW_ADMIN_USER:-admin}"

if [ -n "${AIRFLOW_ADMIN_PASSWORD:-}" ]; then
  python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_PASSWORDS_FILE"])
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_suffix(".tmp")
# The password comes from the environment and is never printed.
tmp.write_text(json.dumps({os.environ["AIRFLOW_ADMIN_USER"]: os.environ["AIRFLOW_ADMIN_PASSWORD"]}) + "\n")
tmp.chmod(0o600)
tmp.replace(path)
print(f"[airflow-login] login set for user '{os.environ['AIRFLOW_ADMIN_USER']}'")
PY
else
  echo "[airflow-login] AIRFLOW_ADMIN_PASSWORD is not set in .env:" \
       "Airflow generates a random password and prints it in this log" >&2
fi

exec airflow api-server
