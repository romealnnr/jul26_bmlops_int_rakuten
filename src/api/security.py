"""API key check for the admin endpoints (/training, /drift/compute).

Usage in a route:  dependencies=[Depends(require_api_key)]

The caller sends the key in one of two headers:
  * X-API-Key: <key>                  (DAG, Streamlit, scripts, curl)
  * Authorization: Bearer <key>       (Grafana 11.3 webhook contact points
                                       cannot add custom headers, but they can
                                       send an Authorization header through
                                       authorization_scheme / _credentials)

The expected key is the API_KEY environment variable, read on every request
so a rotated key takes effect without code changes.

Decision when API_KEY is not set: refuse (HTTP 503 "API_KEY not configured")
rather than let the request through with a warning. These endpoints retrain and
replace the served model; a deployment that forgot to set the key would
otherwise be silently open, which is exactly the defect being fixed.
Failing closed makes the misconfiguration visible on the first call, and the
fix is one line in .env. Read-only endpoints (/health, /model, /predict,
/metrics) and the Grafana webhook /training/trigger are not affected.
"""

from __future__ import annotations

import hmac
import logging
import os

from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)

API_KEY_ENV = "API_KEY"
API_KEY_HEADER = "X-API-Key"


def _expected_key() -> str | None:
    # Surrounding whitespace (e.g. a trailing space or newline in .env) is not
    # part of the key, on either side of the comparison; otherwise the same
    # key would pass as X-API-Key and fail as a Bearer token.
    value = os.environ.get(API_KEY_ENV, "").strip()
    return value or None


def _presented_key(x_api_key: str | None, authorization: str | None) -> str | None:
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()
    if authorization:
        scheme, _, credentials = authorization.partition(" ")
        if scheme.lower() == "bearer" and credentials.strip():
            return credentials.strip()
    return None


def require_api_key(
    x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
    authorization: str | None = Header(default=None),
) -> None:
    """FastAPI dependency: 401 without a valid key, 503 when none is configured."""
    expected = _expected_key()
    if expected is None:
        logger.error("Admin endpoint called but %s is not set: refusing", API_KEY_ENV)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{API_KEY_ENV} not configured on the server",
        )

    presented = _presented_key(x_api_key, authorization)
    # hmac.compare_digest takes the same time whatever the first differing
    # character is, so the key cannot be guessed one character at a time from
    # response times. Compare bytes so non-ASCII input cannot raise.
    if presented is None or not hmac.compare_digest(
        presented.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing or invalid API key (header {API_KEY_HEADER})",
            headers={"WWW-Authenticate": "Bearer"},
        )
