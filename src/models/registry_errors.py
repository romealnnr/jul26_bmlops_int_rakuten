"""Tell "this model or alias does not exist" apart from real registry errors.

Only a missing model or a missing alias means "no champion yet". Anything else
(registry unreachable, permission denied, server error) must surface: treating
it as "no champion" would make the API report no model, or make a training run
promote its model as if it were the first one.
"""
from __future__ import annotations

from mlflow.exceptions import MlflowException


def is_not_found(exc: Exception) -> bool:
    """True when MLflow says the registered model or the alias does not exist.

    MLflow 3 answers RESOURCE_DOES_NOT_EXIST for an unknown model and
    INVALID_PARAMETER_VALUE ("Registered model alias ... not found") for an
    unknown alias on a known model."""
    if not isinstance(exc, MlflowException):
        return False
    if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
        return True
    message = str(exc)
    return (
        exc.error_code == "INVALID_PARAMETER_VALUE"
        and "alias" in message
        and "not found" in message
    )
