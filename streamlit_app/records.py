"""Turn an uploaded CSV into /predict records (pure pandas, no Streamlit import)."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

import pandas as pd

# Columns that are integers in the API schema. pandas reads an integer column that
# has empty cells as float (2462 -> 2462.0), and /predict rejects 2462.0 for
# true_label (strict integer), so whole-number floats are turned back into int.
INTEGER_FIELDS = ("productid", "imageid", "true_label")


def _clean(field: str, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Real) and not isinstance(value, Integral):
        if math.isnan(float(value)):
            return None
        if field in INTEGER_FIELDS and float(value).is_integer():
            return int(value)
        return float(value)
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, str) or value is pd.NA:
        return None if value is pd.NA else value
    return value


def records_for_api(df: pd.DataFrame, fields: list[str]) -> list[dict[str, Any]]:
    """Keep only the accepted columns, empty cells become None, integers stay integers."""
    keep = [c for c in fields if c in df.columns]
    return [
        {field: _clean(field, value) for field, value in row.items()}
        for row in df[keep].to_dict("records")
    ]
