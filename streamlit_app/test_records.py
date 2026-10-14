"""CSV upload -> /predict records: run with `pytest streamlit_app/test_records.py`."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from records import records_for_api  # noqa: E402

FIELDS = ["designation", "description", "productid", "imageid", "image_path", "true_label"]


def test_partly_empty_true_label_becomes_int_or_none():
    df = pd.DataFrame(
        {
            "designation": ["a", "b"],
            "description": ["x", None],
            "true_label": [2462, None],  # pandas stores this column as float64
            "productid": [123, 456],
            "extra": [1, 2],
        }
    )
    assert df["true_label"].dtype == "float64"
    # The page sends df.head(n).astype(object); both forms must give the same records.
    for frame in (df, df.astype(object)):
        records = records_for_api(frame, FIELDS)
        assert records == [
            {"designation": "a", "description": "x", "productid": 123, "true_label": 2462},
            {"designation": "b", "description": None, "productid": 456, "true_label": None},
        ]
        assert type(records[0]["true_label"]) is int and type(records[0]["productid"]) is int


def test_records_are_accepted_by_the_api_schema():
    # The same strict schema the API applies (src/api/schemas.py).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.api.schemas import ProductRecord

    df = pd.DataFrame({"designation": ["a", "b"], "true_label": [2462.0, float("nan")]})
    for record in records_for_api(df, FIELDS):
        ProductRecord.model_validate(record)
