"""Unit tests for point 16: a missing or wrong image_path is reported, never silently
replaced by a blank image. No database, no MLflow server, no ResNet download."""

from __future__ import annotations

import os

for _key, _value in {
    "POSTGRES_USER": "t",
    "POSTGRES_PASSWORD": "t",
    "POSTGRES_DB": "t",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "MODEL_NAME": "baseline_logreg",
    "TARGET_COLUMN": "target",
    "TEST_SIZE": "0.2",
    "RANDOM_STATE": "42",
    "API_HOST": "0.0.0.0",
    "API_PORT": "8000",
    "LOG_LEVEL": "INFO",
    "MLFLOW_EXPERIMENT_NAME": "rakuten-test",
    "MLFLOW_REGISTERED_MODEL_NAME": "rakuten_baseline",
}.items():
    os.environ.setdefault(_key, _value)

import logging  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer  # noqa: E402

from src.features.image_features import IMAGE_EMBEDDING_DIM, ImageEmbeddingTransformer  # noqa: E402
from src.models import predict as predict_module  # noqa: E402

FEATURES = ["designation", "description", "productid", "imageid", "image_path"]


def has_image(frame) -> np.ndarray:
    """Stub image encoder: one column, 1.0 when a path reached the model."""
    values = frame.iloc[:, 0] if hasattr(frame, "iloc") else frame
    return np.array([[0.0 if v is None or isinstance(v, float) else 1.0] for v in values])


@pytest.fixture
def raw_dir(monkeypatch, tmp_path):
    root = tmp_path / "raw"
    (root / "image_train").mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(root / "image_train" / "ok.jpg")
    (root / "image_train" / "broken.jpg").write_text("not an image")
    monkeypatch.setattr(predict_module, "RAW_DATA_DIR", root)
    return root


@pytest.fixture
def served(monkeypatch):
    pipeline = Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        ("text", TfidfVectorizer(), "designation"),
                        ("image", FunctionTransformer(has_image), ["image_path"]),
                    ]
                ),
            ),
            ("classifier", LogisticRegression(max_iter=500)),
        ]
    )
    X = pd.DataFrame(
        {
            "designation": ["livre roman", "piscine gonflable"] * 4,
            "description": [""] * 8,
            "productid": [1] * 8,
            "imageid": [1] * 8,
            "image_path": [None] * 8,
        }
    )
    pipeline.fit(X, [10, 2583] * 4)
    artifact = {"pipeline": pipeline, "version": "1", "feature_names": FEATURES}
    monkeypatch.setattr(predict_module, "load_artifact", lambda force_reload=False: artifact)
    return artifact


def test_valid_image_inside_raw_dir_is_used(raw_dir):
    path, warning = predict_module._check_image(str(raw_dir / "image_train" / "ok.jpg"))
    assert warning is None and path.endswith("ok.jpg")


def test_relative_image_path_is_read_from_raw_dir(raw_dir):
    path, warning = predict_module._check_image("image_train/ok.jpg")
    assert warning is None and path == str((raw_dir / "image_train" / "ok.jpg").resolve())


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (None, "no image_path"),
        ("", "no image_path"),
        ("image_train/missing.jpg", "image_path not found"),
        ("image_train/broken.jpg", "image_path unreadable"),
        ("/etc/hostname", "image_path outside the raw data folder"),
        ("../../etc/passwd", "image_path outside the raw data folder"),
    ],
)
def test_unusable_image_is_reported(raw_dir, given, expected):
    path, warning = predict_module._check_image(given)
    assert path is None
    assert warning.startswith(expected) and warning.endswith("prediction uses text only")


def test_predict_reports_whether_the_image_was_used(served, raw_dir):
    results = predict_module.predict(
        [
            {"designation": "livre roman", "image_path": "image_train/ok.jpg"},
            {"designation": "livre roman", "image_path": "image_train/missing.jpg"},
            {"designation": "piscine gonflable"},
        ],
        log_to_db=False,
    )
    assert [r["image_used"] for r in results] == [True, False, False]
    assert results[0]["warnings"] == []
    assert results[1]["warnings"][0].startswith("image_path not found")
    assert results[2]["warnings"][0].startswith("no image_path")


def test_image_transformer_logs_unreadable_paths(monkeypatch, tmp_path, caplog):
    broken = tmp_path / "broken.jpg"
    broken.write_text("not an image")
    transformer = ImageEmbeddingTransformer(batch_size=4)

    class Stub(torch.nn.Module):
        def forward(self, x):
            return torch.ones(x.shape[0], IMAGE_EMBEDDING_DIM)

    monkeypatch.setattr(transformer, "_get_model", lambda: Stub())
    with caplog.at_level(logging.INFO, logger="src.features.image_features"):
        out = transformer.transform([str(broken), None])

    assert out.shape == (2, IMAGE_EMBEDDING_DIM) and not out.any()
    assert "1 of 2 image paths could not be read" in caplog.text
    assert "1 of 2 rows have no image_path" in caplog.text
