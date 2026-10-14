"""Validation of the records sent to /predict (also used by predict() for CLI and DAG callers)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

MAX_RECORDS_PER_REQUEST = 1000


class ProductRecord(BaseModel):
    """One product to classify.

    Unknown fields are rejected (422) instead of being silently ignored, and the
    title (designation) is required: a prediction without a title is a guess.
    """

    model_config = ConfigDict(extra="forbid")

    designation: str = Field(description="Product title (required, not blank)")
    description: str | None = Field(default=None, description="Product description")
    productid: int | None = Field(default=None, description="Catalogue id")
    imageid: int | None = Field(default=None, description="Image id")
    image_path: str | None = Field(default=None, description="Image file path")
    true_label: StrictInt | None = Field(
        default=None,
        description="True category if known (stored with the prediction, never a model input)",
    )

    @field_validator("designation")
    @classmethod
    def designation_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("designation must not be empty")
        return value


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "records": [
                    {
                        "designation": "Console de jeu portable retro",
                        "description": "400 jeux inclus, écran couleur",
                        "productid": 123456,
                        "imageid": 987654,
                        "image_path": None,
                        "true_label": 2462,
                    }
                ]
            }
        },
    )

    records: list[ProductRecord] = Field(min_length=1, max_length=MAX_RECORDS_PER_REQUEST)
