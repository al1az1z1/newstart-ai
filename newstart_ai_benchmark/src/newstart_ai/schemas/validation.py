"""Result schema for non-destructive dataset validation (see src/newstart_ai/data/validation.py).

Validation only reports problems -- it never rewrites the source dataset. A new dataset
version is created upstream (notebooks/00_data_acquisition) if a real fix is needed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClassCount(BaseModel):
    label: str
    count: int
    percentage: float


class LengthStats(BaseModel):
    mean: float
    median: float
    minimum: int
    maximum: int
    p95: float


class ValidationReport(BaseModel):
    row_count: int
    required_columns_present: bool
    missing_columns: list[str] = Field(default_factory=list)

    document_id_column_unique: bool
    duplicate_document_id_count: int

    empty_text_count: int
    duplicate_text_count: int

    valid_labels: bool
    invalid_label_values: list[str] = Field(default_factory=list)
    class_counts: list[ClassCount] = Field(default_factory=list)
    minimum_class_count: int
    imbalance_ratio: float  # majority class count / minority class count

    stratified_split_feasible: bool
    stratified_split_blockers: list[str] = Field(default_factory=list)

    text_length: LengthStats

    warnings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    @property
    def has_critical_errors(self) -> bool:
        """True when training must not proceed until the underlying data problem is fixed."""
        return (
            not self.required_columns_present
            or not self.document_id_column_unique
            or not self.valid_labels
            or not self.stratified_split_feasible
        )
