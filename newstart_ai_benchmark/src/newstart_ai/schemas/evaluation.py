"""Result schema produced by the Evaluator for a single method's run.

Macro F1 is the primary metric everywhere (docs/BLUEPRINT.md Section 6) -- accuracy alone is
never used to declare a winner, and per-class metrics + confusion matrix are always included
so a small class like IRS isn't hidden behind an aggregate number.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from newstart_ai.schemas.classification import Method


class PerClassMetrics(BaseModel):
    label: str
    precision: float
    recall: float
    f1: float
    support: int


class MetricsReport(BaseModel):
    method: Method
    split: str  # "validation" | "test" -- validation-only passes never feed the test comparison

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float

    per_class: list[PerClassMetrics] = Field(default_factory=list)
    confusion_matrix: list[list[int]]
    confusion_matrix_labels: list[str]

    mean_latency_ms: float

    # LLM / LLM+RAG only
    total_token_usage: int | None = None
    total_estimated_cost: float | None = None
    cost_per_document: float | None = None

    notes: list[str] = Field(default_factory=list)  # e.g. IRS small-sample caveat
