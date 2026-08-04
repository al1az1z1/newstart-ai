"""Result schemas for the Version 6 family-aware robustness research's English-filtering
step (Robustness_v6_Family_Aware_Chunked_BERT.md, Checkpoint 2).

Kept separate from the historical classification schemas -- this audit describes the source
dataset, not a classifier's predictions, and never touches historical splits or artifacts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

LanguageStatus = Literal["confidently_english", "confidently_non_english", "uncertain_review"]


class LanguageAuditRow(BaseModel):
    document_id: str
    agency: str
    filename: str
    form_number: str | None = None
    text_length: int
    detected_language: str | None = None
    confidence: float | None = None
    status: LanguageStatus
    reason: str


class LanguageFilterManifest(BaseModel):
    detector_name: str
    detector_version: str
    detector_config: dict
    created_at: str
    source_dataset_path: str
    source_dataset_fingerprint: str
    total_documents: int
    counts_by_status: dict[str, int]
    counts_by_status_and_agency: dict[str, dict[str, int]]
    notes: list[str] = Field(default_factory=list)
