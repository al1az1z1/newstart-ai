"""Result schema for the Version 6 family-aware split (Checkpoint 4).

Splitting groups by `effective_family_id` (never `family_id`, and never individual
documents) so that every eligible member of an effective family lands in exactly one split.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SplitAgencyCounts(BaseModel):
    agency: str
    document_count: int
    family_count: int


class SplitCounts(BaseModel):
    split: str
    document_count: int
    family_count: int
    percentage_of_eligible_documents: float
    by_agency: list[SplitAgencyCounts] = Field(default_factory=list)


class FamilyAwareSplitManifest(BaseModel):
    version: str
    created_at: str
    random_seed: int
    configured_ratios: dict[str, float]

    source_dataset_fingerprint: str
    eligibility_manifest_fingerprint: str
    override_artifact_fingerprint: str
    override_artifact_version: str

    total_eligible_documents: int
    total_eligible_families: int

    splits: list[SplitCounts]
    split_fingerprints: dict[str, str]

    all_agencies_in_every_split: bool
    agencies_missing_by_split: dict[str, list[str]]

    zero_document_overlap: bool
    zero_family_overlap: bool
    every_eligible_document_assigned_exactly_once: bool
    no_excluded_or_unresolved_document_in_any_split: bool

    notes: list[str] = Field(default_factory=list)
