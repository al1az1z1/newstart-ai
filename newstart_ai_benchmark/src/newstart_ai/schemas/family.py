"""Result schemas for the Version 6 family-aware robustness research's family-discovery
step (Robustness_v6_Family_Aware_Chunked_BERT.md, Checkpoint 3).

A "family" groups documents that must never be split across train/validation/test (a main
form, its instructions, supplements, and translated/revised versions), while every document
keeps its own independent `document_id`. Nothing here modifies `final_dataset.csv`, the
historical splits, or historical artifacts -- overrides are proposals only, applied (if
approved) starting in a later checkpoint.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RelationshipType = Literal[
    "form",
    "instructions",
    "supplement",
    "checklist",
    "translated_form",
    "singleton",
    "unclassified",
]

EvidenceType = Literal[
    "form_number_exact",
    "filename_code_match",
    "singleton_no_evidence",
]

ReviewStatus = Literal[
    "auto_grouped", "needs_review", "singleton_confirmed", "resolved_manual_review"
]

ModelingEligibility = Literal[
    "include_english_corpus",
    "exclude_non_english",
    "exclude_insufficient_text",
    "pending_review",
]


class FamilyAuditRow(BaseModel):
    """Four concepts are kept deliberately separate (never collapsed into one field):

    - `family_id` / `effective_family_id`: leakage-safe grouping. `family_id` is the
      original, source-label-derived grouping; `effective_family_id` is what actually
      governs splitting once any approved agency or manual family override is applied.
    - `agency` / `effective_agency`: the source label, and the label after any approved
      override.
    - `language_status`: detected/manually-confirmed language evidence -- a fact about the
      text, not a decision.
    - `recommended_modeling_eligibility` / `final_modeling_eligibility`: whether THIS
      document (not its family) enters the corrected English modeling corpus. Family
      members may share a family_id while having different eligibility -- an English form
      and its translation are never both forced to the same eligibility.
    """

    document_id: str
    agency: str
    filename: str
    form_number: str | None = None
    document_type: str
    language_status: str  # from Checkpoint 2's LanguageAuditRow.status, or manually confirmed
    detected_language: str | None = None
    manual_language_notes: str | None = None

    family_id: str
    effective_family_id: str
    family_size: int
    relationship_type: RelationshipType
    evidence_type: EvidenceType
    evidence_detail: str
    confidence: float
    review_status: ReviewStatus
    conflict_reason: str | None = None

    recommended_modeling_eligibility: ModelingEligibility
    final_modeling_eligibility: ModelingEligibility
    agency_override_proposed: str | None = None
    effective_agency: str
    family_override_proposed: str | None = None


class DuplicateCandidate(BaseModel):
    document_id_a: str
    document_id_b: str
    same_family: bool
    similarity: float
    method: Literal["exact_normalized_text_hash", "tfidf_cosine"]


class CrossAgencyConflict(BaseModel):
    base_code: str
    agencies: list[str]
    document_ids: list[str]
    note: str


class OverrideFieldChange(BaseModel):
    field: Literal["agency", "modeling_eligibility", "family_id", "effective_family_id"]
    before: str | None
    after: str | None


class OverrideProposal(BaseModel):
    document_id: str
    field_changes: list[OverrideFieldChange]
    evidence: list[str]
    status: Literal["proposed"] = "proposed"
    requires_approval: bool = True


class ManualReviewFlag(BaseModel):
    document_id: str
    reason: str
    evidence: list[str]


class FamilyAuditManifest(BaseModel):
    version: str
    created_at: str
    source_dataset_fingerprint: str
    total_documents: int
    total_families: int
    singleton_family_count: int
    non_singleton_family_count: int
    families_by_agency: dict[str, int]
    documents_by_agency: dict[str, int]
    form_instruction_or_supplement_family_count: int
    cross_language_family_count: int
    duplicate_candidate_count: int
    cross_agency_conflict_count: int
    ambiguous_review_count: int
    notes: list[str] = Field(default_factory=list)
