"""Result schemas for Version 6 Checkpoint 6: document aggregation, long-document training
balance, deterministic partial-input selection, identifier masking, and the shared
evaluation-condition registry.

Every manifest here is built from configuration, training data, and validation data only --
none of them may be constructed from test labels, test predictions, or test chunk
distributions (see `TestIsolationProof`).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AggregationComparisonManifest(BaseModel):
    version: str
    created_at: str
    policy_version: str

    candidate_methods: list[str]
    default_method: str
    tie_breaker: str

    selection_basis: str
    validation_chunk_count_structure: dict = Field(default_factory=dict)
    deterministic_properties_by_method: dict[str, list[str]] = Field(default_factory=dict)

    provisional: bool
    reconfirmation_plan: str

    notes: list[str] = Field(default_factory=list)


class LargeDocumentContribution(BaseModel):
    document_id: str
    agency: str
    total_chunks: int
    raw_chunk_share_percent: float
    weighted_contribution_share_percent: float


class DocumentBalancingManifest(BaseModel):
    version: str
    created_at: str
    policy_version: str
    method: str

    total_training_documents: int
    total_training_chunks: int
    weight_sum_equals_document_count: bool

    largest_documents_effect: list[LargeDocumentContribution] = Field(default_factory=list)

    separate_from_agency_class_weighting: bool
    notes: list[str] = Field(default_factory=list)


class PartialInputSelectionRow(BaseModel):
    document_id: str
    split: str
    condition: str
    requested_regions: list[str]
    selected_chunk_indices: list[int]
    total_chunks: int
    fallback_reason: str | None
    selection_hash: str
    policy_version: str


class PartialInputManifest(BaseModel):
    version: str
    created_at: str
    policy_version: str
    chunks_per_region: int

    total_documents: int
    fallback_document_counts_by_condition: dict[str, int] = Field(default_factory=dict)
    no_unjustified_duplicate_selection: bool

    notes: list[str] = Field(default_factory=list)


class MaskingRuleMatchSummary(BaseModel):
    rule_name: str
    total_matches: int
    documents_with_at_least_one_match: int


class MaskingAuditExample(BaseModel):
    document_id: str
    agency: str
    split: str
    rule_match_counts: dict[str, int]
    total_replacements: int
    masked_text_snippet: str
    note: str


class MaskingSplitSummary(BaseModel):
    split: str
    document_count: int
    total_replacements: int
    documents_with_zero_matches: int


class MaskingManifest(BaseModel):
    version: str
    created_at: str
    policy_version: str

    rule_names: list[str]
    rule_match_summary: list[MaskingRuleMatchSummary] = Field(default_factory=list)
    per_split_summary: list[MaskingSplitSummary] = Field(default_factory=list)
    audit_examples: list[MaskingAuditExample] = Field(default_factory=list)

    ground_truth_label_unchanged: bool
    fitted_without_examining_test_outcomes: bool
    applies_identically_across_methods: bool

    notes: list[str] = Field(default_factory=list)


class ConditionDefinition(BaseModel):
    name: str
    definition: str
    masked: bool
    region: str
    policy_version: str


class ConditionRegistryManifest(BaseModel):
    version: str
    created_at: str
    policy_version: str

    conditions: list[ConditionDefinition]
    total_documents: int
    total_rows: int
    per_condition_row_counts: dict[str, int] = Field(default_factory=dict)
    registry_fingerprint: str

    notes: list[str] = Field(default_factory=list)


class TestIsolationProof(BaseModel):
    version: str
    created_at: str

    functions_exercised: list[str]
    input_files_used: list[str]
    test_files_referenced: list[str]
    isolation_holds: bool
    proof_statement: str
