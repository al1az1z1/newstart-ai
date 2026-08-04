"""Result schemas for Version 6 Checkpoint 8: the one-time family-aware BERT test evaluation.

`PreTestFreezeRecord` must be built and saved BEFORE any test file is opened -- it is the
attestation that nothing (checkpoint, aggregation, thresholds, masking/partial-input rules)
will change based on what the test results show.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from newstart_ai.schemas.checkpoint6 import ConditionDefinition


class PreTestFreezeRecord(BaseModel):
    version: str
    created_at: str

    best_checkpoint_artifact_id: str
    best_checkpoint_file_hashes: dict[str, str]

    aggregation_method: str
    aggregation_policy_version: str

    label_order: list[str]

    tokenizer_name: str
    tokenizer_revision: str
    tokenizer_resolved_commit_hash: str | None
    tokenizer_file_hashes: dict[str, str]

    condition_definitions: list[ConditionDefinition]

    configuration_fingerprint: str
    test_split_fingerprint: str
    test_chunk_fingerprint: str

    no_changes_confirmation: str
    frozen: bool = True


class PerAgencyTestMetrics(BaseModel):
    agency: str
    precision: float
    recall: float
    f1: float
    support: int


class TestMisclassificationDetail(BaseModel):
    document_id: str
    effective_family_id: str
    total_chunks: int
    true_label: str
    predicted_label: str
    confidence: float


class PrimaryTestResult(BaseModel):
    version: str
    created_at: str
    condition: str
    aggregation_method: str
    document_count: int

    document_macro_f1: float
    document_accuracy: float
    macro_precision: float
    macro_recall: float

    per_agency: list[PerAgencyTestMetrics]
    confusion_matrix: dict[str, dict[str, int]]
    misclassifications: list[TestMisclassificationDetail]

    inference_latency_ms_per_document: float
    peak_gpu_memory_mb: float | None

    notes: list[str] = Field(default_factory=list)


class TestConditionResult(BaseModel):
    condition: str
    masked: bool
    region: str
    document_count: int
    document_macro_f1: float
    document_accuracy: float
    per_agency_f1: dict[str, float]
    per_agency_support: dict[str, int]
    error_count: int
    error_rate: float
    difference_from_complete_unmasked_macro_f1: float


class TestConditionSweepManifest(BaseModel):
    version: str
    created_at: str
    aggregation_method_used: str
    condition_registry_fingerprint: str
    results: list[TestConditionResult]
    notes: list[str] = Field(default_factory=list)


class ChunkBucketErrorCounts(BaseModel):
    bucket: str
    correct: int
    incorrect: int


class FamilyErrorCounts(BaseModel):
    effective_family_id: str
    document_count: int
    error_count: int


class TestErrorAnalysis(BaseModel):
    version: str
    created_at: str

    by_agency_error_count: dict[str, int]
    by_agency_document_count: dict[str, int]
    by_effective_family: list[FamilyErrorCounts]
    by_chunk_count_bucket: list[ChunkBucketErrorCounts]
    by_condition_error_rate: dict[str, float]
    by_region_error_rate: dict[str, float]

    low_confidence_observations: list[str]
    masking_replacement_observations: list[str]
    irs_caution_note: str

    observations_vs_causal_disclaimer: str


class TestIntegrityProof(BaseModel):
    version: str
    created_at: str

    exact_test_document_count: int
    expected_test_document_count: int
    exact_document_set_matches_frozen_split: bool
    every_document_appears_exactly_once: bool
    no_train_document_overlap: bool
    no_validation_document_overlap: bool
    no_train_family_overlap: bool
    no_validation_family_overlap: bool
    no_excluded_document_evaluated: bool

    checkpoint_used_matches_approved: bool
    checkpoint_artifact_id: str
    aggregation_method_used: str
    aggregation_matches_frozen_policy: bool
    condition_policy_versions_match_frozen: bool

    no_retraining_or_policy_revision_triggered: bool

    notes: list[str] = Field(default_factory=list)


class HistoricalComparisonContext(BaseModel):
    version: str
    created_at: str

    historical_test_document_count: int
    historical_test_accuracy: float
    historical_test_macro_f1: float
    historical_split_note: str

    new_test_document_count: int
    new_test_note: str

    comparison_guidance: str


class TestReproducibilityManifest(BaseModel):
    version: str
    created_at: str

    test_split_fingerprint: str
    test_chunk_fingerprint: str
    prediction_fingerprint_by_condition: dict[str, str]

    checkpoint_artifact_id: str
    checkpoint_file_hashes: dict[str, str]

    torch_version: str
    transformers_version: str
    python_packages: dict[str, str]

    notes: list[str] = Field(default_factory=list)
