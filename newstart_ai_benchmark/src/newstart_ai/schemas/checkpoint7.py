"""Result schemas for Version 6 Checkpoint 7: training and validating the new family-aware
chunked BERT classifier.

Every manifest here is built only from configuration, the family-aware TRAIN split/chunks,
and the family-aware VALIDATION split/chunks/conditions -- none of them may reference the
test split (see the extended `TestIsolationProof` produced alongside these, reusing
`newstart_ai.schemas.checkpoint6.TestIsolationProof`).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgencyClassWeightManifest(BaseModel):
    version: str
    created_at: str
    label_order: list[str]
    training_document_counts: dict[str, int]
    imbalance_ratio: float
    weighted_loss_threshold: float
    weighting_applied: bool
    formula: str
    raw_weights: dict[str, float]
    normalized_weights: dict[str, float]
    computed_from: str
    notes: list[str] = Field(default_factory=list)


class CombinedWeightingPolicy(BaseModel):
    version: str
    formula: str
    document_balancing_policy_version: str
    agency_class_weight_manifest_version: str
    notes: list[str] = Field(default_factory=list)


class TrainingHistoryEpoch(BaseModel):
    epoch: int
    train_loss: float
    validation_loss: float
    validation_document_macro_f1: float
    validation_document_accuracy: float
    learning_rate: float
    epoch_duration_seconds: float


class FamilyAwareModelMetadata(BaseModel):
    artifact_id: str
    display_name: str = "family-aware-chunked-bert"
    base_model: str
    tokenizer_revision: str
    tokenizer_resolved_commit_hash: str | None = None
    label_order: list[str]

    source_train_chunk_fingerprint: str
    source_validation_chunk_fingerprint: str
    source_train_split_fingerprint: str
    source_validation_split_fingerprint: str
    chunking_policy_version: str
    document_balancing_policy_version: str

    random_seed: int
    torch_version: str
    transformers_version: str
    cuda_available: bool
    cuda_device_name: str | None = None
    deterministic_algorithms_warnings: list[str] = Field(default_factory=list)

    training_config: dict = Field(default_factory=dict)
    class_weights: dict[str, float] = Field(default_factory=dict)

    history: list[TrainingHistoryEpoch] = Field(default_factory=list)
    best_epoch: int
    stopping_reason: str
    checkpoint_selection_metric: str
    checkpoint_selection_aggregation_method: str
    best_validation_document_macro_f1: float

    training_time_seconds: float
    peak_gpu_memory_mb: float | None = None

    status: str = "training"
    created_at: str
    ready_at: str | None = None


class AggregationMethodResult(BaseModel):
    method: str
    validation_document_macro_f1: float
    validation_document_accuracy: float
    worst_agency_f1: float
    per_agency_f1: dict[str, float]


class AggregationReconfirmationManifest(BaseModel):
    version: str
    created_at: str
    policy_version: str
    evaluated_on_checkpoint: str

    candidate_results: list[AggregationMethodResult]
    provisional_method: str
    selected_method: str
    method_changed: bool
    tie_break_steps_applied: list[str]
    supersedes: str | None = None

    notes: list[str] = Field(default_factory=list)


class ConditionEvaluationResult(BaseModel):
    condition: str
    masked: bool
    region: str
    document_count: int
    document_macro_f1: float
    document_accuracy: float
    per_agency_f1: dict[str, float]
    per_agency_support: dict[str, int]


class ConditionEvaluationManifest(BaseModel):
    version: str
    created_at: str
    aggregation_method_used: str
    condition_registry_fingerprint: str
    results: list[ConditionEvaluationResult]
    notes: list[str] = Field(default_factory=list)


class MisclassifiedDocument(BaseModel):
    document_id: str
    effective_family_id: str
    total_chunks: int
    true_label: str
    predicted_label: str


class ErrorConcentrationReport(BaseModel):
    by_chunk_count_bucket: dict[str, dict[str, int]]
    by_masked_vs_unmasked_error_rate: dict[str, float]
    by_complete_vs_partial_error_rate: dict[str, float]
    unseen_family_note: str


class DiagnosticsReport(BaseModel):
    version: str
    created_at: str
    training_time_seconds: float
    peak_gpu_memory_mb: float | None
    inference_latency_ms_per_document_mean: float
    confusion_matrix: dict[str, dict[str, int]]
    per_agency_document_support: dict[str, int]
    misclassified_documents: list[MisclassifiedDocument]
    error_concentration: ErrorConcentrationReport
    chunk_level_diagnostics_disclaimer: str


class Checkpoint7ReproducibilityManifest(BaseModel):
    version: str
    created_at: str

    source_train_split_fingerprint: str
    source_validation_split_fingerprint: str
    source_train_chunk_fingerprint: str
    source_validation_chunk_fingerprint: str
    configuration_fingerprint: str

    tokenizer_name: str
    tokenizer_revision: str
    tokenizer_resolved_commit_hash: str | None

    random_seed: int
    torch_version: str
    transformers_version: str
    python_packages: dict[str, str]

    label_order: list[str]
    class_weights: dict[str, float]
    document_balancing_policy_version: str

    best_checkpoint_artifact_id: str
    best_checkpoint_file_hashes: dict[str, str]

    validation_prediction_fingerprint: str
    final_aggregation_policy_version: str

    notes: list[str] = Field(default_factory=list)
