"""Result schemas for Version 6 Checkpoint 10: the one-time frozen head-to-head test
evaluation of BERT, plain Gemini LLM, and Gemini LLM+RAG.

`Checkpoint10FreezeRecord` must be built and saved BEFORE any Gemini classification request
-- it is the attestation that no model, prompt, retrieval policy, parsing policy, or
evaluation rule will be changed based on test outcomes.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from newstart_ai.schemas.checkpoint6 import ConditionDefinition


class Checkpoint10FreezeRecord(BaseModel):
    version: str
    created_at: str

    llm_model_name: str
    generation_temperature: float
    generation_max_output_tokens: int | None

    plain_prompt_version: str
    plain_prompt_hash: str
    rag_prompt_version: str
    rag_prompt_hash: str
    response_schema_hash: str
    parser_version: str

    allowed_labels: list[str]
    label_order: list[str]

    invalid_output_policy: str
    refusal_policy: str
    retry_policy: str
    timeout_policy: str
    api_failure_policy: str
    max_attempts: int
    retry_backoff_seconds: float

    bert_checkpoint_artifact_id: str
    bert_checkpoint_file_hashes: dict[str, str]
    bert_aggregation_method: str

    rag_embedding_model: str
    rag_unmasked_corpus_fingerprint: str
    rag_masked_corpus_fingerprint: str

    retrieval_candidate_pool_size: int
    retrieval_top_k: int
    retrieval_max_chunks_per_parent_document: int
    retrieval_max_results_per_effective_family: int
    retrieval_tie_breaker: str
    retrieval_duplicate_handling: str
    retrieval_fewer_than_k_behavior: str

    condition_definitions: list[ConditionDefinition]

    test_split_fingerprint: str
    test_condition_registry_fingerprint: str

    no_changes_confirmation: str
    frozen: bool = True


class RetrievedChunkProvenance(BaseModel):
    chunk_id: str
    rank: int
    similarity: float
    parent_document_id: str
    effective_family_id: str
    effective_agency: str
    text_hash: str
    masked: bool


class CaseResult(BaseModel):
    method: str  # "bert" | "llm" | "llm_rag"
    document_id: str
    effective_family_id: str
    condition: str
    true_label: str

    input_fingerprint: str
    retrieval_context_fingerprint: str | None = None

    predicted_label: str | None
    confidence: float | None = None
    raw_response_hash: str | None = None

    status: str  # "success" | "invalid" | "failed"
    error_type: str | None = None
    attempt_count: int
    truncated: bool = False

    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None

    retrieved_chunks: list[RetrievedChunkProvenance] = Field(default_factory=list)

    cache_key: str


class PerAgencyMetrics(BaseModel):
    agency: str
    precision: float
    recall: float
    f1: float
    support: int


class LatencyStats(BaseModel):
    mean_ms: float
    median_ms: float
    p95_ms: float
    total_ms: float


class MethodConditionMetrics(BaseModel):
    method: str
    condition: str
    document_count: int
    coverage_rate: float
    invalid_count: int
    failed_count: int

    document_macro_f1: float
    document_accuracy: float
    macro_precision: float
    macro_recall: float
    per_agency: list[PerAgencyMetrics]
    confusion_matrix: dict[str, dict[str, int]]
    error_count: int
    error_rate: float

    latency: LatencyStats
    estimated_tokens_total: int | None = None
    estimated_cost_total_usd: float | None = None

    notes: list[str] = Field(default_factory=list)


class PrimaryPairedComparison(BaseModel):
    version: str
    created_at: str
    condition: str
    document_count: int

    all_three_correct: list[str]
    all_three_incorrect: list[str]
    bert_only_errors: list[str]
    plain_llm_only_errors: list[str]
    rag_only_errors: list[str]
    rag_corrects_plain_llm: list[str]
    rag_breaks_plain_llm: list[str]
    plain_llm_and_rag_identical_predictions: list[str]

    plain_llm_and_rag_agreement_rate: float
    notes: list[str] = Field(default_factory=list)


class RobustnessConditionDelta(BaseModel):
    method: str
    condition: str
    macro_f1_delta_from_complete_unmasked: float
    accuracy_delta_from_complete_unmasked: float
    error_count: int
    error_rate: float


class RobustnessComparisonManifest(BaseModel):
    version: str
    created_at: str
    deltas: list[RobustnessConditionDelta]
    masking_effect_notes: list[str]
    partial_input_effect_notes: list[str]
    rag_help_notes: list[str]
    error_concentration_notes: list[str]
    irs_caution_note: str
    disclaimer: str


class BootstrapResult(BaseModel):
    metric: str
    method: str
    point_estimate: float
    ci_low: float
    ci_high: float
    n_bootstrap: int
    seed: int


class PairedBootstrapResult(BaseModel):
    metric: str
    method_a: str
    method_b: str
    observed_difference: float
    ci_low: float
    ci_high: float
    n_bootstrap: int
    seed: int


class McNemarResult(BaseModel):
    method_a: str
    method_b: str
    a_correct_b_incorrect: int
    a_incorrect_b_correct: int
    statistic: float | None
    p_value: float | None
    note: str


class StatisticalUncertaintyManifest(BaseModel):
    version: str
    created_at: str
    condition: str
    seed: int
    bootstrap_results: list[BootstrapResult]
    paired_bootstrap_results: list[PairedBootstrapResult]
    mcnemar_results: list[McNemarResult]
    interpretation_caution: str


class EvaluationIntegrityProof(BaseModel):
    version: str
    created_at: str

    exact_99_test_documents_evaluated: bool
    exactly_990_cases_per_method: bool
    one_record_per_document_condition_per_method: bool
    condition_fingerprints_match_across_methods: bool

    no_train_validation_text_used_outside_rag_indexes: bool
    both_rag_indexes_train_only: bool
    no_test_label_or_agency_metadata_in_prompts: bool
    masked_queries_used_masked_index_only: bool
    unmasked_queries_used_unmasked_index_only: bool

    approved_model_prompt_parser_checkpoint_retrieval_used: bool
    no_training_or_policy_selection_function_ran: bool
    cached_resumed_results_no_duplicates: bool

    historical_artifacts_unchanged: bool
    checkpoint_4_9_artifacts_unchanged: bool

    notes: list[str] = Field(default_factory=list)


class Checkpoint10CostRuntimeReport(BaseModel):
    version: str
    created_at: str

    estimated_plain_llm_calls: int
    estimated_rag_calls: int
    estimated_query_embeddings_after_cache: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    estimated_runtime_seconds: float
    estimate_methodology: str

    actual_plain_llm_calls: int
    actual_rag_calls: int
    actual_query_embeddings: int
    actual_prompt_tokens: int
    actual_completion_tokens: int
    actual_cost_usd: float
    actual_runtime_seconds: float
    actual_retries: int
    actual_failures: int
    actual_invalid: int

    cost_rate_is_placeholder: bool
    notes: list[str] = Field(default_factory=list)
