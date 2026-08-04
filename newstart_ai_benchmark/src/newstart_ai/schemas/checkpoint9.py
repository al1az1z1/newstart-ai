"""Result schemas for Version 6 Checkpoint 9: the family-aware RAG retrieval layer.

Built and validated using only frozen family-aware training documents/chunks (indexing) and
validation documents/conditions (retrieval diagnostics) -- none of these schemas may be
populated from test data.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EmbeddingConfigManifest(BaseModel):
    version: str
    created_at: str
    provider: str
    model_name: str
    document_task_type: str
    query_task_type: str
    configured_output_dimensionality: int | None
    observed_vector_dimension: int
    embedding_batch_size: int
    observed_mean_l2_norm: float
    observed_norm_is_unit_length: bool
    genai_sdk_version: str
    configuration_fingerprint: str
    cache_dir: str
    notes: list[str] = Field(default_factory=list)


class EmbeddingUsageReport(BaseModel):
    version: str
    created_at: str
    corpus: str  # "unmasked_training_corpus" | "masked_training_corpus" | "validation_queries"
    total_texts_requested: int
    cache_hits: int
    cache_misses: int
    api_requests_made: int
    total_tokens_billed: int
    tokens_are_estimated: bool
    retries: int
    failures: int
    wall_clock_seconds: float
    estimated_cost_usd: float
    cost_rate_per_million_tokens_usd: float
    cost_rate_is_placeholder: bool
    notes: list[str] = Field(default_factory=list)


class CorpusManifest(BaseModel):
    version: str
    created_at: str
    corpus_type: str  # "unmasked" | "masked"
    indexed_chunk_count: int
    indexed_document_count: int
    indexed_family_count: int
    corpus_fingerprint: str
    embedding_config_fingerprint: str
    source_train_chunk_fingerprint: str
    persist_dir: str
    collection_name: str
    notes: list[str] = Field(default_factory=list)


class DiversificationPolicyManifest(BaseModel):
    version: str
    created_at: str
    policy_version: str
    candidate_pool_size: int
    top_k: int
    max_chunks_per_parent_document: int
    max_results_per_effective_family: int
    minimum_similarity_threshold: float | None
    tie_breaker: str
    duplicate_handling: str
    fewer_than_k_behavior: str
    notes: list[str] = Field(default_factory=list)


class RetrievalBeforeAfterSample(BaseModel):
    query_document_id: str
    condition: str
    top_parent_document_share_before: float
    top_parent_document_share_after: float
    top_family_share_before: float
    top_family_share_after: float
    result_count_before: int
    result_count_after: int


class DiversificationEffectReport(BaseModel):
    version: str
    created_at: str
    sample_size: int
    mean_top_parent_share_before: float
    mean_top_parent_share_after: float
    mean_top_family_share_before: float
    mean_top_family_share_after: float
    document_739_max_positions_before: int
    document_739_max_positions_after: int
    samples: list[RetrievalBeforeAfterSample]
    notes: list[str] = Field(default_factory=list)


class PerAgencyRetrievalDiagnostic(BaseModel):
    agency: str
    query_document_count: int
    top_k_agency_hit_rate: float
    mean_reciprocal_rank: float
    note: str = ""


class ConditionRetrievalDiagnostic(BaseModel):
    condition: str
    masked: bool
    index_used: str
    query_document_count: int
    top_k_agency_hit_rate: float
    mean_reciprocal_rank: float
    mean_similarity: float
    similarity_std: float
    percent_queries_with_fewer_than_k_results: float
    mean_parent_duplication_before: float
    mean_parent_duplication_after: float
    mean_family_duplication_before: float
    mean_family_duplication_after: float
    per_agency: list[PerAgencyRetrievalDiagnostic]


class ValidationRetrievalDiagnosticsManifest(BaseModel):
    version: str
    created_at: str
    candidate_configs_considered: list[dict]
    selected_configuration: dict
    selection_basis: str
    results: list[ConditionRetrievalDiagnostic]
    classifier_performance_disclaimer: str
    notes: list[str] = Field(default_factory=list)


class RetrievalResultRecord(BaseModel):
    query_document_id: str
    condition: str
    masked: bool
    rank: int
    chunk_id: str
    parent_document_id: str
    effective_family_id: str
    effective_agency: str
    similarity: float
    text_hash: str


class RagIntegrityProof(BaseModel):
    version: str
    created_at: str

    every_indexed_vector_from_train_split: bool
    no_validation_document_indexed: bool
    no_test_document_indexed: bool
    no_validation_family_indexed: bool
    no_test_family_indexed: bool
    no_excluded_document_indexed: bool

    masked_index_text_matches_frozen_masking_policy: bool
    no_unmasked_identifier_recovered_via_masked_index: bool

    query_fingerprints_match_condition_registry: bool
    chunk_document_family_metadata_internally_consistent: bool

    rebuild_from_cache_reproduces_identical_corpus_fingerprint: bool
    rebuild_reproduces_identical_retrieval_ordering: bool
    determinism_caveat: str

    historical_rag_store_unchanged: bool

    notes: list[str] = Field(default_factory=list)


class CostRuntimeReport(BaseModel):
    version: str
    created_at: str
    total_embedding_api_requests: int
    total_tokens_billed: int
    total_retries: int
    total_failures: int
    total_wall_clock_seconds: float
    total_estimated_cost_usd: float
    cost_rate_is_placeholder: bool
    by_corpus: list[EmbeddingUsageReport]
