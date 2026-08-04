"""Result schemas for Version 6 tokenizer-aware chunking with provenance (Checkpoint 5).

`ChunkRow` is the per-chunk record written to the family-aware chunk CSVs.
`FamilyAwareChunkManifest` is the versioned report proving every chunking invariant held.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkRow(BaseModel):
    chunk_id: str
    document_id: str
    effective_family_id: str
    agency: str
    effective_agency: str
    split: str
    chunk_index: int
    total_chunks: int
    tokenizer_name: str
    tokenizer_revision: str
    token_start: int
    token_end: int
    content_token_count: int
    encoded_sequence_length: int
    chunk_text: str
    chunk_text_hash: str
    parent_text_hash: str
    chunking_policy_version: str


class ChunkAgencyCounts(BaseModel):
    agency: str
    chunk_count: int


class ChunkCountDistribution(BaseModel):
    min: int
    p50_median: float
    mean: float
    p90: float
    p95: float
    max: int


class LargestDocumentByChunks(BaseModel):
    document_id: str
    agency: str
    effective_agency: str
    chunk_count: int


class SplitChunkCounts(BaseModel):
    split: str
    document_count: int
    family_count: int
    chunk_count: int
    chunks_by_agency: list[ChunkAgencyCounts] = Field(default_factory=list)
    chunks_per_document_distribution: ChunkCountDistribution
    single_chunk_document_count: int
    single_chunk_document_percentage: float
    multi_chunk_document_count: int
    multi_chunk_document_percentage: float
    largest_documents_by_chunk_count: list[LargestDocumentByChunks] = Field(default_factory=list)


class FamilyAwareChunkManifest(BaseModel):
    version: str
    created_at: str
    chunking_policy_version: str

    tokenizer_name: str
    tokenizer_revision: str
    tokenizer_resolved_commit_hash: str | None = None
    tokenizer_file_hashes: dict[str, str] = Field(default_factory=dict)
    max_seq_length: int
    num_special_tokens: int
    content_window_tokens: int
    chunk_overlap_tokens: int
    step_tokens: int

    source_split_fingerprints: dict[str, str]
    chunk_fingerprints: dict[str, str]

    splits: list[SplitChunkCounts]

    every_chunk_maps_to_one_eligible_parent: bool
    every_chunk_inherits_parent_split: bool
    zero_document_id_cross_split_leakage: bool
    zero_family_id_cross_split_leakage: bool
    zero_chunk_id_cross_split_leakage: bool
    every_eligible_document_has_at_least_one_chunk: bool
    no_excluded_document_produced_a_chunk: bool
    no_duplicate_chunk_ids: bool
    chunk_indices_contiguous_and_unique_per_document: bool
    reported_total_chunks_matches_actual_per_document: bool
    no_empty_or_special_token_only_chunks: bool
    rerun_produces_identical_chunks: bool

    documents_requiring_fallback_behavior: list[str] = Field(default_factory=list)

    notes: list[str] = Field(default_factory=list)
