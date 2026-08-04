"""Dual (unmasked/masked) family-aware RAG vector index (Version 6, Checkpoint 9).

Two entirely separate Chroma collections, built only from the frozen family-aware TRAINING
chunks -- one from unmasked `chunk_text`, one from masked `masked_chunk_text` -- so a query
against the masked index can never retrieve unmasked identifier text. Both share the same
eligible training documents, chunk identities, embedding model, and retrieval policy; the
only difference is the frozen masking transformation.

Entirely separate from `newstart_ai.rag.index` (the historical routing index) -- different
persist directories, different collection names, never touches the historical Chroma store.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import chromadb
import pandas as pd

from newstart_ai.rag.family_aware_embeddings import (
    GEMINI_EMBEDDING_COST_PER_MILLION_TOKENS_PLACEHOLDER,
    FamilyAwareGeminiEmbeddingProvider,
    sha256_text,
)
from newstart_ai.schemas.checkpoint9 import CorpusManifest, DiversificationPolicyManifest, EmbeddingUsageReport

UNMASKED_COLLECTION_NAME = "family_aware_routing_unmasked"
MASKED_COLLECTION_NAME = "family_aware_routing_masked"


def _get_client(settings, persist_dir_key: str) -> chromadb.ClientAPI:
    persist_dir = settings.resolve_path(getattr(settings.family_aware.rag, persist_dir_key))
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def _sanitize_metadata(value):
    return value if value is not None else ""


def _corpus_fingerprint(chunks_df: pd.DataFrame, text_column: str) -> str:
    columns = ["chunk_id", "document_id", "effective_family_id", "effective_agency", text_column]
    ordered = chunks_df[columns].astype(str).sort_values("chunk_id").reset_index(drop=True)
    payload = "\n".join("|".join(row) for row in ordered.itertuples(index=False))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_family_aware_corpus_index(
    chunks_df: pd.DataFrame,
    text_column: str,
    text_hash_column: str,
    masked: bool,
    embedding_config_fingerprint: str,
    settings,
    embedding_provider: FamilyAwareGeminiEmbeddingProvider | None = None,
) -> tuple[CorpusManifest, EmbeddingUsageReport]:
    """Builds one Chroma collection (unmasked or masked) from `chunks_df` (already merged
    with document_id/effective_family_id/effective_agency/split/chunk_index/total_chunks/
    token_start/token_end -- must be the frozen TRAIN split's chunks only, never validation
    or test)."""
    rag_cfg = settings.family_aware.rag
    embedding_provider = embedding_provider or FamilyAwareGeminiEmbeddingProvider(settings)

    persist_dir_key = "persist_dir_masked" if masked else "persist_dir_unmasked"
    collection_name = MASKED_COLLECTION_NAME if masked else UNMASKED_COLLECTION_NAME
    corpus_type = "masked" if masked else "unmasked"

    client = _get_client(settings, persist_dir_key)
    existing = {c.name for c in client.list_collections()}
    if collection_name in existing:
        client.delete_collection(collection_name)
    collection = client.create_collection(collection_name, metadata={"hnsw:space": rag_cfg.retrieval.similarity_metric})

    texts = chunks_df[text_column].astype(str).tolist()
    vectors, usage = embedding_provider.embed_texts(texts, rag_cfg.document_task_type)

    ids = chunks_df["chunk_id"].astype(str).tolist()
    metadatas = [
        {
            "document_id": _sanitize_metadata(str(row.document_id)),
            "effective_family_id": _sanitize_metadata(str(row.effective_family_id)),
            "agency": _sanitize_metadata(str(row.agency)),
            "effective_agency": _sanitize_metadata(str(row.effective_agency)),
            "split": _sanitize_metadata(str(row.split)),
            "chunk_index": int(row.chunk_index),
            "total_chunks": int(row.total_chunks),
            "token_start": int(row.token_start),
            "token_end": int(row.token_end),
            "masked": bool(masked),
            "text_hash": _sanitize_metadata(str(getattr(row, text_hash_column))),
            "embedding_config_fingerprint": embedding_config_fingerprint,
        }
        for row in chunks_df.itertuples(index=False)
    ]
    collection.add(ids=ids, embeddings=vectors, metadatas=metadatas)

    corpus_manifest = CorpusManifest(
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        corpus_type=corpus_type,
        indexed_chunk_count=len(ids),
        indexed_document_count=int(chunks_df["document_id"].nunique()),
        indexed_family_count=int(chunks_df["effective_family_id"].nunique()),
        corpus_fingerprint=_corpus_fingerprint(chunks_df, text_column),
        embedding_config_fingerprint=embedding_config_fingerprint,
        source_train_chunk_fingerprint="",  # filled by caller (fingerprint_chunks(train_chunks))
        persist_dir=str(settings.resolve_path(getattr(rag_cfg, persist_dir_key))),
        collection_name=collection_name,
        notes=[
            f"Built exclusively from the frozen family-aware TRAINING split's chunks ({text_column})."
        ],
    )

    usage_report = EmbeddingUsageReport(
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        corpus=f"{corpus_type}_training_corpus",
        total_texts_requested=usage["total_texts_requested"],
        cache_hits=usage["cache_hits"],
        cache_misses=usage["cache_misses"],
        api_requests_made=usage["api_requests_made"],
        total_tokens_billed=usage["total_tokens_billed"],
        tokens_are_estimated=usage.get("tokens_are_estimated", False),
        retries=usage["retries"],
        failures=usage["failures"],
        wall_clock_seconds=usage["wall_clock_seconds"],
        estimated_cost_usd=round(usage["total_tokens_billed"] / 1_000_000 * GEMINI_EMBEDDING_COST_PER_MILLION_TOKENS_PLACEHOLDER, 6),
        cost_rate_per_million_tokens_usd=GEMINI_EMBEDDING_COST_PER_MILLION_TOKENS_PLACEHOLDER,
        cost_rate_is_placeholder=True,
        notes=[
            "Cost is a placeholder estimate based on a documented per-million-token rate, not a real invoice.",
            "Token counts are ESTIMATED from character count (Gemini Developer API does not "
            "return embedding.statistics.token_count in this SDK/API mode) -- not API-confirmed."
            if usage.get("tokens_are_estimated")
            else "Token counts are API-confirmed via embedding.statistics.token_count.",
        ],
    )

    return corpus_manifest, usage_report


def load_family_aware_collection(settings, masked: bool):
    persist_dir_key = "persist_dir_masked" if masked else "persist_dir_unmasked"
    collection_name = MASKED_COLLECTION_NAME if masked else UNMASKED_COLLECTION_NAME
    return _get_client(settings, persist_dir_key).get_collection(collection_name)


def _seeded_tie_key(chunk_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{chunk_id}".encode("utf-8")).hexdigest()


def query_candidates(collection, query_vector: list[float], candidate_pool_size: int) -> list[dict]:
    """Returns up to `candidate_pool_size` nearest neighbors, deterministically ordered
    (similarity descending, seeded chunk_id hash ascending as tie-break)."""
    result = collection.query(query_embeddings=[query_vector], n_results=candidate_pool_size)
    if not result["ids"] or not result["ids"][0]:
        return []
    candidates = []
    for chunk_id, distance, metadata in zip(result["ids"][0], result["distances"][0], result["metadatas"][0]):
        candidates.append(
            {
                "chunk_id": chunk_id,
                "similarity": 1 - distance,
                "parent_document_id": metadata["document_id"],
                "effective_family_id": metadata["effective_family_id"],
                "effective_agency": metadata["effective_agency"],
                "text_hash": metadata["text_hash"],
                "metadata": metadata,
            }
        )
    candidates.sort(key=lambda c: (-c["similarity"], _seeded_tie_key(c["chunk_id"], "family_aware_rag_v1")))
    return candidates


def diversify_candidates(
    candidates: list[dict],
    top_k: int,
    max_chunks_per_parent_document: int,
    max_results_per_effective_family: int,
) -> list[dict]:
    """Greedily selects up to `top_k` candidates (already deterministically ordered),
    skipping any that would exceed the per-parent-document or per-family cap, and
    deduplicating exact-text-hash repeats. Returns fewer than `top_k` results (rather than
    padding with lower-quality/duplicate candidates) if the diversified pool is exhausted --
    this is the documented, intentional fewer-than-k fallback behavior."""
    selected: list[dict] = []
    parent_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    seen_text_hashes: set[str] = set()

    for candidate in candidates:
        if len(selected) >= top_k:
            break
        parent = candidate["parent_document_id"]
        family = candidate["effective_family_id"]
        text_hash = candidate["text_hash"]

        if parent_counts.get(parent, 0) >= max_chunks_per_parent_document:
            continue
        if family_counts.get(family, 0) >= max_results_per_effective_family:
            continue
        if text_hash in seen_text_hashes:
            continue

        selected.append(candidate)
        parent_counts[parent] = parent_counts.get(parent, 0) + 1
        family_counts[family] = family_counts.get(family, 0) + 1
        seen_text_hashes.add(text_hash)

    return selected


def retrieve_diversified(collection, query_vector: list[float], settings) -> tuple[list[dict], list[dict]]:
    """Returns (before, after) -- `before` is the top_k slice of the raw similarity-ranked
    candidate pool (no diversification), `after` is the diversified top_k -- so the effect of
    diversification is directly visible."""
    retrieval_cfg = settings.family_aware.rag.retrieval
    candidates = query_candidates(collection, query_vector, retrieval_cfg.candidate_pool_size)
    before = candidates[: retrieval_cfg.top_k]
    after = diversify_candidates(
        candidates, retrieval_cfg.top_k, retrieval_cfg.max_chunks_per_parent_document, retrieval_cfg.max_results_per_effective_family
    )
    return before, after


def build_diversification_policy_manifest(settings) -> DiversificationPolicyManifest:
    cfg = settings.family_aware.rag.retrieval
    return DiversificationPolicyManifest(
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        policy_version=cfg.policy_version,
        candidate_pool_size=cfg.candidate_pool_size,
        top_k=cfg.top_k,
        max_chunks_per_parent_document=cfg.max_chunks_per_parent_document,
        max_results_per_effective_family=cfg.max_results_per_effective_family,
        minimum_similarity_threshold=cfg.minimum_similarity_threshold,
        tie_breaker=cfg.tie_breaker,
        duplicate_handling="Exact chunk text_hash duplicates are skipped after the first occurrence within a query's diversified result set.",
        fewer_than_k_behavior=(
            "If the per-parent/per-family caps exhaust the candidate pool before reaching "
            "top_k, fewer than top_k results are returned rather than padding with "
            "lower-similarity or duplicate candidates."
        ),
        notes=[
            "Frozen using validation retrieval diagnostics only -- never test data.",
        ],
    )
