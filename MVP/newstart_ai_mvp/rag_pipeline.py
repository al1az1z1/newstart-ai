"""Self-contained RAG embedding, indexing, and retrieval pipeline (Checkpoint 9).

A copy of the original project's family-aware embedding provider and Chroma index builder.
Manifests are plain dicts. The methodology (disk-cached embeddings keyed by
(task_type, sha256(text)), the 6,000-character client-side truncation, the candidate-pool +
diversification retrieval policy) is unchanged.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

UNMASKED_COLLECTION_NAME = "family_aware_routing_unmasked"
MASKED_COLLECTION_NAME = "family_aware_routing_masked"

# gemini-embedding-001 (Developer API) does not support server-side auto_truncate -- text is
# truncated client-side to this conservative character budget before every embed call. Shared
# with the LLM's own input truncation (llm_pipeline.truncate_for_llm) so both use one policy.
MAX_EMBEDDING_INPUT_CHARACTERS = 6000
CHARACTERS_PER_TOKEN_ESTIMATE = 4


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


class FamilyAwareGeminiEmbeddingProvider:
    """Caching, retrying Gemini embedding provider. Every embedded text is cached on disk
    keyed by (task_type, sha256(text)) so an interrupted build can restart without
    re-paying for already-embedded texts."""

    def __init__(self, settings):
        import google.genai as genai

        self.settings = settings
        self.rag_cfg = settings.family_aware.rag
        self.model_name = settings.rag.embedding_model
        api_key = settings.llm.resolve_api_key()
        self.client = genai.Client(api_key=api_key)
        self.cache_dir = settings.resolve_path(self.rag_cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, task_type: str, text_hash: str) -> Path:
        subdir = self.cache_dir / task_type
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{text_hash}.npz"

    def _load_cached(self, task_type: str, text_hash: str):
        path = self._cache_path(task_type, text_hash)
        if not path.exists():
            return None
        data = np.load(path)
        return data["vector"].tolist(), int(data["token_count"])

    def _save_cache(self, task_type: str, text_hash: str, vector: list[float], token_count: int) -> None:
        np.savez(self._cache_path(task_type, text_hash), vector=np.array(vector, dtype=np.float32), token_count=np.array(token_count))

    def embed_texts(self, texts: list[str], task_type: str) -> tuple[list[list[float]], dict]:
        from google.genai import types

        start_time = time.time()
        n = len(texts)
        vectors: list[list[float] | None] = [None] * n
        token_counts: list[int] = [0] * n
        hashes = [sha256_text(t) for t in texts]

        cache_hits = cache_misses = truncated_count = 0
        tokens_are_estimated = False
        to_fetch = []
        for i, h in enumerate(hashes):
            cached = self._load_cached(task_type, h)
            if cached is not None:
                vectors[i], token_counts[i] = cached
                cache_hits += 1
            else:
                to_fetch.append(i)
                cache_misses += 1

        batch_size = self.rag_cfg.embedding_batch_size
        api_requests = retries = failures = 0
        config_kwargs = {"task_type": task_type}
        if self.rag_cfg.output_dimensionality is not None:
            config_kwargs["output_dimensionality"] = self.rag_cfg.output_dimensionality
        config = types.EmbedContentConfig(**config_kwargs)

        for start in range(0, len(to_fetch), batch_size):
            batch_indices = to_fetch[start:start + batch_size]
            batch_texts = []
            for i in batch_indices:
                original = texts[i]
                if len(original) > MAX_EMBEDDING_INPUT_CHARACTERS:
                    truncated_count += 1
                batch_texts.append(original[:MAX_EMBEDDING_INPUT_CHARACTERS])

            attempt = 0
            while True:
                try:
                    response = self.client.models.embed_content(model=self.model_name, contents=batch_texts, config=config)
                    api_requests += 1
                    for idx, sent_text, embedding in zip(batch_indices, batch_texts, response.embeddings):
                        vec = list(embedding.values)
                        if embedding.statistics and embedding.statistics.token_count:
                            tok = int(embedding.statistics.token_count)
                        else:
                            tok = max(1, len(sent_text) // CHARACTERS_PER_TOKEN_ESTIMATE)
                            tokens_are_estimated = True
                        vectors[idx], token_counts[idx] = vec, tok
                        self._save_cache(task_type, hashes[idx], vec, tok)
                    break
                except Exception as exc:
                    attempt += 1
                    if attempt > self.rag_cfg.max_retries:
                        failures += 1
                        raise RuntimeError(f"Embedding batch failed after {attempt - 1} retries: {exc}") from exc
                    retries += 1
                    time.sleep(self.rag_cfg.retry_backoff_seconds * (2 ** (attempt - 1)))

        usage = {
            "total_texts_requested": n, "cache_hits": cache_hits, "cache_misses": cache_misses,
            "api_requests_made": api_requests, "total_tokens_billed": sum(token_counts), "retries": retries,
            "failures": failures, "wall_clock_seconds": time.time() - start_time,
            "truncated_count": truncated_count, "tokens_are_estimated": tokens_are_estimated,
        }
        return vectors, usage


def _get_client(settings, persist_dir_key: str):
    import chromadb

    persist_dir = settings.resolve_path(getattr(settings.family_aware.rag, persist_dir_key))
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def _corpus_fingerprint(chunks_df: pd.DataFrame, text_column: str) -> str:
    columns = ["chunk_id", "document_id", "effective_family_id", "effective_agency", text_column]
    ordered = chunks_df[columns].astype(str).sort_values("chunk_id").reset_index(drop=True)
    payload = "\n".join("|".join(row) for row in ordered.itertuples(index=False))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_family_aware_corpus_index(chunks_df: pd.DataFrame, text_column: str, text_hash_column: str, masked: bool, embedding_config_fingerprint: str, settings, embedding_provider=None) -> tuple[dict, dict]:
    """Builds one Chroma collection (unmasked or masked) from TRAIN-only chunks. If a
    collection with this name already exists at the target persist_dir, it is deleted and
    recreated -- callers that need to protect a frozen index must redirect persist_dir_key
    before calling this."""
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
            "document_id": str(row.document_id), "effective_family_id": str(row.effective_family_id),
            "agency": str(row.agency), "effective_agency": str(row.effective_agency), "split": str(row.split),
            "chunk_index": int(row.chunk_index), "total_chunks": int(row.total_chunks),
            "token_start": int(row.token_start), "token_end": int(row.token_end), "masked": bool(masked),
            "text_hash": str(getattr(row, text_hash_column)), "embedding_config_fingerprint": embedding_config_fingerprint,
        }
        for row in chunks_df.itertuples(index=False)
    ]
    collection.add(ids=ids, embeddings=vectors, metadatas=metadatas)

    corpus_manifest = {
        "version": "v1", "created_at": datetime.now(timezone.utc).isoformat(), "corpus_type": corpus_type,
        "indexed_chunk_count": len(ids), "indexed_document_count": int(chunks_df["document_id"].nunique()),
        "indexed_family_count": int(chunks_df["effective_family_id"].nunique()),
        "corpus_fingerprint": _corpus_fingerprint(chunks_df, text_column),
        "embedding_config_fingerprint": embedding_config_fingerprint,
        "persist_dir": str(settings.resolve_path(getattr(rag_cfg, persist_dir_key))), "collection_name": collection_name,
        "notes": [f"Built exclusively from the frozen family-aware TRAINING split's chunks ({text_column})."],
    }
    usage_report = {
        "version": "v1", "created_at": datetime.now(timezone.utc).isoformat(), "corpus": f"{corpus_type}_training_corpus",
        **usage,
        "estimated_cost_usd": round(usage["total_tokens_billed"] / 1_000_000 * 0.15, 6),
    }
    return corpus_manifest, usage_report


def load_family_aware_collection(settings, masked: bool):
    persist_dir_key = "persist_dir_masked" if masked else "persist_dir_unmasked"
    collection_name = MASKED_COLLECTION_NAME if masked else UNMASKED_COLLECTION_NAME
    return _get_client(settings, persist_dir_key).get_collection(collection_name)


def _seeded_tie_key(chunk_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{chunk_id}".encode("utf-8")).hexdigest()


def query_candidates(collection, query_vector: list[float], candidate_pool_size: int) -> list[dict]:
    """Returns up to candidate_pool_size nearest neighbors, deterministically ordered
    (similarity descending, seeded chunk_id hash ascending as tie-break)."""
    result = collection.query(query_embeddings=[query_vector], n_results=candidate_pool_size)
    if not result["ids"] or not result["ids"][0]:
        return []
    candidates = []
    for chunk_id, distance, metadata in zip(result["ids"][0], result["distances"][0], result["metadatas"][0]):
        candidates.append({
            "chunk_id": chunk_id, "similarity": 1 - distance, "parent_document_id": metadata["document_id"],
            "effective_family_id": metadata["effective_family_id"], "effective_agency": metadata["effective_agency"],
            "text_hash": metadata["text_hash"], "metadata": metadata,
        })
    candidates.sort(key=lambda c: (-c["similarity"], _seeded_tie_key(c["chunk_id"], "family_aware_rag_v1")))
    return candidates


def diversify_candidates(candidates: list[dict], top_k: int, max_chunks_per_parent_document: int, max_results_per_effective_family: int) -> list[dict]:
    """Greedily selects up to top_k candidates, skipping any that would exceed the
    per-parent-document or per-family cap, deduplicating exact-text-hash repeats. Returns
    fewer than top_k rather than padding with lower-quality/duplicate candidates."""
    selected: list[dict] = []
    parent_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    seen_text_hashes: set[str] = set()
    for candidate in candidates:
        if len(selected) >= top_k:
            break
        parent, family, text_hash = candidate["parent_document_id"], candidate["effective_family_id"], candidate["text_hash"]
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
    retrieval_cfg = settings.family_aware.rag.retrieval
    candidates = query_candidates(collection, query_vector, retrieval_cfg.candidate_pool_size)
    before = candidates[: retrieval_cfg.top_k]
    after = diversify_candidates(candidates, retrieval_cfg.top_k, retrieval_cfg.max_chunks_per_parent_document, retrieval_cfg.max_results_per_effective_family)
    return before, after


def build_diversification_policy_manifest(settings) -> dict:
    cfg = settings.family_aware.rag.retrieval
    return {
        "version": "v1", "created_at": datetime.now(timezone.utc).isoformat(), "policy_version": cfg.policy_version,
        "candidate_pool_size": cfg.candidate_pool_size, "top_k": cfg.top_k,
        "max_chunks_per_parent_document": cfg.max_chunks_per_parent_document,
        "max_results_per_effective_family": cfg.max_results_per_effective_family,
        "duplicate_handling": "Exact chunk text_hash duplicates are skipped after the first occurrence.",
        "fewer_than_k_behavior": "Fewer than top_k results are returned rather than padding with lower-similarity/duplicate candidates.",
    }


def evaluate_condition_retrieval(condition_name: str, query_rows: pd.DataFrame, true_label_by_doc: dict, unmasked_collection, masked_collection, embedding_provider, settings) -> dict:
    """Retrieval-only diagnostics (top-k agency hit rate, mean reciprocal rank) -- never
    classifier accuracy."""
    masked = condition_name.endswith("_masked")
    collection = masked_collection if masked else unmasked_collection
    hits, reciprocal_ranks = 0, []
    for row in query_rows.itertuples(index=False):
        vectors, _usage = embedding_provider.embed_texts([row.text], settings.family_aware.rag.query_task_type)
        _before, after = retrieve_diversified(collection, vectors[0], settings)
        true_agency = true_label_by_doc[row.document_id]
        rank = next((i + 1 for i, c in enumerate(after) if c["effective_agency"] == true_agency), None)
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    n = len(query_rows)
    return {
        "condition": condition_name, "index_used": "masked" if masked else "unmasked", "query_document_count": n,
        "top_k_agency_hit_rate": round(100 * hits / n, 2) if n else 0.0,
        "mean_reciprocal_rank": round(sum(reciprocal_ranks) / n, 4) if n else 0.0,
    }
