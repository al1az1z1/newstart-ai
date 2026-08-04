"""Caching, retrying Gemini embedding provider for the family-aware RAG layer (Version 6,
Checkpoint 9).

Kept fully separate from `newstart_ai.rag.embeddings.GeminiEmbeddingProvider` (the historical
routing index's provider) -- this module never touches the historical vector store. Reuses
`configs/rag.yaml`'s `embedding_provider`/`embedding_model` so the family-aware index can
never silently target a different embedding model than the historical one without an
explicit config change in one place.

Every embedded text is cached on disk keyed by (task_type, sha256(text)) so an interrupted
build can restart without re-paying for already-embedded texts, and a full rebuild from cache
alone (no API calls) reproduces the identical corpus.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone

import numpy as np

from newstart_ai.schemas.checkpoint9 import EmbeddingConfigManifest

# Placeholder per-million-token embedding rate for cost estimation only -- not billed against
# a real invoice, matching the existing placeholder convention in models/llm/provider.py.
GEMINI_EMBEDDING_COST_PER_MILLION_TOKENS_PLACEHOLDER = 0.15

# gemini-embedding-001 (Gemini Developer API, not Vertex/Enterprise mode) does not support
# server-side auto_truncate -- an over-limit request raises an error instead of truncating.
# Some "complete" condition queries are full document text (up to ~640k characters in this
# dataset), far beyond the model's ~2048-token input limit, so text is truncated client-side
# to this conservative character budget (~2048 tokens at a safe ~3 chars/token floor for
# dense/legal-style English text) before every embed call; truncation is recorded per-text.
MAX_EMBEDDING_INPUT_CHARACTERS = 6000

# Confirmed empirically (see Checkpoint 9 report): the Gemini Developer API's embed_content
# response never populates `embedding.statistics`/`response.metadata` in this SDK/API mode
# (unlike Vertex AI Enterprise mode), so no API-reported token count is available. Token
# usage is instead estimated from character count at this floor -- reported explicitly as an
# ESTIMATE, never presented as an API-confirmed figure.
CHARACTERS_PER_TOKEN_ESTIMATE = 4


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FamilyAwareGeminiEmbeddingProvider:
    def __init__(self, settings):
        import google.genai as genai

        self.settings = settings
        self.rag_cfg = settings.family_aware.rag
        self.model_name = settings.rag.embedding_model
        api_key = settings.llm.resolve_api_key()
        self.client = genai.Client(api_key=api_key)
        self.cache_dir = settings.resolve_path(self.rag_cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, task_type: str, text_hash: str):
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
        path = self._cache_path(task_type, text_hash)
        np.savez(path, vector=np.array(vector, dtype=np.float32), token_count=np.array(token_count))

    def embed_texts(self, texts: list[str], task_type: str) -> tuple[list[list[float]], dict]:
        """Returns (vectors, usage) where usage has cache_hits/cache_misses/api_requests/
        total_tokens/retries/failures/wall_clock_seconds. Every unique (task_type, text)
        pair is fetched from the API at most once across the lifetime of the cache dir."""
        from google.genai import types

        start_time = time.time()
        n = len(texts)
        vectors: list[list[float] | None] = [None] * n
        token_counts: list[int] = [0] * n
        hashes = [sha256_text(t) for t in texts]

        cache_hits = 0
        cache_misses = 0
        truncated_count = 0
        tokens_are_estimated = False
        to_fetch_indices: list[int] = []
        for i, h in enumerate(hashes):
            cached = self._load_cached(task_type, h)
            if cached is not None:
                vectors[i], token_counts[i] = cached
                cache_hits += 1
            else:
                to_fetch_indices.append(i)
                cache_misses += 1

        batch_size = self.rag_cfg.embedding_batch_size
        api_requests = 0
        retries = 0
        failures = 0

        config_kwargs = {"task_type": task_type}
        if self.rag_cfg.output_dimensionality is not None:
            config_kwargs["output_dimensionality"] = self.rag_cfg.output_dimensionality
        config = types.EmbedContentConfig(**config_kwargs)

        for start in range(0, len(to_fetch_indices), batch_size):
            batch_indices = to_fetch_indices[start : start + batch_size]
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
                            # API did not report token usage (Developer API mode) -- estimate
                            # from the actual (possibly truncated) text sent to the model.
                            tok = max(1, len(sent_text) // CHARACTERS_PER_TOKEN_ESTIMATE)
                            tokens_are_estimated = True
                        vectors[idx] = vec
                        token_counts[idx] = tok
                        self._save_cache(task_type, hashes[idx], vec, tok)
                    break
                except Exception as exc:
                    attempt += 1
                    if attempt > self.rag_cfg.max_retries:
                        failures += 1
                        raise RuntimeError(
                            f"Embedding batch failed after {attempt - 1} retries (task_type={task_type}): {exc}"
                        ) from exc
                    retries += 1
                    time.sleep(self.rag_cfg.retry_backoff_seconds * (2 ** (attempt - 1)))

        wall_clock = time.time() - start_time
        usage = {
            "total_texts_requested": n,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "api_requests_made": api_requests,
            "total_tokens_billed": sum(token_counts),
            "retries": retries,
            "failures": failures,
            "wall_clock_seconds": wall_clock,
            "truncated_count": truncated_count,
            "tokens_are_estimated": tokens_are_estimated,
        }
        return vectors, usage  # type: ignore[return-value]


def build_embedding_config_manifest(settings, sample_vectors: list[list[float]], configuration_fingerprint: str) -> EmbeddingConfigManifest:
    import importlib.metadata

    rag_cfg = settings.family_aware.rag
    arr = np.array(sample_vectors, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1)
    mean_norm = float(norms.mean())

    return EmbeddingConfigManifest(
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        provider=settings.rag.embedding_provider,
        model_name=settings.rag.embedding_model,
        document_task_type=rag_cfg.document_task_type,
        query_task_type=rag_cfg.query_task_type,
        configured_output_dimensionality=rag_cfg.output_dimensionality,
        observed_vector_dimension=int(arr.shape[1]),
        embedding_batch_size=rag_cfg.embedding_batch_size,
        observed_mean_l2_norm=mean_norm,
        observed_norm_is_unit_length=bool(abs(mean_norm - 1.0) < 0.01),
        genai_sdk_version=importlib.metadata.version("google-genai"),
        configuration_fingerprint=configuration_fingerprint,
        cache_dir=str(settings.resolve_path(rag_cfg.cache_dir)),
        notes=[
            "output_dimensionality is null (model default, no truncation) -- gemini-embedding-001's "
            "native dimensionality is used as-is.",
            "Normalization behavior was confirmed empirically (observed_mean_l2_norm), not assumed.",
        ],
    )
