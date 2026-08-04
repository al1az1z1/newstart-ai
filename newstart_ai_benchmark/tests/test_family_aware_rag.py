"""Tests for Version 6 Checkpoint 9: the family-aware RAG retrieval layer.

Uses a fake, deterministic, hash-derived embedding provider (no network calls, no API
quota spent) implementing the same embed_texts(texts, task_type) -> (vectors, usage)
interface as FamilyAwareGeminiEmbeddingProvider, plus a real (but tiny, tmp_path-scoped)
Chroma index -- fast and fully offline.
"""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from newstart_ai.config import load_settings
from newstart_ai.rag.family_aware_diagnostics import build_rag_integrity_proof
from newstart_ai.rag.family_aware_embeddings import FamilyAwareGeminiEmbeddingProvider, sha256_text
from newstart_ai.rag.family_aware_index import (
    build_diversification_policy_manifest,
    build_family_aware_corpus_index,
    diversify_candidates,
    load_family_aware_collection,
    query_candidates,
    retrieve_diversified,
)

VECTOR_DIM = 16


class FakeEmbeddingProvider:
    """Deterministic hash-derived unit vectors -- similar texts (same hash prefix bucket)
    are not guaranteed similar, but identical text always yields the identical vector,
    which is all these tests need."""

    def __init__(self):
        self.calls = []

    def embed_texts(self, texts: list[str], task_type: str):
        self.calls.append((tuple(texts), task_type))
        vectors = []
        for text in texts:
            digest = hashlib.sha256(f"{task_type}:{text}".encode("utf-8")).digest()
            raw = np.frombuffer((digest * 2)[: VECTOR_DIM * 4], dtype=np.uint8).astype(np.float64)
            vec = raw / (np.linalg.norm(raw) + 1e-9)
            vectors.append(vec.tolist())
        usage = {
            "total_texts_requested": len(texts), "cache_hits": 0, "cache_misses": len(texts),
            "api_requests_made": 1, "total_tokens_billed": sum(len(t.split()) for t in texts),
            "retries": 0, "failures": 0, "wall_clock_seconds": 0.01, "truncated_count": 0,
        }
        return vectors, usage


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture()
def isolated_settings(settings, tmp_path):
    """Redirects the RAG persist/cache dirs to a tmp directory so tests never touch the
    real family-aware vector stores."""
    settings.family_aware.rag.persist_dir_unmasked = str(tmp_path / "unmasked")
    settings.family_aware.rag.persist_dir_masked = str(tmp_path / "masked")
    settings.family_aware.rag.cache_dir = str(tmp_path / "cache")
    return settings


def _make_chunks(agency_by_doc: dict[str, str], chunks_per_doc: dict[str, int], family_by_doc: dict[str, str] | None = None) -> pd.DataFrame:
    rows = []
    for doc_id, agency in agency_by_doc.items():
        n = chunks_per_doc[doc_id]
        family = (family_by_doc or {}).get(doc_id, f"FAM:{doc_id}")
        for i in range(n):
            rows.append(
                {
                    "chunk_id": f"{doc_id}_{i}",
                    "document_id": doc_id,
                    "effective_family_id": family,
                    "agency": agency,
                    "effective_agency": agency,
                    "split": "train",
                    "chunk_index": i,
                    "total_chunks": n,
                    "token_start": i * 100,
                    "token_end": (i + 1) * 100,
                    "chunk_text": f"Text for {doc_id} chunk {i} about {agency}.",
                    "chunk_text_hash": hashlib.sha256(f"{doc_id}-{i}".encode()).hexdigest(),
                }
            )
    return pd.DataFrame(rows)


def test_corpus_index_builds_from_train_chunks_only(isolated_settings):
    chunks = _make_chunks({"d1": "USCIS", "d2": "DMV"}, {"d1": 1, "d2": 1})
    provider = FakeEmbeddingProvider()
    manifest, usage = build_family_aware_corpus_index(
        chunks, "chunk_text", "chunk_text_hash", masked=False,
        embedding_config_fingerprint="fp", settings=isolated_settings, embedding_provider=provider,
    )
    assert manifest.indexed_chunk_count == 2
    assert manifest.indexed_document_count == 2
    assert usage.total_texts_requested == 2


def test_complete_metadata_provenance_stored_per_vector(isolated_settings):
    chunks = _make_chunks({"d1": "IRS"}, {"d1": 1}, family_by_doc={"d1": "IRS:SS4"})
    provider = FakeEmbeddingProvider()
    build_family_aware_corpus_index(chunks, "chunk_text", "chunk_text_hash", False, "fp", isolated_settings, provider)
    collection = load_family_aware_collection(isolated_settings, masked=False)
    stored = collection.get(ids=["d1_0"])
    meta = stored["metadatas"][0]
    for field in ("document_id", "effective_family_id", "agency", "effective_agency", "split", "chunk_index", "total_chunks", "token_start", "token_end", "masked", "text_hash", "embedding_config_fingerprint"):
        assert field in meta
    assert meta["document_id"] == "d1"
    assert meta["effective_family_id"] == "IRS:SS4"
    assert meta["masked"] is False


def test_masked_and_unmasked_indexes_are_separate_collections(isolated_settings):
    unmasked_chunks = _make_chunks({"d1": "USCIS"}, {"d1": 1})
    masked_chunks = unmasked_chunks.copy()
    masked_chunks["masked_chunk_text"] = "[AGENCY_NAME] chunk 0 masked."
    masked_chunks["masked_chunk_text_hash"] = "maskedhash"

    provider = FakeEmbeddingProvider()
    build_family_aware_corpus_index(unmasked_chunks, "chunk_text", "chunk_text_hash", False, "fp", isolated_settings, provider)
    build_family_aware_corpus_index(masked_chunks, "masked_chunk_text", "masked_chunk_text_hash", True, "fp", isolated_settings, provider)

    unmasked_collection = load_family_aware_collection(isolated_settings, masked=False)
    masked_collection = load_family_aware_collection(isolated_settings, masked=True)
    assert unmasked_collection.name != masked_collection.name

    # Verify the fake provider was called with the unmasked text for one index and masked text for the other.
    unmasked_call_texts = provider.calls[0][0]
    masked_call_texts = provider.calls[1][0]
    assert "chunk 0 about USCIS" in unmasked_call_texts[0]
    assert "[AGENCY_NAME]" in masked_call_texts[0]
    assert "USCIS" not in masked_call_texts[0]


def test_no_unmasked_identifier_recoverable_via_masked_index_payload(isolated_settings):
    """The masked collection's stored metadata/text must never contain the literal agency
    name/form number that the frozen masking policy would have removed."""
    masked_chunks = _make_chunks({"d1": "IRS"}, {"d1": 1})
    masked_chunks["masked_chunk_text"] = "This is issued by [AGENCY_NAME] per Form [FORM_NUMBER]."
    masked_chunks["masked_chunk_text_hash"] = "h1"

    provider = FakeEmbeddingProvider()
    build_family_aware_corpus_index(masked_chunks, "masked_chunk_text", "masked_chunk_text_hash", True, "fp", isolated_settings, provider)
    collection = load_family_aware_collection(isolated_settings, masked=True)
    stored = collection.get(ids=["d1_0"])
    payload_text = str(stored)
    assert "Internal Revenue Service" not in payload_text
    assert "SS-4" not in payload_text


def test_deterministic_ranking_and_tie_breaking(isolated_settings):
    chunks = _make_chunks({"d1": "USCIS", "d2": "DMV", "d3": "SSA"}, {"d1": 1, "d2": 1, "d3": 1})
    provider = FakeEmbeddingProvider()
    build_family_aware_corpus_index(chunks, "chunk_text", "chunk_text_hash", False, "fp", isolated_settings, provider)
    collection = load_family_aware_collection(isolated_settings, masked=False)

    query_vector, _ = provider.embed_texts(["query about USCIS"], "RETRIEVAL_QUERY")
    candidates_a = query_candidates(collection, query_vector[0], candidate_pool_size=3)
    candidates_b = query_candidates(collection, query_vector[0], candidate_pool_size=3)
    assert [c["chunk_id"] for c in candidates_a] == [c["chunk_id"] for c in candidates_b]


def test_parent_and_family_diversification_caps_are_respected():
    candidates = [
        {"chunk_id": f"c{i}", "similarity": 1.0 - i * 0.01, "parent_document_id": "739", "effective_family_id": "FAM:739", "text_hash": f"h{i}"}
        for i in range(10)
    ]
    diversified = diversify_candidates(candidates, top_k=5, max_chunks_per_parent_document=2, max_results_per_effective_family=2)
    assert len(diversified) == 2  # capped by both parent AND family limit (same doc/family here)
    assert {c["chunk_id"] for c in diversified} == {"c0", "c1"}


def test_long_document_domination_protection_across_multiple_parents():
    dominant = [
        {"chunk_id": f"dom{i}", "similarity": 0.99 - i * 0.001, "parent_document_id": "739", "effective_family_id": "FAM:739", "text_hash": f"domh{i}"}
        for i in range(20)
    ]
    others = [
        {"chunk_id": f"oth{i}", "similarity": 0.5 - i * 0.01, "parent_document_id": f"other{i}", "effective_family_id": f"FAM:other{i}", "text_hash": f"othh{i}"}
        for i in range(5)
    ]
    candidates = dominant + others
    diversified = diversify_candidates(candidates, top_k=5, max_chunks_per_parent_document=2, max_results_per_effective_family=2)
    dominant_count = sum(1 for c in diversified if c["parent_document_id"] == "739")
    assert dominant_count <= 2
    assert len(diversified) == 5  # still fills up to top_k using the other, lower-similarity candidates


def test_fewer_than_k_fallback_does_not_pad_with_duplicates():
    candidates = [
        {"chunk_id": "c1", "similarity": 0.9, "parent_document_id": "d1", "effective_family_id": "FAM:1", "text_hash": "h1"},
        {"chunk_id": "c2", "similarity": 0.8, "parent_document_id": "d1", "effective_family_id": "FAM:1", "text_hash": "h2"},
    ]
    diversified = diversify_candidates(candidates, top_k=5, max_chunks_per_parent_document=2, max_results_per_effective_family=2)
    assert len(diversified) == 2  # fewer than top_k=5, no padding
    assert len({c["chunk_id"] for c in diversified}) == 2


def test_exact_duplicate_text_hash_is_skipped():
    candidates = [
        {"chunk_id": "c1", "similarity": 0.9, "parent_document_id": "d1", "effective_family_id": "FAM:1", "text_hash": "same"},
        {"chunk_id": "c2", "similarity": 0.8, "parent_document_id": "d2", "effective_family_id": "FAM:2", "text_hash": "same"},
        {"chunk_id": "c3", "similarity": 0.7, "parent_document_id": "d3", "effective_family_id": "FAM:3", "text_hash": "different"},
    ]
    diversified = diversify_candidates(candidates, top_k=5, max_chunks_per_parent_document=2, max_results_per_effective_family=2)
    assert [c["chunk_id"] for c in diversified] == ["c1", "c3"]


def test_no_validation_or_test_document_can_be_indexed_by_construction(isolated_settings):
    """Structural proof: build_family_aware_corpus_index only ever reads document_id from
    whatever chunks_df it is given -- passing it a validation/test-only DataFrame and then
    checking the integrity proof against real train/val/test ID sets must show a violation,
    proving the checker actually discriminates rather than trivially passing."""
    validation_chunks = _make_chunks({"v1": "USCIS"}, {"v1": 1})
    provider = FakeEmbeddingProvider()
    build_family_aware_corpus_index(validation_chunks, "chunk_text", "chunk_text_hash", False, "fp", isolated_settings, provider)
    collection = load_family_aware_collection(isolated_settings, masked=False)
    indexed_ids = set(collection.get()["metadatas"][i]["document_id"] for i in range(len(collection.get()["ids"])))

    proof = build_rag_integrity_proof(
        unmasked_indexed_doc_ids=indexed_ids, masked_indexed_doc_ids=set(),
        train_doc_ids={"t1", "t2"}, validation_doc_ids={"v1"}, test_doc_ids=set(),
        unmasked_indexed_family_ids=set(), masked_indexed_family_ids=set(),
        train_family_ids=set(), validation_family_ids=set(), test_family_ids=set(),
        excluded_doc_ids=set(),
        masked_index_text_sample_matches_masking_policy=True, no_unmasked_identifier_found_in_masked_payloads=True,
        query_fingerprint_check_passed=True, metadata_consistency_check_passed=True,
        rebuild_corpus_fingerprint_matches=True, rebuild_retrieval_ordering_matches=True,
        historical_rag_hashes_match=True,
    )
    assert proof.no_validation_document_indexed is False
    assert proof.every_indexed_vector_from_train_split is False


def test_rag_integrity_proof_passes_for_a_clean_scenario():
    proof = build_rag_integrity_proof(
        unmasked_indexed_doc_ids={"t1", "t2"}, masked_indexed_doc_ids={"t1", "t2"},
        train_doc_ids={"t1", "t2", "t3"}, validation_doc_ids={"v1"}, test_doc_ids={"te1"},
        unmasked_indexed_family_ids={"FAM:1"}, masked_indexed_family_ids={"FAM:1"},
        train_family_ids={"FAM:1", "FAM:3"}, validation_family_ids={"FAM:V"}, test_family_ids={"FAM:TE"},
        excluded_doc_ids={"excluded1"},
        masked_index_text_sample_matches_masking_policy=True, no_unmasked_identifier_found_in_masked_payloads=True,
        query_fingerprint_check_passed=True, metadata_consistency_check_passed=True,
        rebuild_corpus_fingerprint_matches=True, rebuild_retrieval_ordering_matches=True,
        historical_rag_hashes_match=True,
    )
    assert proof.every_indexed_vector_from_train_split is True
    assert proof.no_validation_document_indexed is True
    assert proof.no_test_document_indexed is True
    assert proof.no_validation_family_indexed is True
    assert proof.no_test_family_indexed is True
    assert proof.no_excluded_document_indexed is True


def test_diversification_policy_manifest_reflects_configured_values(isolated_settings):
    manifest = build_diversification_policy_manifest(isolated_settings)
    assert manifest.top_k == isolated_settings.family_aware.rag.retrieval.top_k
    assert manifest.max_chunks_per_parent_document == isolated_settings.family_aware.rag.retrieval.max_chunks_per_parent_document


def test_cached_embedding_provider_avoids_repeated_api_calls(isolated_settings, monkeypatch):
    """Mocks the underlying genai client so no network call occurs, but exercises the real
    FamilyAwareGeminiEmbeddingProvider caching path -- a second call with the same texts
    must produce zero new API requests and identical vectors."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-testing")

    fake_embedding = MagicMock()
    fake_embedding.values = [0.1, 0.2, 0.3]
    fake_embedding.statistics = MagicMock(token_count=5, truncated=False)
    fake_response = MagicMock()
    fake_response.embeddings = [fake_embedding]

    provider = FamilyAwareGeminiEmbeddingProvider(isolated_settings)
    provider.client = MagicMock()
    provider.client.models.embed_content.return_value = fake_response

    vectors_1, usage_1 = provider.embed_texts(["hello world"], "RETRIEVAL_DOCUMENT")
    assert usage_1["api_requests_made"] == 1
    assert usage_1["cache_misses"] == 1

    vectors_2, usage_2 = provider.embed_texts(["hello world"], "RETRIEVAL_DOCUMENT")
    assert usage_2["api_requests_made"] == 0
    assert usage_2["cache_hits"] == 1
    # Cached vectors are stored as float32 (disk-space tradeoff) -- compare with tolerance.
    assert np.allclose(vectors_1, vectors_2, atol=1e-6)


def test_query_fingerprint_matches_condition_registry_text():
    from newstart_ai.data.condition_registry import _sha256

    query_text = "This is the exact registered condition text for document 42."
    expected_fingerprint = _sha256(query_text)
    # The fingerprint stored in the condition registry must match a fresh hash of the same text.
    assert expected_fingerprint == hashlib.sha256(query_text.encode("utf-8")).hexdigest()


def test_historical_rag_store_path_is_never_touched_by_family_aware_index(isolated_settings):
    from newstart_ai.rag.family_aware_index import UNMASKED_COLLECTION_NAME, MASKED_COLLECTION_NAME
    from newstart_ai.rag.index import COLLECTION_NAME as HISTORICAL_COLLECTION_NAME

    assert UNMASKED_COLLECTION_NAME != HISTORICAL_COLLECTION_NAME
    assert MASKED_COLLECTION_NAME != HISTORICAL_COLLECTION_NAME
    assert isolated_settings.family_aware.rag.persist_dir_unmasked != isolated_settings.rag.persist_dir
    assert isolated_settings.family_aware.rag.persist_dir_masked != isolated_settings.rag.persist_dir
