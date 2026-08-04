"""One-time frozen head-to-head test evaluation: plain Gemini LLM and Gemini LLM+RAG case
runners (Version 6, Checkpoint 10).

Every case is cached atomically to disk keyed by (method, model, prompt_version,
document_id, condition_fingerprint, retrieval_context_fingerprint) so an interrupted
evaluation can resume without ever re-issuing a paid call for an already-completed case.

Failure policy (frozen before any call): only transient transport/timeout/rate-limit/server
errors are retried, up to `max_attempts`. A parsed-but-invalid label or JSON parse failure is
recorded as `status="invalid"`, never retried in hope of a different answer. No schema-repair
retry is attempted, because that behavior does not already exist in the frozen
`GeminiProvider._classify_rendered` this module wraps.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from newstart_ai.models.llm.provider import GEMINI_INPUT_COST_PER_MILLION_TOKENS, GEMINI_OUTPUT_COST_PER_MILLION_TOKENS
from newstart_ai.rag.family_aware_embeddings import MAX_EMBEDDING_INPUT_CHARACTERS
from newstart_ai.schemas.checkpoint10 import CaseResult, Checkpoint10FreezeRecord, RetrievedChunkProvenance

PARSER_VERSION = "json_predicted_label_v1"  # parse response.text as JSON, extract
# "predicted_label", validate against prompt.allowed_labels -- raises ValueError (-> status
# "invalid") on parse failure or an out-of-schema label; unchanged since Phase 1.


def sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def truncate_for_llm(text: str) -> tuple[str, bool]:
    """Reuses the already-frozen 6,000-character truncation policy (Checkpoint 9) -- no new
    truncation policy is introduced for this checkpoint."""
    if len(text) > MAX_EMBEDDING_INPUT_CHARACTERS:
        return text[:MAX_EMBEDDING_INPUT_CHARACTERS], True
    return text, False


def compute_cache_key(method: str, model: str, prompt_version: str, document_id: str, condition: str, condition_fingerprint: str, retrieval_context_fingerprint: str | None) -> str:
    """`condition` is included explicitly and separately from `condition_fingerprint`: for
    short/single-chunk test documents, multiple conditions (e.g. beginning_only, middle_only,
    end_only, and even complete) can legitimately share byte-identical registered text --
    and therefore an identical content fingerprint -- while still being distinct
    (document_id, condition) cases that must each get their own cache entry. Keying only on
    the content fingerprint caused exactly this collision in an earlier run (167 of 990
    plain-LLM cases and 121 of 990 LLM+RAG cases silently shared a cache slot); this was
    found and fixed before computing any metric from those results, and the affected
    evaluation was fully rerun under the corrected key."""
    payload = f"{method}|{model}|{prompt_version}|{document_id}|{condition}|{condition_fingerprint}|{retrieval_context_fingerprint or ''}"
    return sha256_str(payload)


def _cache_path(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / f"{cache_key}.json"


def _load_cached_case(cache_dir: Path, cache_key: str) -> CaseResult | None:
    path = _cache_path(cache_dir, cache_key)
    if not path.exists():
        return None
    return CaseResult.model_validate_json(path.read_text(encoding="utf-8"))


def _save_case_atomic(cache_dir: Path, case: CaseResult) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, case.cache_key)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(case.model_dump_json(indent=2), encoding="utf-8")
    tmp_path.replace(path)  # atomic on POSIX and Windows (same filesystem)


def _is_retryable(exc: Exception, retryable_substrings: list[str]) -> bool:
    message = str(exc).lower()
    return any(s.lower() in message for s in retryable_substrings)


def run_plain_llm_case(
    document_id: str,
    effective_family_id: str,
    condition: str,
    true_label: str,
    text: str,
    condition_fingerprint: str,
    llm_provider,
    prompt,
    settings,
) -> CaseResult:
    """Runs (or reuses a cached) plain-LLM classification for one (document, condition)."""
    eval_cfg = settings.family_aware.evaluation
    cache_dir = settings.resolve_path(eval_cfg.cache_dir) / "llm"
    cache_key = compute_cache_key("llm", llm_provider.model_name, prompt.version, document_id, condition, condition_fingerprint, None)

    cached = _load_cached_case(cache_dir, cache_key)
    if cached is not None:
        return cached

    truncated_text, was_truncated = truncate_for_llm(text)

    attempt = 0
    last_exc: Exception | None = None
    while attempt < eval_cfg.max_attempts:
        attempt += 1
        try:
            result = llm_provider.classify(
                text=truncated_text, document_id=document_id, prompt=prompt, method="llm",
            )
            case = CaseResult(
                method="llm", document_id=document_id, effective_family_id=effective_family_id,
                condition=condition, true_label=true_label, input_fingerprint=condition_fingerprint,
                retrieval_context_fingerprint=None, predicted_label=result.predicted_label,
                raw_response_hash=sha256_str(json.dumps(result.model_dump(), sort_keys=True, default=str)),
                status="success", attempt_count=attempt, truncated=was_truncated,
                latency_ms=result.latency_ms,
                prompt_tokens=result.token_usage.prompt_tokens if result.token_usage else None,
                completion_tokens=result.token_usage.completion_tokens if result.token_usage else None,
                total_tokens=result.token_usage.total_tokens if result.token_usage else None,
                estimated_cost_usd=result.estimated_cost,
                cache_key=cache_key,
            )
            _save_case_atomic(cache_dir, case)
            return case
        except ValueError as exc:
            # Invalid/out-of-schema label from _classify_rendered -- never retried.
            case = CaseResult(
                method="llm", document_id=document_id, effective_family_id=effective_family_id,
                condition=condition, true_label=true_label, input_fingerprint=condition_fingerprint,
                retrieval_context_fingerprint=None, predicted_label=None, raw_response_hash=None,
                status="invalid", error_type=type(exc).__name__, attempt_count=attempt,
                truncated=was_truncated, latency_ms=0.0, cache_key=cache_key,
            )
            _save_case_atomic(cache_dir, case)
            return case
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc, eval_cfg.retryable_error_substrings) or attempt >= eval_cfg.max_attempts:
                break
            time.sleep(eval_cfg.retry_backoff_seconds * (2 ** (attempt - 1)))

    case = CaseResult(
        method="llm", document_id=document_id, effective_family_id=effective_family_id,
        condition=condition, true_label=true_label, input_fingerprint=condition_fingerprint,
        retrieval_context_fingerprint=None, predicted_label=None, raw_response_hash=None,
        status="failed", error_type=type(last_exc).__name__ if last_exc else "unknown",
        attempt_count=attempt, truncated=was_truncated, latency_ms=0.0, cache_key=cache_key,
    )
    _save_case_atomic(cache_dir, case)
    return case


def format_context_no_labels(retrieved: list[dict]) -> str:
    """Formats retrieved chunks as untrusted reference TEXT ONLY -- no agency, document_id,
    family, filename, or any other metadata is ever included in what Gemini sees."""
    if not retrieved:
        return "(no reference excerpts retrieved)"
    lines = [f"Excerpt {i}:\n{chunk['text']}" for i, chunk in enumerate(retrieved, start=1)]
    return "\n\n".join(lines)


def run_llm_rag_case(
    document_id: str,
    effective_family_id: str,
    condition: str,
    true_label: str,
    text: str,
    condition_fingerprint: str,
    masked: bool,
    unmasked_collection,
    masked_collection,
    chunk_text_by_id: dict[str, str],
    embedding_provider,
    llm_provider,
    prompt,
    settings,
) -> CaseResult:
    """Runs (or reuses a cached) LLM+RAG classification for one (document, condition).

    Routes unmasked conditions to the unmasked index and masked conditions to the masked
    index. The classification prompt only ever sees retrieved chunk TEXT -- never
    effective_agency, document_id, family, or any other metadata (see
    `format_context_no_labels`). `chunk_text_by_id` resolves retrieved chunk_ids to their
    text (Chroma stores only text_hash + structural metadata, never raw text -- see
    Checkpoint 9) using the SAME masked/unmasked source the retrieved chunk came from.
    """
    from newstart_ai.rag.family_aware_index import retrieve_diversified

    eval_cfg = settings.family_aware.evaluation
    cache_dir = settings.resolve_path(eval_cfg.cache_dir) / "llm_rag"

    truncated_query_text, query_was_truncated = truncate_for_llm(text)

    query_vectors, _query_usage = embedding_provider.embed_texts([truncated_query_text], settings.family_aware.rag.query_task_type)
    query_vector = query_vectors[0]

    collection = masked_collection if masked else unmasked_collection
    _before, after = retrieve_diversified(collection, query_vector, settings)

    retrieved_provenance = [
        RetrievedChunkProvenance(
            chunk_id=r["chunk_id"], rank=rank, similarity=r["similarity"],
            parent_document_id=r["parent_document_id"], effective_family_id=r["effective_family_id"],
            effective_agency=r["effective_agency"], text_hash=r["text_hash"], masked=masked,
        )
        for rank, r in enumerate(after, start=1)
    ]

    retrieval_context_fingerprint = sha256_str(
        "|".join(f"{p.chunk_id}:{p.text_hash}" for p in retrieved_provenance)
    )
    cache_key = compute_cache_key(
        "llm_rag", llm_provider.model_name, prompt.version, document_id, condition, condition_fingerprint, retrieval_context_fingerprint
    )

    cached = _load_cached_case(cache_dir, cache_key)
    if cached is not None:
        return cached

    retrieved_dicts = [{"text": chunk_text_by_id[p.chunk_id]} for p in retrieved_provenance]
    context = format_context_no_labels(retrieved_dicts)

    attempt = 0
    last_exc: Exception | None = None
    while attempt < eval_cfg.max_attempts:
        attempt += 1
        try:
            result = llm_provider.classify_with_context(
                text=truncated_query_text, context=context, document_id=document_id, prompt=prompt, method="llm_rag",
            )
            case = CaseResult(
                method="llm_rag", document_id=document_id, effective_family_id=effective_family_id,
                condition=condition, true_label=true_label, input_fingerprint=condition_fingerprint,
                retrieval_context_fingerprint=retrieval_context_fingerprint, predicted_label=result.predicted_label,
                raw_response_hash=sha256_str(json.dumps(result.model_dump(), sort_keys=True, default=str)),
                status="success", attempt_count=attempt, truncated=query_was_truncated,
                latency_ms=result.latency_ms,
                prompt_tokens=result.token_usage.prompt_tokens if result.token_usage else None,
                completion_tokens=result.token_usage.completion_tokens if result.token_usage else None,
                total_tokens=result.token_usage.total_tokens if result.token_usage else None,
                estimated_cost_usd=result.estimated_cost,
                retrieved_chunks=retrieved_provenance,
                cache_key=cache_key,
            )
            _save_case_atomic(cache_dir, case)
            return case
        except ValueError as exc:
            case = CaseResult(
                method="llm_rag", document_id=document_id, effective_family_id=effective_family_id,
                condition=condition, true_label=true_label, input_fingerprint=condition_fingerprint,
                retrieval_context_fingerprint=retrieval_context_fingerprint, predicted_label=None, raw_response_hash=None,
                status="invalid", error_type=type(exc).__name__, attempt_count=attempt,
                truncated=query_was_truncated, latency_ms=0.0, retrieved_chunks=retrieved_provenance, cache_key=cache_key,
            )
            _save_case_atomic(cache_dir, case)
            return case
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc, eval_cfg.retryable_error_substrings) or attempt >= eval_cfg.max_attempts:
                break
            time.sleep(eval_cfg.retry_backoff_seconds * (2 ** (attempt - 1)))

    case = CaseResult(
        method="llm_rag", document_id=document_id, effective_family_id=effective_family_id,
        condition=condition, true_label=true_label, input_fingerprint=condition_fingerprint,
        retrieval_context_fingerprint=retrieval_context_fingerprint, predicted_label=None, raw_response_hash=None,
        status="failed", error_type=type(last_exc).__name__ if last_exc else "unknown",
        attempt_count=attempt, truncated=query_was_truncated, latency_ms=0.0,
        retrieved_chunks=retrieved_provenance, cache_key=cache_key,
    )
    _save_case_atomic(cache_dir, case)
    return case


def build_checkpoint10_freeze_record(
    settings,
    plain_prompt,
    rag_prompt,
    bert_checkpoint_artifact_id: str,
    bert_checkpoint_file_hashes: dict[str, str],
    bert_aggregation_method: str,
    rag_unmasked_corpus_fingerprint: str,
    rag_masked_corpus_fingerprint: str,
    condition_definitions: list,
    test_split_fingerprint: str,
    test_condition_registry_fingerprint: str,
) -> Checkpoint10FreezeRecord:
    """Builds and returns the pre-evaluation freeze record. Must be saved before any Gemini
    classification request is made."""
    eval_cfg = settings.family_aware.evaluation
    retrieval_cfg = settings.family_aware.rag.retrieval

    plain_prompt_payload = json.dumps(
        {"system_prompt": plain_prompt.system_prompt, "user_template": plain_prompt.user_template, "response_schema": plain_prompt.response_schema},
        sort_keys=True,
    )
    rag_prompt_payload = json.dumps(
        {"system_prompt": rag_prompt.system_prompt, "user_template": rag_prompt.user_template, "response_schema": rag_prompt.response_schema},
        sort_keys=True,
    )
    response_schema_hash = sha256_str(json.dumps(plain_prompt.response_schema, sort_keys=True))

    return Checkpoint10FreezeRecord(
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        llm_model_name=settings.llm.model,
        generation_temperature=0.0,  # matches GeminiProvider._classify_rendered, hardcoded there
        generation_max_output_tokens=None,  # not set for classify() -- unchanged from the frozen provider
        plain_prompt_version=plain_prompt.version,
        plain_prompt_hash=sha256_str(plain_prompt_payload),
        rag_prompt_version=rag_prompt.version,
        rag_prompt_hash=sha256_str(rag_prompt_payload),
        response_schema_hash=response_schema_hash,
        parser_version=PARSER_VERSION,
        allowed_labels=list(plain_prompt.allowed_labels),
        label_order=list(settings.base.labels),
        invalid_output_policy="A JSON parse failure or an out-of-schema predicted_label is recorded as status=invalid; never retried in hope of a different label.",
        refusal_policy="A refusal (empty/blocked response) raises the same way as any other malformed response and is recorded as status=invalid.",
        retry_policy=f"Only transient transport/timeout/rate-limit/server errors are retried, up to {eval_cfg.max_attempts} attempts with exponential backoff starting at {eval_cfg.retry_backoff_seconds}s.",
        timeout_policy="Transport-level timeouts raised by the Gemini client are treated as retryable (see retry_policy).",
        api_failure_policy="If all retry attempts are exhausted, status=failed and the case is still included in coverage/metrics denominators, never excluded.",
        max_attempts=eval_cfg.max_attempts,
        retry_backoff_seconds=eval_cfg.retry_backoff_seconds,
        bert_checkpoint_artifact_id=bert_checkpoint_artifact_id,
        bert_checkpoint_file_hashes=bert_checkpoint_file_hashes,
        bert_aggregation_method=bert_aggregation_method,
        rag_embedding_model=settings.rag.embedding_model,
        rag_unmasked_corpus_fingerprint=rag_unmasked_corpus_fingerprint,
        rag_masked_corpus_fingerprint=rag_masked_corpus_fingerprint,
        retrieval_candidate_pool_size=retrieval_cfg.candidate_pool_size,
        retrieval_top_k=retrieval_cfg.top_k,
        retrieval_max_chunks_per_parent_document=retrieval_cfg.max_chunks_per_parent_document,
        retrieval_max_results_per_effective_family=retrieval_cfg.max_results_per_effective_family,
        retrieval_tie_breaker=retrieval_cfg.tie_breaker,
        retrieval_duplicate_handling="Exact chunk text_hash duplicates are skipped after the first occurrence.",
        retrieval_fewer_than_k_behavior="Fewer than top_k results are returned rather than padding with lower-similarity/duplicate candidates.",
        condition_definitions=condition_definitions,
        test_split_fingerprint=test_split_fingerprint,
        test_condition_registry_fingerprint=test_condition_registry_fingerprint,
        no_changes_confirmation=(
            "No model, prompt, retrieval policy, parsing policy, or evaluation rule will be "
            "changed based on any result produced by this evaluation. Every field in this "
            "record was fixed before the first Gemini classification request was made."
        ),
        frozen=True,
    )
