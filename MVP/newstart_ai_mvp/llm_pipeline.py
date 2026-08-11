"""Self-contained Gemini LLM and Gemini+RAG evaluation pipeline (Checkpoint 10).

A copy of the original project's plain-LLM and LLM+RAG case runners, reorganized into one
module. `CaseResult` is a plain dict instead of a Pydantic schema. The methodology (temperature
0, structured JSON output, the frozen 6,000-character truncation shared with the embedding
provider, disk-cached per-case results, transient-error-only retries) is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from newstart_ai_mvp.rag_pipeline import MAX_EMBEDDING_INPUT_CHARACTERS


def sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def truncate_for_llm(text: str) -> tuple[str, bool]:
    """Reuses the same frozen 6,000-character policy as the embedding provider -- no
    separate truncation policy for the LLM input."""
    if len(text) > MAX_EMBEDDING_INPUT_CHARACTERS:
        return text[:MAX_EMBEDDING_INPUT_CHARACTERS], True
    return text, False


def compute_cache_key(method: str, model: str, prompt_version: str, document_id: str, condition: str, condition_fingerprint: str, retrieval_context_fingerprint: str | None) -> str:
    payload = f"{method}|{model}|{prompt_version}|{document_id}|{condition}|{condition_fingerprint}|{retrieval_context_fingerprint or ''}"
    return sha256_str(payload)


def _cache_path(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / f"{cache_key}.json"


def _load_cached_case(cache_dir: Path, cache_key: str) -> dict | None:
    path = _cache_path(cache_dir, cache_key)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_cached_case(cache_dir: Path, cache_key: str, case: dict) -> None:
    """Atomic write (tmp file then replace) so an interrupted process never leaves a
    corrupt/partial cache entry behind."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, cache_key)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(case, f, default=str)
    tmp_path.replace(path)


class PromptTemplate:
    """A copy of the original project's versioned prompt template loader."""

    def __init__(self, version, system_prompt, user_template, allowed_labels=None, response_schema=None, **_ignored):
        self.version = version
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.allowed_labels = allowed_labels
        self.response_schema = response_schema

    def render(self, **placeholders: str) -> str:
        message = self.user_template
        for key, value in placeholders.items():
            message = message.replace("{" + key + "}", value)
        return message

    def render_user_message(self, text: str) -> str:
        return self.render(text=text)


def load_prompt(path: Path) -> PromptTemplate:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return PromptTemplate(**raw)


def load_classification_prompt(settings) -> PromptTemplate:
    return load_prompt(settings.resolve_path(settings.llm.classification_prompt_path))


def load_family_aware_rag_classification_prompt(settings) -> PromptTemplate:
    return load_prompt(settings.resolve_path(settings.family_aware.evaluation.family_aware_rag_classification_prompt_path))


GEMINI_INPUT_COST_PER_MILLION_TOKENS = 0.10
GEMINI_OUTPUT_COST_PER_MILLION_TOKENS = 0.40


class GeminiProvider:
    """The real Gemini classification client -- temperature fixed at 0, structured JSON
    output enforced via response_schema, an out-of-allowed-labels response raises rather
    than being silently coerced."""

    def __init__(self, settings):
        import google.genai as genai

        self.settings = settings
        self.model_name = settings.llm.model
        api_key = settings.llm.resolve_api_key()
        self.client = genai.Client(api_key=api_key)

    def classify(self, text: str, document_id: str, prompt: PromptTemplate, method: str = "llm") -> dict:
        return self._classify_rendered(prompt.render_user_message(text), document_id, prompt, method)

    def classify_with_context(self, text: str, context: str, document_id: str, prompt: PromptTemplate, method: str = "llm_rag") -> dict:
        return self._classify_rendered(prompt.render(text=text, context=context), document_id, prompt, method)

    def _classify_rendered(self, user_message: str, document_id: str, prompt: PromptTemplate, method: str) -> dict:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=prompt.system_prompt, response_mime_type="application/json",
            response_schema=prompt.response_schema, temperature=0,
        )
        start = time.perf_counter()
        response = self.client.models.generate_content(model=self.model_name, contents=user_message, config=config)
        latency_ms = (time.perf_counter() - start) * 1000

        parsed = json.loads(response.text)
        predicted_label = parsed["predicted_label"]
        if predicted_label not in prompt.allowed_labels:
            raise ValueError(f"Gemini returned label {predicted_label!r}, outside the allowed set {prompt.allowed_labels}.")

        usage = response.usage_metadata
        prompt_tokens, completion_tokens, total_tokens = usage.prompt_token_count, usage.candidates_token_count, usage.total_token_count
        cost = round((prompt_tokens or 0) / 1e6 * GEMINI_INPUT_COST_PER_MILLION_TOKENS + (completion_tokens or 0) / 1e6 * GEMINI_OUTPUT_COST_PER_MILLION_TOKENS, 8)
        return {
            "method": method, "document_id": document_id, "predicted_label": predicted_label, "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens,
            "estimated_cost_usd": cost, "raw_response_hash": sha256_str(response.text),
        }


# =========================================================================================
# Plain-LLM case runner
# =========================================================================================

RETRYABLE_ERROR_SUBSTRINGS = ("timeout", "rate limit", "503", "500", "connection", "unavailable")


def _is_retryable(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(s in message for s in RETRYABLE_ERROR_SUBSTRINGS)


def run_plain_llm_case(document_id: str, effective_family_id: str, condition: str, true_label: str, text: str, condition_fingerprint: str, llm_provider: GeminiProvider, prompt: PromptTemplate, settings) -> dict:
    """Runs (or reuses a cached) plain-LLM classification for one (document, condition).
    Input is truncated at the frozen 6,000-character limit; a truncation is recorded as a
    boolean flag, never a character count. Only transient errors are retried."""
    eval_cfg = settings.family_aware.evaluation
    cache_dir = settings.resolve_path(eval_cfg.cache_dir) / "llm"
    cache_key = compute_cache_key("llm", llm_provider.model_name, prompt.version, document_id, condition, condition_fingerprint, None)

    cached = _load_cached_case(cache_dir, cache_key)
    if cached is not None:
        return cached

    truncated_text, was_truncated = truncate_for_llm(text)
    attempt_count = 0
    for attempt in range(1, eval_cfg.max_attempts + 1):
        attempt_count = attempt
        try:
            result = llm_provider.classify(truncated_text, document_id, prompt, method="llm")
            case = {
                "method": "llm", "document_id": document_id, "effective_family_id": effective_family_id,
                "condition": condition, "true_label": true_label, "input_fingerprint": condition_fingerprint,
                "retrieval_context_fingerprint": None, "predicted_label": result["predicted_label"],
                "raw_response_hash": result["raw_response_hash"], "status": "success", "error_type": None,
                "attempt_count": attempt_count, "truncated": was_truncated, "latency_ms": result["latency_ms"],
                "prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"], "estimated_cost_usd": result["estimated_cost_usd"],
                "retrieved_chunks": [], "cache_key": cache_key,
            }
            _save_cached_case(cache_dir, cache_key, case)
            return case
        except ValueError:
            case = {
                "method": "llm", "document_id": document_id, "effective_family_id": effective_family_id,
                "condition": condition, "true_label": true_label, "input_fingerprint": condition_fingerprint,
                "retrieval_context_fingerprint": None, "predicted_label": None, "status": "invalid",
                "error_type": "invalid_label", "attempt_count": attempt_count, "truncated": was_truncated,
                "retrieved_chunks": [], "cache_key": cache_key,
            }
            _save_cached_case(cache_dir, cache_key, case)
            return case
        except Exception as exc:
            if not _is_retryable(exc) or attempt == eval_cfg.max_attempts:
                case = {
                    "method": "llm", "document_id": document_id, "effective_family_id": effective_family_id,
                    "condition": condition, "true_label": true_label, "input_fingerprint": condition_fingerprint,
                    "retrieval_context_fingerprint": None, "predicted_label": None, "status": "failed",
                    "error_type": type(exc).__name__, "attempt_count": attempt_count, "truncated": was_truncated,
                    "retrieved_chunks": [], "cache_key": cache_key,
                }
                _save_cached_case(cache_dir, cache_key, case)
                return case
            time.sleep(eval_cfg.retry_backoff_seconds * (2 ** (attempt - 1)))


def format_context_no_labels(retrieved: list[dict]) -> str:
    """Formats retrieved chunks as untrusted reference TEXT ONLY -- no agency, document_id,
    family, filename, or any other metadata is ever included in what Gemini sees."""
    if not retrieved:
        return "(no reference excerpts retrieved)"
    return "\n\n".join(f"Excerpt {i}:\n{chunk['text']}" for i, chunk in enumerate(retrieved, start=1))


def run_llm_rag_case(document_id: str, effective_family_id: str, condition: str, true_label: str, text: str, condition_fingerprint: str, masked: bool, unmasked_collection, masked_collection, chunk_text_by_id: dict[str, str], embedding_provider, llm_provider: GeminiProvider, prompt: PromptTemplate, settings) -> dict:
    """Runs (or reuses a cached) LLM+RAG classification. Routes unmasked conditions to the
    unmasked index and masked conditions to the masked index. The prompt only ever sees
    retrieved chunk TEXT (format_context_no_labels)."""
    from newstart_ai_mvp.rag_pipeline import retrieve_diversified

    eval_cfg = settings.family_aware.evaluation
    cache_dir = settings.resolve_path(eval_cfg.cache_dir) / "llm_rag"

    truncated_query_text, was_truncated = truncate_for_llm(text)
    query_vectors, _usage = embedding_provider.embed_texts([truncated_query_text], settings.family_aware.rag.query_task_type)
    query_vector = query_vectors[0]

    collection = masked_collection if masked else unmasked_collection
    _before, after = retrieve_diversified(collection, query_vector, settings)

    retrieved_chunks = [
        {"chunk_id": r["chunk_id"], "rank": rank, "similarity": r["similarity"], "parent_document_id": r["parent_document_id"],
         "effective_family_id": r["effective_family_id"], "effective_agency": r["effective_agency"], "text_hash": r["text_hash"], "masked": masked}
        for rank, r in enumerate(after, start=1)
    ]
    retrieval_context_fingerprint = sha256_str("|".join(f"{c['chunk_id']}:{c['text_hash']}" for c in retrieved_chunks))
    cache_key = compute_cache_key("llm_rag", llm_provider.model_name, prompt.version, document_id, condition, condition_fingerprint, retrieval_context_fingerprint)

    cached = _load_cached_case(cache_dir, cache_key)
    if cached is not None:
        return cached

    context = format_context_no_labels([{"text": chunk_text_by_id[c["chunk_id"]]} for c in retrieved_chunks])

    attempt_count = 0
    for attempt in range(1, eval_cfg.max_attempts + 1):
        attempt_count = attempt
        try:
            result = llm_provider.classify_with_context(truncated_query_text, context, document_id, prompt, method="llm_rag")
            case = {
                "method": "llm_rag", "document_id": document_id, "effective_family_id": effective_family_id,
                "condition": condition, "true_label": true_label, "input_fingerprint": condition_fingerprint,
                "retrieval_context_fingerprint": retrieval_context_fingerprint, "predicted_label": result["predicted_label"],
                "raw_response_hash": result["raw_response_hash"], "status": "success", "error_type": None,
                "attempt_count": attempt_count, "truncated": was_truncated, "latency_ms": result["latency_ms"],
                "prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"], "estimated_cost_usd": result["estimated_cost_usd"],
                "retrieved_chunks": retrieved_chunks, "cache_key": cache_key,
            }
            _save_cached_case(cache_dir, cache_key, case)
            return case
        except ValueError:
            case = {
                "method": "llm_rag", "document_id": document_id, "effective_family_id": effective_family_id,
                "condition": condition, "true_label": true_label, "input_fingerprint": condition_fingerprint,
                "retrieval_context_fingerprint": retrieval_context_fingerprint, "predicted_label": None,
                "status": "invalid", "error_type": "invalid_label", "attempt_count": attempt_count,
                "truncated": was_truncated, "retrieved_chunks": retrieved_chunks, "cache_key": cache_key,
            }
            _save_cached_case(cache_dir, cache_key, case)
            return case
        except Exception as exc:
            if not _is_retryable(exc) or attempt == eval_cfg.max_attempts:
                case = {
                    "method": "llm_rag", "document_id": document_id, "effective_family_id": effective_family_id,
                    "condition": condition, "true_label": true_label, "input_fingerprint": condition_fingerprint,
                    "retrieval_context_fingerprint": retrieval_context_fingerprint, "predicted_label": None,
                    "status": "failed", "error_type": type(exc).__name__, "attempt_count": attempt_count,
                    "truncated": was_truncated, "retrieved_chunks": retrieved_chunks, "cache_key": cache_key,
                }
                _save_cached_case(cache_dir, cache_key, case)
                return case
            time.sleep(eval_cfg.retry_backoff_seconds * (2 ** (attempt - 1)))


# =========================================================================================
# Metrics + integrity (Checkpoint 10)
# =========================================================================================


def build_method_condition_metrics(cases: list[dict], label_order: list[str]) -> dict:
    """Failed/invalid cases are scored as automatically wrong, never excluded from any
    denominator."""
    from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

    if not cases:
        raise ValueError("Cannot build metrics for zero cases.")
    method, condition = cases[0]["method"], cases[0]["condition"]
    n = len(cases)
    invalid_count = sum(1 for c in cases if c["status"] == "invalid")
    failed_count = sum(1 for c in cases if c["status"] == "failed")
    coverage_rate = round(100 * sum(1 for c in cases if c["status"] == "success") / n, 2)

    sentinel = "__NO_PREDICTION__"
    y_true = [c["true_label"] for c in cases]
    y_pred = [c["predicted_label"] if c["status"] == "success" else sentinel for c in cases]

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=label_order, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    precisions, recalls, f1s, supports = precision_recall_fscore_support(y_true, y_pred, labels=label_order, zero_division=0)

    confusion: dict[str, dict[str, int]] = {t: {p: 0 for p in label_order} for t in label_order}
    for t, p in zip(y_true, y_pred):
        if p in confusion.get(t, {}):
            confusion[t][p] += 1

    token_totals = [c.get("total_tokens") for c in cases if c.get("total_tokens") is not None]
    cost_totals = [c.get("estimated_cost_usd") for c in cases if c.get("estimated_cost_usd") is not None]

    return {
        "method": method, "condition": condition, "document_count": n, "coverage_rate": coverage_rate,
        "invalid_count": invalid_count, "failed_count": failed_count,
        "document_macro_f1": macro_f1, "document_accuracy": accuracy,
        "macro_precision": float(precisions.mean()), "macro_recall": float(recalls.mean()),
        "per_agency": [{"agency": label, "precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)} for label, p, r, f, s in zip(label_order, precisions, recalls, f1s, supports)],
        "confusion_matrix": confusion, "error_count": sum(1 for t, p in zip(y_true, y_pred) if t != p),
        "estimated_tokens_total": int(sum(token_totals)) if token_totals else None,
        "estimated_cost_total_usd": round(sum(cost_totals), 6) if cost_totals else None,
        "notes": ["Failed/invalid cases are scored as incorrect and included in every denominator above."],
    }


def build_evaluation_integrity_proof(condition_fingerprints_by_method: dict[str, dict], **flags) -> dict:
    """Compares the EXACT SHA-256 fingerprint recorded by each method for every
    (document_id, condition) pair."""
    methods = list(condition_fingerprints_by_method.keys())
    fingerprints_match = True
    if len(methods) > 1:
        reference = condition_fingerprints_by_method[methods[0]]
        for method in methods[1:]:
            if condition_fingerprints_by_method[method] != reference:
                fingerprints_match = False
                break
    return {"version": "v1", "created_at": datetime.now(timezone.utc).isoformat(), "condition_fingerprints_match_across_methods": fingerprints_match, **flags}
