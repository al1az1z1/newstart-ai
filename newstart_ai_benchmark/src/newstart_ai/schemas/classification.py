"""Shared result schemas returned by every classifier and guidance agent.

Every method (BERT, LLM, LLM+RAG) returns the same ClassificationResult shape so notebooks,
the evaluator, and the API can treat them uniformly -- but method-specific confidence values
(BERT softmax, RAG retrieval similarity) stay in separate optional fields and are never
combined into one universal score.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Method = Literal["bert", "llm", "llm_rag"]
AgencyLabel = Literal["USCIS", "DMV", "SSA", "IRS"]


class TokenUsage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ClassificationResult(BaseModel):
    method: Method
    document_id: str
    predicted_label: AgencyLabel
    true_label: AgencyLabel | None = None

    # BERT only
    probabilities: dict[AgencyLabel, float] | None = None

    # LLM+RAG only -- diagnostic retrieval info, never a substitute for confidence
    retrieval_similarity: float | None = None
    retrieved_document_ids: list[str] | None = None

    latency_ms: float

    # LLM / LLM+RAG only
    token_usage: TokenUsage | None = None
    estimated_cost: float | None = None

    # provider, model_version, prompt_version, split ("train"/"validation"/"test"), etc.
    metadata: dict = Field(default_factory=dict)


class GuidanceResult(BaseModel):
    """Output of a GuidanceAgent -- a short demonstration answer, not a research artifact."""

    agency: AgencyLabel
    guidance_text: str
    latency_ms: float
    metadata: dict = Field(default_factory=dict)
