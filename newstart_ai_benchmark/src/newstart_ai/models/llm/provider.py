"""LLM provider abstraction. Gemini is the only implementation for this MVP, but every
caller depends on the LLMProvider protocol -- no Gemini-specific code appears outside this
module, so adding OpenAI/Claude later is a new adapter class, not a rewrite.
"""

from __future__ import annotations

import json
import time
from typing import Protocol

import google.genai as genai
from google.genai import types

from newstart_ai.config.settings import Settings
from newstart_ai.models.llm.prompts import PromptTemplate
from newstart_ai.schemas.classification import ClassificationResult, GuidanceResult, Method, TokenUsage

# Placeholder per-million-token rates for cost estimation, kept in code (not billed against a
# live pricing API) since this MVP's evaluation volume is small. Update if Gemini's published
# pricing for the configured model changes.
GEMINI_INPUT_COST_PER_MILLION_TOKENS = 0.10
GEMINI_OUTPUT_COST_PER_MILLION_TOKENS = 0.40


class LLMProvider(Protocol):
    def classify(
        self, text: str, document_id: str, prompt: PromptTemplate, method: Method
    ) -> ClassificationResult: ...

    def generate_guidance(self, text: str, agency: str, prompt: PromptTemplate) -> GuidanceResult: ...


class GeminiProvider:
    """The only LLMProvider implementation for this MVP.

    provider/model/API key/endpoint all come from configs/llm.yaml + .env -- never
    hard-coded here, so swapping providers or models is a configuration change.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.llm.model
        api_key = settings.llm.resolve_api_key()
        http_options = types.HttpOptions(base_url=settings.llm.endpoint) if settings.llm.endpoint else None
        self.client = genai.Client(api_key=api_key, http_options=http_options)

    def classify(
        self,
        text: str,
        document_id: str,
        prompt: PromptTemplate,
        method: Method = "llm",
        extra_metadata: dict | None = None,
    ) -> ClassificationResult:
        """Classifies one document's text into exactly one of prompt.allowed_labels."""
        return self._classify_rendered(
            user_message=prompt.render_user_message(text),
            document_id=document_id,
            prompt=prompt,
            method=method,
            extra_metadata=extra_metadata,
        )

    def classify_with_context(
        self,
        text: str,
        context: str,
        document_id: str,
        prompt: PromptTemplate,
        method: Method = "llm_rag",
        extra_metadata: dict | None = None,
    ) -> ClassificationResult:
        """Like classify(), but for a prompt template with both {text} and {context}
        placeholders (the RAG classification prompt) -- used by RagEnhancedClassifier."""
        return self._classify_rendered(
            user_message=prompt.render(text=text, context=context),
            document_id=document_id,
            prompt=prompt,
            method=method,
            extra_metadata=extra_metadata,
        )

    def _classify_rendered(
        self,
        user_message: str,
        document_id: str,
        prompt: PromptTemplate,
        method: Method,
        extra_metadata: dict | None,
    ) -> ClassificationResult:
        """Raises if Gemini returns a label outside the allowed set, rather than silently
        coercing it -- an unexpected label is a real problem to surface, not paper over."""
        config = types.GenerateContentConfig(
            system_instruction=prompt.system_prompt,
            response_mime_type="application/json",
            response_schema=prompt.response_schema,
            temperature=0,
        )

        start = time.perf_counter()
        response = self.client.models.generate_content(
            model=self.model_name, contents=user_message, config=config
        )
        latency_ms = (time.perf_counter() - start) * 1000

        parsed = json.loads(response.text)
        predicted_label = parsed["predicted_label"]
        if predicted_label not in prompt.allowed_labels:
            raise ValueError(
                f"Gemini returned label {predicted_label!r}, outside the allowed set "
                f"{prompt.allowed_labels}."
            )

        usage = response.usage_metadata
        token_usage = TokenUsage(
            prompt_tokens=usage.prompt_token_count,
            completion_tokens=usage.candidates_token_count,
            total_tokens=usage.total_token_count,
        )

        metadata = {
            "provider": "gemini",
            "model_version": self.model_name,
            "prompt_version": prompt.version,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return ClassificationResult(
            method=method,
            document_id=document_id,
            predicted_label=predicted_label,
            latency_ms=latency_ms,
            token_usage=token_usage,
            estimated_cost=_estimate_cost(usage),
            metadata=metadata,
        )

    def generate_guidance(self, text: str, agency: str, prompt: PromptTemplate) -> GuidanceResult:
        """Produces the short demonstration guidance text for the Random Form Routing Demo.

        Not part of the research evaluation -- answer quality is out of scope (docs/BLUEPRINT.md
        Section 10).
        """
        config = types.GenerateContentConfig(
            system_instruction=prompt.system_prompt,
            temperature=0.2,
            max_output_tokens=300,
            # gemini-3.6-flash reasons before answering by default, and that reasoning
            # counts against max_output_tokens -- without capping it, a low token budget
            # can be exhausted entirely on hidden reasoning, cutting off the visible answer
            # (observed directly: MAX_TOKENS with an empty/truncated response). MINIMAL
            # thinking is appropriate here since this is a short, low-stakes demo answer,
            # not a task requiring multi-step reasoning.
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
        )
        user_message = prompt.render_user_message(text)

        start = time.perf_counter()
        response = self.client.models.generate_content(
            model=self.model_name, contents=user_message, config=config
        )
        latency_ms = (time.perf_counter() - start) * 1000

        return GuidanceResult(
            agency=agency,
            guidance_text=response.text.strip(),
            latency_ms=latency_ms,
            metadata={
                "provider": "gemini",
                "model_version": self.model_name,
                "prompt_version": prompt.version,
            },
        )


def _estimate_cost(usage) -> float:
    prompt_cost = (usage.prompt_token_count or 0) / 1_000_000 * GEMINI_INPUT_COST_PER_MILLION_TOKENS
    # "Thinking" tokens are billed as output tokens alongside the visible completion.
    output_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
    output_cost = output_tokens / 1_000_000 * GEMINI_OUTPUT_COST_PER_MILLION_TOKENS
    return round(prompt_cost + output_cost, 8)
