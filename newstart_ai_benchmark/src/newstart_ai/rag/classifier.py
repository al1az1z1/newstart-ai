"""LLM+RAG routing classifier: retrieves similar training examples, then asks the LLM to
classify using them as context. Retrieval similarity is stored as diagnostic RAG information
and is never combined with BERT softmax or the LLM's own output into one confidence value.
"""

from __future__ import annotations

from newstart_ai.models.llm.prompts import PromptTemplate
from newstart_ai.models.llm.provider import LLMProvider
from newstart_ai.rag.index import Retriever
from newstart_ai.schemas.classification import ClassificationResult, Method
from newstart_ai.schemas.rag import RetrievedDocument


def format_context(retrieved_docs: list[RetrievedDocument]) -> str:
    lines = [
        f"{i}. Agency: {doc.label}\n   Excerpt: {doc.text_snippet}"
        for i, doc in enumerate(retrieved_docs, start=1)
    ]
    return "\n".join(lines) if lines else "(no similar examples retrieved)"


class RagEnhancedClassifier:
    def __init__(
        self,
        retriever: Retriever,
        llm_provider: LLMProvider,
        prompt: PromptTemplate,
    ):
        self.retriever = retriever
        self.llm_provider = llm_provider
        self.prompt = prompt

    def classify(
        self,
        text: str,
        document_id: str,
        method: Method = "llm_rag",
        extra_metadata: dict | None = None,
    ) -> ClassificationResult:
        retrieved = self.retriever.retrieve(text)
        context = format_context(retrieved)

        result = self.llm_provider.classify_with_context(
            text=text,
            context=context,
            document_id=document_id,
            prompt=self.prompt,
            method=method,
            extra_metadata=extra_metadata,
        )
        result.retrieval_similarity = retrieved[0].similarity if retrieved else None
        result.retrieved_document_ids = [doc.document_id for doc in retrieved]
        return result
