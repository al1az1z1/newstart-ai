"""Gemini embedding provider for the routing RAG index.

Kept independent from the generation LLM (docs/BLUEPRINT.md Section 8) -- the embedding
model is its own config value (configs/rag.yaml: embedding_model), so another generation
provider could reuse the same vector index later.
"""

from __future__ import annotations

import google.genai as genai

from newstart_ai.config.settings import Settings

# Keeps individual embedContent calls modest in size -- some documents in this dataset are
# very long (up to ~640k characters), so batching too many together risks an oversized
# request even though a single long document embeds fine on its own.
EMBEDDING_BATCH_SIZE = 10


class GeminiEmbeddingProvider:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.rag.embedding_model
        # Embeddings use the same Gemini API key as generation -- "independent" (Section 8)
        # means the embedding model is swappable independently, not a separate account.
        api_key = settings.llm.resolve_api_key()
        self.client = genai.Client(api_key=api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Returns one embedding vector per input text, in the same order."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + EMBEDDING_BATCH_SIZE]
            response = self.client.models.embed_content(model=self.model_name, contents=batch)
            vectors.extend(embedding.values for embedding in response.embeddings)
        return vectors
