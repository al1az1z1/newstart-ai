"""Result schema for one retrieved reference document (docs/BLUEPRINT.md Section 8).

Kept separate from ClassificationResult.retrieved_document_ids/retrieval_similarity, which
summarize retrieval for storage -- this schema is the richer intermediate the Retriever
returns before RagEnhancedClassifier builds the final ClassificationResult.
"""

from __future__ import annotations

from pydantic import BaseModel


class RetrievedDocument(BaseModel):
    document_id: str
    label: str
    similarity: float
    text_snippet: str
