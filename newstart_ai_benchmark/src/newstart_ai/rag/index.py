"""Routing-only RAG vector index (Chroma).

Built from the training split only (docs/BLUEPRINT.md Section 8) -- never call
build_routing_index with validation or test data. 07_rag_index_creation.ipynb is the only
notebook that builds this index; every other notebook only queries it.
"""

from __future__ import annotations

import chromadb

from newstart_ai.config.settings import Settings
from newstart_ai.rag.embeddings import GeminiEmbeddingProvider
from newstart_ai.schemas.rag import RetrievedDocument

COLLECTION_NAME = "routing_index"
SNIPPET_LENGTH = 500  # characters of each indexed document shown as RAG prompt context


def _get_client(settings: Settings) -> chromadb.ClientAPI:
    persist_dir = settings.resolve_path(settings.rag.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def build_routing_index(
    train_df, settings: Settings, embedding_provider: GeminiEmbeddingProvider | None = None
) -> int:
    """(Re)builds the routing index from train_df only. Returns the number of indexed rows."""
    ds_cfg = settings.base.dataset
    embedding_provider = embedding_provider or GeminiEmbeddingProvider(settings)

    client = _get_client(settings)
    existing = {c.name for c in client.list_collections()}
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    ids = train_df[ds_cfg.id_column].astype(str).tolist()
    texts = train_df[ds_cfg.text_column].tolist()
    labels = train_df[ds_cfg.label_column].tolist()

    embeddings = embedding_provider.embed(texts)
    metadatas = [{"label": label, "snippet": text[:SNIPPET_LENGTH]} for label, text in zip(labels, texts)]

    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
    return len(ids)


def load_routing_index(settings: Settings):
    return _get_client(settings).get_collection(COLLECTION_NAME)


def assert_no_test_ids_in_index(settings: Settings, test_document_ids: list[str]) -> None:
    """Proves the routing index contains none of the frozen test set's document IDs."""
    collection = load_routing_index(settings)
    indexed_ids = set(collection.get()["ids"])
    leaking = indexed_ids & set(test_document_ids)
    if leaking:
        raise ValueError(f"Routing RAG index contains test document IDs: {leaking}")


class Retriever:
    """Embeds a query document and retrieves its top_k nearest neighbors from the routing
    index, each with a cosine similarity score."""

    def __init__(self, settings: Settings, embedding_provider: GeminiEmbeddingProvider | None = None):
        self.settings = settings
        self.embedding_provider = embedding_provider or GeminiEmbeddingProvider(settings)
        self.collection = load_routing_index(settings)

    def retrieve(self, text: str, top_k: int | None = None) -> list[RetrievedDocument]:
        top_k = top_k or self.settings.rag.top_k
        [query_embedding] = self.embedding_provider.embed([text])
        result = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

        retrieved = []
        for doc_id, distance, metadata in zip(
            result["ids"][0], result["distances"][0], result["metadatas"][0]
        ):
            retrieved.append(
                RetrievedDocument(
                    document_id=doc_id,
                    label=metadata["label"],
                    similarity=1 - distance,  # cosine space: distance = 1 - cosine similarity
                    text_snippet=metadata["snippet"],
                )
            )
        return retrieved
