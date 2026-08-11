"""Stage: Gemini embeddings + masked/unmasked Chroma index construction (Checkpoint 9).

Real logic lives in newstart_ai_mvp.rag_pipeline (FamilyAwareGeminiEmbeddingProvider,
build_family_aware_corpus_index).

Default mode reports real Chroma collection counts via a read-only client.get_collection(...)
.count() call -- no query, no write.

--rebuild-embeddings --rebuild-index calls the real Gemini embedding API for every TRAIN
chunk not already cache-hit and rebuilds both Chroma collections. This MUST run inside
redirect_frozen_outputs: build_family_aware_corpus_index calls client.delete_collection() on
the target collection name before recreating it, so if this were ever pointed at the real,
frozen persist_dir it would delete the submitted vector store outright.

    python -m newstart_ai_mvp.build_rag_index                                      # safe (default)
    python -m newstart_ai_mvp.build_rag_index --rebuild-embeddings --rebuild-index    # real API calls
"""

from __future__ import annotations

from newstart_ai_mvp import artifact_report as ar
from newstart_ai_mvp.cli_common import build_base_parser, get_settings, print_expensive_mode_banner, print_safe_mode_banner
from newstart_ai_mvp.run_scope import new_run_id, redirect_frozen_outputs, run_root

STAGE_NAME = "build_rag_index"


def run_safe(settings) -> None:
    print_safe_mode_banner(STAGE_NAME)
    ar.print_report("RAG vector stores", ar.describe_rag_index(settings))


def run_expensive(settings, run_id: str, input_run_id: str | None) -> None:
    import hashlib

    import pandas as pd

    from newstart_ai_mvp.rag_pipeline import FamilyAwareGeminiEmbeddingProvider, build_family_aware_corpus_index

    print_expensive_mode_banner(STAGE_NAME, run_id)
    print(f"[{STAGE_NAME}] This calls the real Gemini embedding API for every train chunk not")
    print(f"[{STAGE_NAME}] already cache-hit under the run's own embedding_cache/. Estimate cost")
    print(f"[{STAGE_NAME}] before proceeding for a large corpus.\n")

    if input_run_id:
        chunks_dir = run_root(input_run_id) / settings.family_aware.chunking.output_dir
        masked_dir = run_root(input_run_id) / settings.family_aware.masking.output_dir
    else:
        chunks_dir = settings.resolve_path(settings.family_aware.chunking.output_dir)
        masked_dir = settings.resolve_path(settings.family_aware.masking.output_dir)

    train_chunks = pd.read_csv(chunks_dir / "train_chunks.csv")
    train_masked_chunks = pd.read_csv(masked_dir / "train_masked_chunks.csv")
    merge_cols = ["chunk_id", "document_id", "effective_family_id", "agency", "effective_agency", "split", "chunk_index", "total_chunks", "token_start", "token_end"]
    unmasked_df = train_chunks
    masked_df = train_masked_chunks.merge(
        train_chunks[[c for c in merge_cols if c not in train_masked_chunks.columns]].assign(
            chunk_id=train_chunks["chunk_id"]
        ),
        on="chunk_id",
        how="left",
    )

    embedding_config_fingerprint = hashlib.sha256(
        f"{settings.bert.base_model}|{settings.family_aware.rag.document_task_type}".encode()
    ).hexdigest()

    with redirect_frozen_outputs(run_id):
        provider = FamilyAwareGeminiEmbeddingProvider(settings)
        unmasked_manifest, unmasked_usage = build_family_aware_corpus_index(
            unmasked_df, "chunk_text", "chunk_text_hash", False, embedding_config_fingerprint, settings, provider
        )
        masked_manifest, masked_usage = build_family_aware_corpus_index(
            masked_df, "masked_chunk_text", "masked_chunk_text_hash", True, embedding_config_fingerprint, settings, provider
        )

    print(f"Unmasked index: {unmasked_manifest['indexed_chunk_count']} chunks, {unmasked_usage['api_requests_made']} API requests")
    print(f"Masked index:   {masked_manifest['indexed_chunk_count']} chunks, {masked_usage['api_requests_made']} API requests")
    print(f"Written under MVP/runs/{run_id}/ -- frozen vector stores untouched.")


def main(argv: list[str] | None = None) -> None:
    parser = build_base_parser(__doc__ or "")
    parser.add_argument("--rebuild-embeddings", action="store_true", help="Required alongside --rebuild-index to actually run.")
    parser.add_argument("--rebuild-index", action="store_true", help="Required alongside --rebuild-embeddings to actually run.")
    args = parser.parse_args(argv)

    settings = get_settings()
    if not (args.rebuild_embeddings and args.rebuild_index):
        run_safe(settings)
        return

    run_id = args.run_id or new_run_id()
    run_expensive(settings, run_id, args.input_run_id)


if __name__ == "__main__":
    main()
