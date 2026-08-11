"""Stage: Gemini+RAG classification sweep (Checkpoint 10, RAG half).

Real logic lives in newstart_ai_mvp.llm_pipeline (run_llm_rag_case, format_context_no_labels
-- the prompt only ever sees retrieved chunk TEXT, never labels or IDs) and
newstart_ai_mvp.rag_pipeline. Masked queries are routed to the masked Chroma collection,
unmasked queries to the unmasked collection -- never mixed.

Default mode recomputes the primary-condition macro F1 from the real, frozen 990-row
checkpoint10_llm_rag_predictions.jsonl and compares it to the saved manifest value.

--run-api calls the real Gemini + embedding APIs. Requires an explicit choice of which
Chroma index to query: --use-frozen-index (query the real, submitted vector store,
read-only) or --index-run-id ID (query a specific prior build_rag_index run's own index).
This forces the choice rather than silently defaulting to one or the other.

    python -m newstart_ai_mvp.evaluate_rag                                          # safe (default)
    python -m newstart_ai_mvp.evaluate_rag --run-api --use-frozen-index               # real API calls
"""

from __future__ import annotations

from newstart_ai_mvp import artifact_report as ar
from newstart_ai_mvp.cli_common import build_base_parser, get_settings, print_expensive_mode_banner, print_safe_mode_banner
from newstart_ai_mvp.run_scope import new_run_id, redirect_frozen_outputs, run_root

STAGE_NAME = "evaluate_rag"


def run_safe(settings) -> None:
    print_safe_mode_banner(STAGE_NAME)
    ar.print_report("LLM+RAG predictions (recomputed vs. reported)", ar.describe_rag_predictions(settings))


def run_expensive(settings, run_id: str, index_run_id: str | None, use_frozen_index: bool, input_run_id: str | None) -> None:
    import json

    import pandas as pd

    from newstart_ai_mvp.llm_pipeline import GeminiProvider, build_method_condition_metrics, load_family_aware_rag_classification_prompt, run_llm_rag_case
    from newstart_ai_mvp.rag_pipeline import FamilyAwareGeminiEmbeddingProvider, load_family_aware_collection

    print_expensive_mode_banner(STAGE_NAME, run_id)

    if input_run_id:
        conditions_dir = run_root(input_run_id) / settings.family_aware.conditions.output_dir
        chunks_dir = run_root(input_run_id) / settings.family_aware.chunking.output_dir
        masked_dir = run_root(input_run_id) / settings.family_aware.masking.output_dir
    else:
        conditions_dir = settings.resolve_path(settings.family_aware.conditions.output_dir)
        chunks_dir = settings.resolve_path(settings.family_aware.chunking.output_dir)
        masked_dir = settings.resolve_path(settings.family_aware.masking.output_dir)

    registry = pd.read_csv(conditions_dir / "condition_registry_test.csv")
    chunk_text_by_id = dict(zip(pd.read_csv(chunks_dir / "train_chunks.csv")["chunk_id"], pd.read_csv(chunks_dir / "train_chunks.csv")["chunk_text"]))
    masked_chunk_text_by_id = dict(
        zip(pd.read_csv(masked_dir / "train_masked_chunks.csv")["chunk_id"], pd.read_csv(masked_dir / "train_masked_chunks.csv")["masked_chunk_text"])
    )
    chunk_text_by_id.update(masked_chunk_text_by_id)

    provider = GeminiProvider(settings)
    embedding_provider = FamilyAwareGeminiEmbeddingProvider(settings)
    prompt = load_family_aware_rag_classification_prompt(settings)

    if use_frozen_index:
        unmasked_collection = load_family_aware_collection(settings, masked=False)
        masked_collection = load_family_aware_collection(settings, masked=True)
    elif index_run_id:
        with redirect_frozen_outputs(index_run_id):
            unmasked_collection = load_family_aware_collection(settings, masked=False)
            masked_collection = load_family_aware_collection(settings, masked=True)
    else:
        raise SystemExit("Pass exactly one of --use-frozen-index or --index-run-id ID.")

    cases = []
    with redirect_frozen_outputs(run_id):
        for row in registry.itertuples(index=False):
            result = run_llm_rag_case(
                document_id=str(row.document_id),
                effective_family_id=row.effective_agency,
                condition=row.condition,
                true_label=row.effective_agency,
                text=row.text,
                condition_fingerprint=row.text_fingerprint,
                masked=bool(row.masked),
                unmasked_collection=unmasked_collection,
                masked_collection=masked_collection,
                chunk_text_by_id=chunk_text_by_id,
                embedding_provider=embedding_provider,
                llm_provider=provider,
                prompt=prompt,
                settings=settings,
            )
            cases.append(result)

        metrics = build_method_condition_metrics(cases, list(settings.base.labels))
        reports_dir = settings.resolve_path("artifacts/family_aware/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        with open(reports_dir / "checkpoint10_llm_rag_predictions.jsonl", "w", encoding="utf-8") as f:
            for case in cases:
                f.write(json.dumps(case, default=str) + "\n")

    print(f"Ran {len(cases)} cases. Primary macro F1: {metrics['document_macro_f1']:.4f}")
    print(f"Written under MVP/runs/{run_id}/ -- frozen predictions untouched.")


def main(argv: list[str] | None = None) -> None:
    parser = build_base_parser(__doc__ or "")
    parser.add_argument("--run-api", action="store_true", help="Actually call the Gemini/embedding APIs.")
    parser.add_argument("--use-frozen-index", action="store_true", help="Query the real, submitted Chroma index (read-only).")
    parser.add_argument("--index-run-id", default=None, help="Query a specific prior build_rag_index run's own index instead.")
    args = parser.parse_args(argv)

    settings = get_settings()
    if not args.run_api:
        run_safe(settings)
        return

    run_id = args.run_id or new_run_id()
    run_expensive(settings, run_id, args.index_run_id, args.use_frozen_index, args.input_run_id)


if __name__ == "__main__":
    main()
