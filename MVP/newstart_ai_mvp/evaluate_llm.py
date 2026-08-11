"""Stage: plain-Gemini classification sweep, no retrieval (Checkpoint 10, no-RAG half).

Real logic lives in newstart_ai_mvp.llm_pipeline (run_plain_llm_case, truncate_for_llm,
build_method_condition_metrics, GeminiProvider). temperature is fixed at 0; inputs longer
than 6,000 characters are truncated (truncate_for_llm), each truncation recorded as a
boolean.

Default mode recomputes the primary-condition macro F1 from the real, frozen 990-row
checkpoint10_llm_predictions.jsonl and compares it to the saved manifest value.

--run-api calls the real Gemini API for every (document, condition) pair. Its cache starts
EMPTY under the run directory (redirect_frozen_outputs also redirects the cache_dir) --
this is intentional: a rerun never silently reuses the frozen, submitted cache.

    python -m newstart_ai_mvp.evaluate_llm                # safe (default)
    python -m newstart_ai_mvp.evaluate_llm --run-api        # real Gemini calls
"""

from __future__ import annotations

from newstart_ai_mvp import artifact_report as ar
from newstart_ai_mvp.cli_common import build_base_parser, get_settings, print_expensive_mode_banner, print_safe_mode_banner
from newstart_ai_mvp.run_scope import new_run_id, redirect_frozen_outputs, run_root

STAGE_NAME = "evaluate_llm"


def run_safe(settings) -> None:
    print_safe_mode_banner(STAGE_NAME)
    ar.print_report("Plain-LLM predictions (recomputed vs. reported)", ar.describe_llm_predictions(settings))


def run_expensive(settings, run_id: str, input_run_id: str | None) -> None:
    import json

    import pandas as pd

    from newstart_ai_mvp.llm_pipeline import GeminiProvider, build_method_condition_metrics, load_classification_prompt, run_plain_llm_case

    print_expensive_mode_banner(STAGE_NAME, run_id)
    print(f"[{STAGE_NAME}] This calls the real Gemini API for up to 990 (document, condition)")
    print(f"[{STAGE_NAME}] pairs -- the run's own cache starts empty, so nothing is skipped.\n")

    if input_run_id:
        conditions_dir = run_root(input_run_id) / settings.family_aware.conditions.output_dir
    else:
        conditions_dir = settings.resolve_path(settings.family_aware.conditions.output_dir)
    registry = pd.read_csv(conditions_dir / "condition_registry_test.csv")

    provider = GeminiProvider(settings)
    prompt = load_classification_prompt(settings)

    cases = []
    with redirect_frozen_outputs(run_id):
        for row in registry.itertuples(index=False):
            condition_fingerprint = row.text_fingerprint
            result = run_plain_llm_case(
                document_id=str(row.document_id),
                effective_family_id=row.effective_agency,
                condition=row.condition,
                true_label=row.effective_agency,
                text=row.text,
                condition_fingerprint=condition_fingerprint,
                llm_provider=provider,
                prompt=prompt,
                settings=settings,
            )
            cases.append(result)

        metrics = build_method_condition_metrics(cases, list(settings.base.labels))
        reports_dir = settings.resolve_path("artifacts/family_aware/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        with open(reports_dir / "checkpoint10_llm_predictions.jsonl", "w", encoding="utf-8") as f:
            for case in cases:
                f.write(json.dumps(case, default=str) + "\n")

    print(f"Ran {len(cases)} cases. Primary macro F1: {metrics['document_macro_f1']:.4f}")
    print(f"Written under MVP/runs/{run_id}/ -- frozen predictions untouched.")


def main(argv: list[str] | None = None) -> None:
    parser = build_base_parser(__doc__ or "")
    parser.add_argument("--run-api", action="store_true", help="Actually call the Gemini API for every case.")
    args = parser.parse_args(argv)

    settings = get_settings()
    if not args.run_api:
        run_safe(settings)
        return

    run_id = args.run_id or new_run_id()
    run_expensive(settings, run_id, args.input_run_id)


if __name__ == "__main__":
    main()
