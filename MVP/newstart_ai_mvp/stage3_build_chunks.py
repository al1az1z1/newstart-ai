"""Stage 3: tokenizer-aware overlapping chunking (Checkpoint 5).

Real logic lives in newstart_ai.data.chunking. Default mode describes the real, frozen
chunks (data/family_aware_chunks/{train,validation,test}_chunks.csv). --run re-executes
chunking against the frozen split (or a previous run's own split via --input-run-id),
writing only under MVP/runs/<run-id>/.

    python -m newstart_ai_mvp.stage3_build_chunks                  # safe (default)
    python -m newstart_ai_mvp.stage3_build_chunks --run             # re-executes
"""

from __future__ import annotations

import pandas as pd

from newstart_ai_mvp import artifact_report as ar
from newstart_ai_mvp.cli_common import build_base_parser, get_settings, print_expensive_mode_banner, print_safe_mode_banner
from newstart_ai_mvp.run_scope import new_run_id, redirect_frozen_outputs, run_root

STAGE_NAME = "stage3_build_chunks"


def run_safe(settings) -> None:
    print_safe_mode_banner(STAGE_NAME)
    ar.print_report("Chunks", ar.describe_chunks(settings))


def _load_upstream_split(settings, input_run_id: str | None) -> dict[str, pd.DataFrame]:
    if input_run_id:
        split_dir = run_root(input_run_id) / settings.family_aware.split.output_dir
    else:
        split_dir = settings.resolve_path(settings.family_aware.split.output_dir)
    return {name: pd.read_csv(split_dir / f"{name}.csv") for name in ("train", "validation", "test")}


def run_expensive(settings, run_id: str, input_run_id: str | None) -> None:
    from newstart_ai_mvp import data_pipeline as data

    print_expensive_mode_banner(STAGE_NAME, run_id)

    splits = _load_upstream_split(settings, input_run_id)
    eligible_df = pd.concat(splits.values(), ignore_index=True)
    audit_df = pd.read_csv(settings.resolve_path("artifacts/family_aware/reports/family_audit_v1.csv"))

    with redirect_frozen_outputs(run_id):
        train_chunks, val_chunks, test_chunks = data.build_all_split_chunks(
            splits["train"], splits["validation"], splits["test"], settings
        )

        data.assert_every_chunk_maps_to_one_eligible_parent(
            pd.concat([train_chunks, val_chunks, test_chunks], ignore_index=True), eligible_df
        )
        document_to_split = {
            str(doc_id): name for name, df in splits.items() for doc_id in df["document_id"]
        }
        data.assert_every_chunk_inherits_parent_split(
            pd.concat([train_chunks, val_chunks, test_chunks], ignore_index=True), document_to_split
        )
        data.assert_no_cross_split_leakage(train_chunks, val_chunks, test_chunks)
        data.assert_no_excluded_document_chunked(
            pd.concat([train_chunks, val_chunks, test_chunks], ignore_index=True), audit_df
        )
        for chunks in (train_chunks, val_chunks, test_chunks):
            data.assert_no_duplicate_chunk_ids(chunks)
            data.assert_chunk_indices_contiguous_and_unique(chunks)
            data.assert_no_empty_chunks(chunks)
        data.assert_every_eligible_document_has_at_least_one_chunk(
            eligible_df, pd.concat([train_chunks, val_chunks, test_chunks], ignore_index=True)
        )

        split_fingerprints = {name: data.fingerprint_split(df) for name, df in splits.items()}
        report = data.build_chunk_report(
            eligible_df,
            splits["train"],
            splits["validation"],
            splits["test"],
            train_chunks,
            val_chunks,
            test_chunks,
            audit_df,
            split_fingerprints,
            settings,
        )
        output_dir = data.save_family_aware_chunks(train_chunks, val_chunks, test_chunks, report, settings)

    print(
        f"Wrote chunks under {output_dir}: "
        f"train={len(train_chunks)} validation={len(val_chunks)} test={len(test_chunks)}; "
        f"all 9 chunking invariants verified."
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_base_parser(__doc__ or "")
    parser.add_argument("--run", action="store_true", help="Actually re-execute this stage (writes under MVP/runs/).")
    args = parser.parse_args(argv)

    settings = get_settings()
    if not args.run:
        run_safe(settings)
        return

    run_id = args.run_id or new_run_id()
    run_expensive(settings, run_id, args.input_run_id)


if __name__ == "__main__":
    main()
