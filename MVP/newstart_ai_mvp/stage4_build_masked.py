"""Stage 4: identifier masking (Checkpoint 6).

Real masking LOGIC lives in newstart_ai_mvp.data_pipeline (build_masking_rules,
apply_masking, build_masked_documents, build_masked_chunks, build_masking_manifest). That
module has no save_*() function for this stage's output, so this module adds the small
persistence glue that writes it in the exact on-disk layout, matching
data/family_aware_masked/{split}_masked_{documents,chunks}.csv and
artifacts/family_aware/manifests/masking_policy_v1.json.

    python -m newstart_ai_mvp.stage4_build_masked                  # safe (default)
    python -m newstart_ai_mvp.stage4_build_masked --run             # re-executes
"""

from __future__ import annotations

import json

import pandas as pd

from newstart_ai_mvp import artifact_report as ar
from newstart_ai_mvp.cli_common import build_base_parser, get_settings, print_expensive_mode_banner, print_safe_mode_banner
from newstart_ai_mvp.run_scope import new_run_id, redirect_frozen_outputs, run_root

STAGE_NAME = "stage4_build_masked"


def run_safe(settings) -> None:
    print_safe_mode_banner(STAGE_NAME)
    ar.print_report("Masked documents", ar.describe_masked(settings))


def _save_masked_artifacts(settings, masked_docs_by_split, masked_chunks_by_split, manifest: dict) -> None:
    """New persistence glue -- data_pipeline.py's masking functions have no save_*() of
    their own. Writes into the exact layout already on disk under data/family_aware_masked/."""
    output_dir = settings.resolve_path(settings.family_aware.masking.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "validation", "test"):
        masked_docs_by_split[split_name].to_csv(output_dir / f"{split_name}_masked_documents.csv", index=False)
        masked_chunks_by_split[split_name].to_csv(output_dir / f"{split_name}_masked_chunks.csv", index=False)

    manifests_dir = settings.resolve_path("artifacts/family_aware/manifests")
    manifests_dir.mkdir(parents=True, exist_ok=True)
    with open(manifests_dir / "masking_policy_v1.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)


def run_expensive(settings, run_id: str, input_run_id: str | None) -> None:
    from newstart_ai_mvp import data_pipeline as data

    print_expensive_mode_banner(STAGE_NAME, run_id)

    if input_run_id:
        split_dir = run_root(input_run_id) / settings.family_aware.split.output_dir
        chunks_dir = run_root(input_run_id) / settings.family_aware.chunking.output_dir
    else:
        split_dir = settings.resolve_path(settings.family_aware.split.output_dir)
        chunks_dir = settings.resolve_path(settings.family_aware.chunking.output_dir)

    with redirect_frozen_outputs(run_id):
        masked_docs_by_split: dict[str, pd.DataFrame] = {}
        masked_chunks_by_split: dict[str, pd.DataFrame] = {}
        audit_examples: list[dict] = []
        for split_name in ("train", "validation", "test"):
            split_df = pd.read_csv(split_dir / f"{split_name}.csv")
            chunks_df = pd.read_csv(chunks_dir / f"{split_name}_chunks.csv")
            masked_docs_by_split[split_name] = data.build_masked_documents(split_df, split_name, settings)
            masked_chunks_by_split[split_name] = data.build_masked_chunks(chunks_df, settings)

        # One audit example per agency with at least one replacement, for the manifest --
        # derived from the real data, never hand-picked/invented.
        all_masked = pd.concat(masked_docs_by_split.values(), ignore_index=True)
        for agency in sorted(all_masked["agency"].unique()):
            candidates = all_masked[(all_masked["agency"] == agency) & (all_masked["total_replacements"] > 0)]
            if len(candidates):
                row = candidates.iloc[0]
                audit_examples.append({"document_id": row["document_id"], "agency": agency})

        manifest = data.build_masking_manifest(masked_docs_by_split, audit_examples, settings)
        _save_masked_artifacts(settings, masked_docs_by_split, masked_chunks_by_split, manifest)

    total_docs = sum(len(df) for df in masked_docs_by_split.values())
    print(f"Wrote masked documents+chunks for {total_docs} documents under MVP/runs/{run_id}/.")


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
