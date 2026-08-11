"""Stage 5: partial-input selection + the shared 10-condition registry (Checkpoint 6).

Real logic lives in newstart_ai_mvp.data_pipeline (build_partial_input_selections,
build_condition_registry). Neither has a save_*() function of its own, so this module adds
the persistence glue, matching the exact on-disk layout: data/family_aware_conditions/
{partial_input_selections,test_partial_input_selections,condition_registry_train_validation,
condition_registry_test}.csv and the three matching manifests. Note: PartialInputConfig has
no output_dir field of its own (verified in config.py) -- selections share
family_aware.conditions.output_dir, exactly matching the real frozen layout.

    python -m newstart_ai_mvp.stage5_build_conditions                  # safe (default)
    python -m newstart_ai_mvp.stage5_build_conditions --run             # re-executes
"""

from __future__ import annotations

import json

import pandas as pd

from newstart_ai_mvp import artifact_report as ar
from newstart_ai_mvp.cli_common import build_base_parser, get_settings, print_expensive_mode_banner, print_safe_mode_banner
from newstart_ai_mvp.run_scope import new_run_id, redirect_frozen_outputs, run_root

STAGE_NAME = "stage5_build_conditions"


def run_safe(settings) -> None:
    print_safe_mode_banner(STAGE_NAME)
    ar.print_report("Conditions", ar.describe_conditions(settings))


def run_expensive(settings, run_id: str, input_run_id: str | None) -> None:
    from newstart_ai_mvp import data_pipeline as data

    print_expensive_mode_banner(STAGE_NAME, run_id)

    if input_run_id:
        split_dir = run_root(input_run_id) / settings.family_aware.split.output_dir
        chunks_dir = run_root(input_run_id) / settings.family_aware.chunking.output_dir
        masked_dir = run_root(input_run_id) / settings.family_aware.masking.output_dir
    else:
        split_dir = settings.resolve_path(settings.family_aware.split.output_dir)
        chunks_dir = settings.resolve_path(settings.family_aware.chunking.output_dir)
        masked_dir = settings.resolve_path(settings.family_aware.masking.output_dir)

    with redirect_frozen_outputs(run_id):
        selections_by_split: dict[str, pd.DataFrame] = {}
        registry_by_split: dict[str, pd.DataFrame] = {}
        for split_name in ("train", "validation", "test"):
            split_df = pd.read_csv(split_dir / f"{split_name}.csv")
            chunks_df = pd.read_csv(chunks_dir / f"{split_name}_chunks.csv")
            masked_docs_df = pd.read_csv(masked_dir / f"{split_name}_masked_documents.csv")
            masked_chunks_df = pd.read_csv(masked_dir / f"{split_name}_masked_chunks.csv")

            selections = data.build_partial_input_selections(chunks_df, split_name, settings)
            selections_by_split[split_name] = selections
            registry_by_split[split_name] = data.build_condition_registry(
                split_df, masked_docs_df, chunks_df, masked_chunks_df, selections, split_name, settings
            )

        train_val_selections = pd.concat(
            [selections_by_split["train"], selections_by_split["validation"]], ignore_index=True
        )
        train_val_registry = pd.concat(
            [registry_by_split["train"], registry_by_split["validation"]], ignore_index=True
        )
        test_selections = selections_by_split["test"]
        test_registry = registry_by_split["test"]

        partial_input_manifest = data.build_partial_input_manifest(
            pd.concat([train_val_selections, test_selections], ignore_index=True), settings
        )
        train_val_registry_manifest = data.build_condition_registry_manifest(train_val_registry, settings)
        test_registry_manifest = data.build_condition_registry_manifest(test_registry, settings)

        output_dir = settings.resolve_path(settings.family_aware.conditions.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        train_val_selections.to_csv(output_dir / "partial_input_selections.csv", index=False)
        test_selections.to_csv(output_dir / "test_partial_input_selections.csv", index=False)
        train_val_registry.to_csv(output_dir / "condition_registry_train_validation.csv", index=False)
        test_registry.to_csv(output_dir / "condition_registry_test.csv", index=False)

        manifests_dir = settings.resolve_path("artifacts/family_aware/manifests")
        manifests_dir.mkdir(parents=True, exist_ok=True)
        with open(manifests_dir / "partial_input_policy_v1.json", "w", encoding="utf-8") as f:
            json.dump(partial_input_manifest, f, indent=2, ensure_ascii=False, default=str)
        with open(manifests_dir / "condition_registry_v1.json", "w", encoding="utf-8") as f:
            json.dump(train_val_registry_manifest, f, indent=2, ensure_ascii=False, default=str)
        with open(manifests_dir / "condition_registry_test_v1.json", "w", encoding="utf-8") as f:
            json.dump(test_registry_manifest, f, indent=2, ensure_ascii=False, default=str)

    print(
        f"Wrote condition registry under MVP/runs/{run_id}/: "
        f"train+validation={len(train_val_registry)} rows, test={len(test_registry)} rows "
        f"(10 conditions x document count per split)."
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
