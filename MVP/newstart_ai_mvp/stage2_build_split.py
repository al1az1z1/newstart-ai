"""Stage 2: frozen family-aware split construction (Checkpoint 4).

Real logic lives in newstart_ai.data.family_split -- this module only adds a CLI entry point.
Default mode describes the real, frozen split (data/family_aware_splits/{train,validation,
test}.csv). --run re-executes the split (same seed from configs/family_aware.yaml) and proves
zero document/family overlap via the same 4 assertion functions used originally, writing only
under MVP/runs/<run-id>/.

    python -m newstart_ai_mvp.stage2_build_split                  # safe (default)
    python -m newstart_ai_mvp.stage2_build_split --run             # re-executes
"""

from __future__ import annotations

from newstart_ai_mvp import artifact_report as ar
from newstart_ai_mvp.cli_common import build_base_parser, get_settings, print_expensive_mode_banner, print_safe_mode_banner
from newstart_ai_mvp.run_scope import new_run_id, redirect_frozen_outputs, run_root

STAGE_NAME = "stage2_build_split"


def run_safe(settings) -> None:
    print_safe_mode_banner(STAGE_NAME)
    ar.print_report("Family-aware split", ar.describe_split(settings))


def run_expensive(
    settings, run_id: str, override_version: str = "v2", input_run_id: str | None = None
) -> None:
    from newstart_ai_mvp import data_pipeline as data

    print_expensive_mode_banner(STAGE_NAME, run_id)

    # Upstream inputs default to the real, frozen, approved family audit -- pass
    # --input-run-id to instead build the split from a specific prior run's own audit
    # output (e.g. when chained from prepare_data --run, which passes the same run_id).
    df = data.load_dataset(settings)
    if input_run_id:
        reports_dir = run_root(input_run_id) / "artifacts/family_aware/reports"
        manifests_dir = run_root(input_run_id) / "artifacts/family_aware/manifests"
    else:
        reports_dir = settings.resolve_path("artifacts/family_aware/reports")
        manifests_dir = settings.resolve_path("artifacts/family_aware/manifests")
    audit_df = __import__("pandas").read_csv(reports_dir / "family_audit_v1.csv")
    override_path = manifests_dir / f"family_overrides_{override_version}.json"

    with redirect_frozen_outputs(run_id):
        eligible_df = data.build_eligible_corpus(audit_df, df, settings)
        train_df, val_df, test_df, family_to_split = data.create_family_aware_split(eligible_df, settings)

        data.assert_no_document_overlap(train_df, val_df, test_df)
        data.assert_no_family_overlap(train_df, val_df, test_df)
        data.assert_every_eligible_document_assigned_exactly_once(eligible_df, train_df, val_df, test_df)
        data.assert_no_excluded_document_in_splits(audit_df, train_df, val_df, test_df)

        report = data.build_split_report(
            eligible_df, train_df, val_df, test_df, audit_df, df, override_path, override_version, settings
        )
        output_dir = data.save_family_split(train_df, val_df, test_df, report, settings)

    print(
        f"Wrote split under {output_dir}: "
        f"train={len(train_df)} validation={len(val_df)} test={len(test_df)}; "
        f"zero document/family overlap verified by the same 4 assertions used originally."
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_base_parser(__doc__ or "")
    parser.add_argument("--run", action="store_true", help="Actually re-execute this stage (writes under MVP/runs/).")
    parser.add_argument(
        "--override-version",
        default="v2",
        help="Which frozen family_overrides_<version>.json to build the split from (default: v2, the approved one).",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    if not args.run:
        run_safe(settings)
        return

    run_id = args.run_id or new_run_id()
    run_expensive(settings, run_id, override_version=args.override_version, input_run_id=args.input_run_id)


if __name__ == "__main__":
    main()
