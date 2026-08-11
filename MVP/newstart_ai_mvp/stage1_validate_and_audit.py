"""Stage 1: dataset validation, language filtering, family grouping and audit.

Real pipeline logic lives in newstart_ai_mvp.data_pipeline -- this module only adds a CLI
entry point around it. Default mode describes the real, frozen artifacts
(artifacts/family_aware/reports/{language_audit_v1,family_audit_v1}.csv and their
manifests). --run actually re-executes the pipeline, writing under MVP/runs/<run-id>/ only.

    python -m newstart_ai_mvp.stage1_validate_and_audit                 # safe (default)
    python -m newstart_ai_mvp.stage1_validate_and_audit --run           # re-executes
"""

from __future__ import annotations

from newstart_ai_mvp import artifact_report as ar
from newstart_ai_mvp.cli_common import build_base_parser, get_settings, print_expensive_mode_banner, print_safe_mode_banner
from newstart_ai_mvp.run_scope import new_run_id, redirect_frozen_outputs

STAGE_NAME = "stage1_validate_and_audit"


def run_safe(settings) -> None:
    print_safe_mode_banner(STAGE_NAME)
    ar.print_report("Language audit", ar.describe_language_audit(settings))
    ar.print_report("Family audit", ar.describe_family_audit(settings))


def run_expensive(settings, run_id: str) -> None:
    from newstart_ai_mvp import data_pipeline as data

    print_expensive_mode_banner(STAGE_NAME, run_id)

    with redirect_frozen_outputs(run_id):
        # -- Dataset validation (pure; no artifact of its own) --
        df = data.load_dataset(settings)
        validation_report = data.validate_dataset(df, settings)
        print(f"validate_dataset: {len(df)} documents, has_critical_errors={validation_report['has_critical_errors']}")

        # -- Language filtering (Checkpoint 2) --
        language_audit_df = data.build_language_audit(df, settings)
        language_manifest = data.build_language_filter_manifest(language_audit_df, df, settings)
        data.save_language_audit(language_audit_df, language_manifest, settings)

        # -- Family grouping + audit (Checkpoint 3) --
        full_audit_df = data.build_full_family_audit(df, language_audit_df, settings)
        category_reports = data.build_category_reports(full_audit_df)
        family_manifest = data.build_family_audit_manifest(full_audit_df, df, settings)
        reports_dir = data.save_family_audit(full_audit_df, category_reports, family_manifest, settings)
        print(f"Wrote family audit + language audit under {reports_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = build_base_parser(__doc__ or "")
    parser.add_argument("--run", action="store_true", help="Actually re-execute this stage (writes under MVP/runs/).")
    args = parser.parse_args(argv)

    settings = get_settings()
    if not args.run:
        run_safe(settings)
        return

    run_id = args.run_id or new_run_id()
    run_expensive(settings, run_id)


if __name__ == "__main__":
    main()
