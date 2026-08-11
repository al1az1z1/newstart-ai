"""Orchestrates Stage 1 -> Stage 5 (data validation through condition registry) under one
run_id, so a single command reproduces the entire data-preparation half of the pipeline.
Each stage remains independently runnable via its own module for finer-grained control.

    python -m newstart_ai_mvp.prepare_data                  # safe (default): 5 status summaries
    python -m newstart_ai_mvp.prepare_data --run             # re-executes stages 1-5 in order
"""

from __future__ import annotations

from newstart_ai_mvp import (
    stage1_validate_and_audit,
    stage2_build_split,
    stage3_build_chunks,
    stage4_build_masked,
    stage5_build_conditions,
)
from newstart_ai_mvp.cli_common import build_base_parser, get_settings
from newstart_ai_mvp.run_scope import new_run_id

STAGES = [
    stage1_validate_and_audit,
    stage2_build_split,
    stage3_build_chunks,
    stage4_build_masked,
    stage5_build_conditions,
]


def run_safe(settings) -> None:
    for stage in STAGES:
        stage.run_safe(settings)


def main(argv: list[str] | None = None) -> None:
    parser = build_base_parser(__doc__ or "")
    parser.add_argument("--run", action="store_true", help="Actually re-execute stages 1-5 (writes under MVP/runs/).")
    args = parser.parse_args(argv)

    settings = get_settings()
    if not args.run:
        run_safe(settings)
        return

    run_id = args.run_id or new_run_id()
    print(f"Running stages 1-5 under a single run_id={run_id}\n")
    # Each later stage reads THIS run's own upstream output (input_run_id=run_id), not the
    # frozen artifacts, so the five stages compose into one internally-consistent run.
    stage1_validate_and_audit.run_expensive(settings, run_id)
    stage2_build_split.run_expensive(settings, run_id, input_run_id=run_id)
    stage3_build_chunks.run_expensive(settings, run_id, input_run_id=run_id)
    stage4_build_masked.run_expensive(settings, run_id, input_run_id=run_id)
    stage5_build_conditions.run_expensive(settings, run_id, input_run_id=run_id)
    print(f"\nAll 5 data-preparation stages complete under MVP/runs/{run_id}/.")


if __name__ == "__main__":
    main()
