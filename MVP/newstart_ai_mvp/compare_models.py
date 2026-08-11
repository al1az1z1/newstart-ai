"""Always-safe: reads the real, frozen Checkpoint 10 comparison manifests and prints (or
exports) the head-to-head and robustness tables. Never recomputes the comparison logic
itself (that's newstart_ai_mvp.llm_pipeline.build_method_condition_metrics -- this module
only reads its already-saved output); no flag, no expensive mode, nothing to guard.

    python -m newstart_ai_mvp.compare_models
    python -m newstart_ai_mvp.compare_models --out comparison.csv
"""

from __future__ import annotations

import json

import pandas as pd

from newstart_ai_mvp.cli_common import build_base_parser, get_settings


def load_comparison_tables(settings) -> dict[str, pd.DataFrame]:
    manifests_dir = settings.resolve_path("artifacts/family_aware/manifests")

    with open(manifests_dir / "checkpoint10_method_condition_metrics_v1.json", encoding="utf-8") as f:
        method_condition_metrics = json.load(f)
    with open(manifests_dir / "checkpoint10_robustness_comparison_v1.json", encoding="utf-8") as f:
        robustness = json.load(f)
    with open(manifests_dir / "checkpoint10_primary_paired_comparison_v1.json", encoding="utf-8") as f:
        primary_paired = json.load(f)

    per_condition = pd.DataFrame(
        [
            {
                "method": m["method"],
                "condition": m["condition"],
                "document_macro_f1": m["document_macro_f1"],
                "document_accuracy": m["document_accuracy"],
                "error_count": m["error_count"],
            }
            for m in method_condition_metrics
        ]
    )
    return {
        "per_condition": per_condition,
        "robustness_manifest": pd.json_normalize(robustness),
        "primary_paired_comparison": pd.json_normalize(primary_paired),
    }


def main(argv: list[str] | None = None) -> None:
    parser = build_base_parser(__doc__ or "")
    parser.add_argument("--out", default=None, help="Write the per-condition comparison table to this CSV path instead of printing.")
    args = parser.parse_args(argv)

    settings = get_settings()
    tables = load_comparison_tables(settings)

    pivot = tables["per_condition"].pivot(index="condition", columns="method", values="document_macro_f1")
    if args.out:
        pivot.to_csv(args.out)
        print(f"Wrote {args.out}")
    else:
        print("\n=== Macro F1 by condition and method (frozen Checkpoint 10 results) ===")
        print(pivot.to_string())


if __name__ == "__main__":
    main()
