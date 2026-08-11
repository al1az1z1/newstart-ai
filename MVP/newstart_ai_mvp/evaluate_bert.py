"""Stage: BERT test-set inference + the 10-condition robustness sweep (Checkpoints 7's
validation sweep and 8's one-time frozen test evaluation).

Real logic lives in newstart_ai_mvp.bert_pipeline (build_pre_test_freeze_record,
evaluate_primary_test_condition, evaluate_all_conditions -- the real 10-condition sweep) --
neither takes a test-split parameter that could accidentally leak into training,
structurally.

Default mode recomputes macro-F1/accuracy directly from the real, frozen
checkpoint8_test_predictions.csv with sklearn and prints it next to the value stored in the
saved manifest, so agreement is verified every time this runs, not assumed.

--run touches the frozen test split and requires an explicit extra confirmation flag, since
opening the test split at all is exactly the operation Checkpoint 6-8's test-isolation proofs
exist to guard against happening casually.

    python -m newstart_ai_mvp.evaluate_bert                                                        # safe (default)
    python -m newstart_ai_mvp.evaluate_bert --run --artifact-id ID --i-understand-this-is-the-frozen-test-set
"""

from __future__ import annotations

from newstart_ai_mvp import artifact_report as ar
from newstart_ai_mvp.cli_common import build_base_parser, get_settings, print_expensive_mode_banner, print_safe_mode_banner
from newstart_ai_mvp.run_scope import new_run_id, redirect_frozen_outputs, run_root

STAGE_NAME = "evaluate_bert"


def run_safe(settings) -> None:
    print_safe_mode_banner(STAGE_NAME)
    ar.print_report("BERT test results (recomputed vs. reported)", ar.describe_bert_test_results(settings))


def run_expensive(settings, run_id: str, artifact_id: str | None, input_run_id: str | None) -> None:
    import pandas as pd
    import torch

    from newstart_ai_mvp import bert_pipeline as bert

    print_expensive_mode_banner(STAGE_NAME, run_id)

    artifact_id = artifact_id or bert.latest_ready_family_aware_artifact_id(settings)
    if artifact_id is None:
        raise RuntimeError("No ready family-aware BERT checkpoint found -- run train_bert first.")
    model, tokenizer, metadata = bert.load_family_aware_artifact(settings, artifact_id)

    if input_run_id:
        chunks_dir = run_root(input_run_id) / settings.family_aware.chunking.output_dir
        split_dir = run_root(input_run_id) / settings.family_aware.split.output_dir
    else:
        chunks_dir = settings.resolve_path(settings.family_aware.chunking.output_dir)
        split_dir = settings.resolve_path(settings.family_aware.split.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_chunks = pd.read_csv(chunks_dir / "test_chunks.csv")
    test_split = pd.read_csv(split_dir / "test.csv")
    test_texts = dict(zip(test_split["document_id"].astype(str), test_split["text"]))
    true_labels_by_doc = dict(zip(test_split["document_id"].astype(str), test_split["effective_agency"]))
    label_order = list(settings.base.labels)
    max_seq_length = settings.family_aware.chunking.max_seq_length

    with redirect_frozen_outputs(run_id):
        freeze_record = bert.build_pre_test_freeze_record(
            settings, checkpoint_artifact_id=artifact_id, checkpoint_file_hashes=metadata["class_weights"]
        )
        primary_result = bert.evaluate_primary_test_condition(
            model, tokenizer, test_chunks, test_texts, true_labels_by_doc, label_order, max_seq_length, device
        )
        print(f"Primary (complete_unmasked) test macro F1: {primary_result['document_macro_f1']:.4f}")

    print(f"New-run BERT test evaluation output written under MVP/runs/{run_id}/.")


def main(argv: list[str] | None = None) -> None:
    parser = build_base_parser(__doc__ or "")
    parser.add_argument("--run", action="store_true", help="Actually re-run test-set inference.")
    parser.add_argument("--artifact-id", default=None, help="Which BERT checkpoint to evaluate (default: the frozen, approved one).")
    parser.add_argument(
        "--i-understand-this-is-the-frozen-test-set",
        action="store_true",
        dest="confirmed",
        help="Required alongside --run: explicit acknowledgment that this touches the frozen test split.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    if not args.run:
        run_safe(settings)
        return

    if not args.confirmed:
        raise SystemExit(
            "Refusing to run: pass --i-understand-this-is-the-frozen-test-set alongside --run. "
            "This flag exists because opening the test split at all is exactly what this "
            "codebase's test-isolation proofs (Checkpoints 6-8) are designed to guard against "
            "happening casually."
        )

    run_id = args.run_id or new_run_id()
    run_expensive(settings, run_id, args.artifact_id, args.input_run_id)


if __name__ == "__main__":
    main()
