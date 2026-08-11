"""Stage: family-aware BERT training + checkpoint selection (Checkpoint 7).

Real logic lives in newstart_ai_mvp.bert_pipeline (train_family_aware_bert,
FamilyAwareChunkDataset, weighted_cross_entropy, class/document weighting, checkpoint
save/load).

Default mode reads the real, frozen checkpoint's metadata.json (training history, selected
epoch, class weights) -- no model weights are loaded, no GPU is touched.

--run-training actually fine-tunes BERT (same seed/LR/batch/optimizer/metric/tie-rule as the
frozen run, read unchanged from configs/family_aware.yaml) and saves a NEW checkpoint under
MVP/runs/<run-id>/artifacts/family_aware/models/<new-artifact-id>/ -- this is a genuinely
expensive, GPU/CPU-bound operation and is never triggered by default.

    python -m newstart_ai_mvp.train_bert                    # safe (default)
    python -m newstart_ai_mvp.train_bert --run-training       # actually trains
"""

from __future__ import annotations

from newstart_ai_mvp import artifact_report as ar
from newstart_ai_mvp.cli_common import build_base_parser, get_settings, print_expensive_mode_banner, print_safe_mode_banner
from newstart_ai_mvp.run_scope import new_run_id, redirect_frozen_outputs, run_root

STAGE_NAME = "train_bert"


def run_safe(settings) -> None:
    print_safe_mode_banner(STAGE_NAME)
    ar.print_report("Family-aware BERT checkpoint", ar.describe_bert_checkpoint(settings))


def run_expensive(settings, run_id: str, input_run_id: str | None) -> None:
    import pandas as pd
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from newstart_ai_mvp import bert_pipeline as bert
    from newstart_ai_mvp.data_pipeline import fingerprint_chunks, fingerprint_split

    print_expensive_mode_banner(STAGE_NAME, run_id)

    if input_run_id:
        split_dir = run_root(input_run_id) / settings.family_aware.split.output_dir
        chunks_dir = run_root(input_run_id) / settings.family_aware.chunking.output_dir
    else:
        split_dir = settings.resolve_path(settings.family_aware.split.output_dir)
        chunks_dir = settings.resolve_path(settings.family_aware.chunking.output_dir)

    train_split = pd.read_csv(split_dir / "train.csv")
    val_split = pd.read_csv(split_dir / "validation.csv")
    train_chunks = pd.read_csv(chunks_dir / "train_chunks.csv")
    val_chunks = pd.read_csv(chunks_dir / "validation_chunks.csv")
    label_order = list(settings.base.labels)

    non_deterministic_warnings = bert.set_determinism(settings.family_aware.training.random_seed)

    class_weight_manifest = bert.build_agency_class_weight_manifest(train_split, label_order, settings)
    bert.build_document_balancing_manifest(train_chunks, settings)

    tokenizer = AutoTokenizer.from_pretrained(settings.bert.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(settings.bert.base_model, num_labels=len(label_order))

    train_texts = dict(zip(train_split["document_id"].astype(str), train_split["text"]))
    val_texts = dict(zip(val_split["document_id"].astype(str), val_split["text"]))

    result = bert.train_family_aware_bert(
        model, tokenizer, train_chunks, train_texts, val_chunks, val_texts, label_order,
        class_weight_manifest["raw_weights"], settings,
    )

    with redirect_frozen_outputs(run_id):
        artifact_id = bert.new_family_aware_artifact_id()
        metadata = {
            "artifact_id": artifact_id, "display_name": "family-aware-chunked-bert",
            "base_model": settings.bert.base_model, "tokenizer_revision": settings.family_aware.chunking.tokenizer_revision,
            "label_order": label_order,
            "source_train_chunk_fingerprint": fingerprint_chunks(train_chunks),
            "source_validation_chunk_fingerprint": fingerprint_chunks(val_chunks),
            "source_train_split_fingerprint": fingerprint_split(train_split),
            "source_validation_split_fingerprint": fingerprint_split(val_split),
            "chunking_policy_version": settings.family_aware.chunking.chunking_policy_version,
            "document_balancing_policy_version": settings.family_aware.document_balancing.policy_version,
            "random_seed": settings.family_aware.training.random_seed,
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "deterministic_algorithms_warnings": non_deterministic_warnings,
            "training_config": {
                "max_epochs": settings.family_aware.training.max_epochs, "batch_size": settings.family_aware.training.batch_size,
                "learning_rate": settings.family_aware.training.learning_rate,
                "checkpoint_selection_metric": settings.family_aware.training.checkpoint_selection_metric,
                "checkpoint_selection_aggregation_method": settings.family_aware.training.checkpoint_selection_aggregation_method,
            },
            "class_weights": class_weight_manifest["raw_weights"], "history": result["history"],
            "best_epoch": result["best_epoch"], "stopping_reason": result["stopping_reason"],
            "checkpoint_selection_metric": settings.family_aware.training.checkpoint_selection_metric,
            "checkpoint_selection_aggregation_method": settings.family_aware.training.checkpoint_selection_aggregation_method,
            "best_validation_document_macro_f1": result["best_validation_document_macro_f1"],
            "status": "ready", "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        }
        metadata["ready_at"] = metadata["created_at"]
        out_dir = bert.save_family_aware_artifact(model, tokenizer, metadata, settings)

    print(f"Trained and saved new checkpoint {artifact_id} under {out_dir}")
    print(f"Best epoch: {result['best_epoch']}  Best validation macro F1: {result['best_validation_document_macro_f1']:.4f}")


def main(argv: list[str] | None = None) -> None:
    parser = build_base_parser(__doc__ or "")
    parser.add_argument(
        "--run-training",
        action="store_true",
        help="Actually fine-tune BERT (GPU/CPU-bound, several minutes; writes under MVP/runs/).",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    if not args.run_training:
        run_safe(settings)
        return

    run_id = args.run_id or new_run_id()
    run_expensive(settings, run_id, args.input_run_id)


if __name__ == "__main__":
    main()
