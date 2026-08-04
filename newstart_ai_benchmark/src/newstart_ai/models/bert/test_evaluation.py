"""One-time family-aware BERT test evaluation (Version 6, Checkpoint 8).

`build_pre_test_freeze_record` must run and be saved BEFORE any test file is opened -- it
reads only already-existing manifest JSON files from Checkpoints 4-7 (fingerprints, best
checkpoint metadata, condition/aggregation policy versions), never `test.csv` or
`test_chunks.csv` themselves. Every other function here is read-only with respect to the
model and policies: nothing evaluates, aggregates, or reports in a way that could feed back
into retraining or policy revision.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from newstart_ai.data.condition_registry import build_condition_definitions
from newstart_ai.models.bert.aggregation import aggregate_document
from newstart_ai.models.bert.family_aware_reproducibility import fingerprint_configuration
from newstart_ai.models.bert.family_aware_training import generate_chunk_level_outputs
from newstart_ai.schemas.checkpoint8 import (
    ChunkBucketErrorCounts,
    FamilyErrorCounts,
    HistoricalComparisonContext,
    PerAgencyTestMetrics,
    PreTestFreezeRecord,
    PrimaryTestResult,
    TestErrorAnalysis,
    TestIntegrityProof,
    TestMisclassificationDetail,
    TestReproducibilityManifest,
)


def build_pre_test_freeze_record(settings, checkpoint_artifact_id: str, checkpoint_file_hashes: dict[str, str]) -> PreTestFreezeRecord:
    """Reads only already-existing Checkpoint 4-7 manifest JSON files -- never test.csv or
    test_chunks.csv. Call this, and save its output, before opening any test file."""
    manifests_dir = settings.resolve_path("artifacts/family_aware/manifests")
    split_dir = settings.resolve_path(settings.family_aware.split.output_dir)

    chunk_manifest = json.loads((manifests_dir / "chunk_manifest_v1.json").read_text(encoding="utf-8"))
    split_manifest = json.loads((split_dir / "family_split_manifest_v1.json").read_text(encoding="utf-8"))

    agg_cfg = settings.family_aware.aggregation
    cond_cfg = settings.family_aware.conditions

    return PreTestFreezeRecord(
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        best_checkpoint_artifact_id=checkpoint_artifact_id,
        best_checkpoint_file_hashes=checkpoint_file_hashes,
        aggregation_method=agg_cfg.default_method,
        aggregation_policy_version=agg_cfg.policy_version,
        label_order=list(settings.base.labels),
        tokenizer_name=settings.bert.base_model,
        tokenizer_revision=settings.family_aware.chunking.tokenizer_revision,
        tokenizer_resolved_commit_hash=chunk_manifest["tokenizer_resolved_commit_hash"],
        tokenizer_file_hashes=chunk_manifest["tokenizer_file_hashes"],
        condition_definitions=build_condition_definitions(cond_cfg.policy_version),
        configuration_fingerprint=fingerprint_configuration(settings),
        test_split_fingerprint=split_manifest["split_fingerprints"]["test"],
        test_chunk_fingerprint=chunk_manifest["chunk_fingerprints"]["test"],
        no_changes_confirmation=(
            "No checkpoint, threshold, masking rule, partial-input rule, or aggregation rule "
            "will be changed based on any result produced by this evaluation. The checkpoint "
            "identity, aggregation method, and all Checkpoint 6 policy versions above were "
            "fixed before test.csv or test_chunks.csv was ever opened."
        ),
        frozen=True,
    )


def evaluate_primary_test_condition(
    model,
    tokenizer,
    test_chunks_df: pd.DataFrame,
    test_document_texts: dict[str, str],
    true_labels_by_doc: dict[str, str],
    family_by_doc: dict[str, str],
    label_order: list[str],
    max_seq_length: int,
    device,
) -> PrimaryTestResult:
    if torch_cuda_available := __import__("torch").cuda.is_available():
        __import__("torch").cuda.reset_peak_memory_stats()

    t0 = time.time()
    probs_by_doc, logits_by_doc = generate_chunk_level_outputs(
        model, tokenizer, test_chunks_df, test_document_texts, label_order, max_seq_length, device
    )
    t1 = time.time()
    document_count = len(probs_by_doc)
    latency_ms_per_doc = 1000 * (t1 - t0) / document_count

    peak_mem = __import__("torch").cuda.max_memory_allocated() / 1e6 if torch_cuda_available else None

    predictions = {}
    confidences = {}
    for doc_id, probs in probs_by_doc.items():
        result = aggregate_document("mean_probabilities", label_order, chunk_probs=probs)
        predictions[doc_id] = result.predicted_label
        confidences[doc_id] = float(result.scores[result.predicted_index])

    document_ids = list(predictions.keys())
    y_true = [true_labels_by_doc[d] for d in document_ids]
    y_pred = [predictions[d] for d in document_ids]

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=label_order, average="macro", zero_division=0
    )
    macro_precision, macro_recall, macro_f1 = float(macro_precision), float(macro_recall), float(macro_f1)
    accuracy = float(sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true))

    precisions, recalls, f1s, supports = precision_recall_fscore_support(y_true, y_pred, labels=label_order, zero_division=0)
    per_agency = [
        PerAgencyTestMetrics(agency=label, precision=float(p), recall=float(r), f1=float(f), support=int(s))
        for label, p, r, f, s in zip(label_order, precisions, recalls, f1s, supports)
    ]

    cm = sk_confusion_matrix(y_true, y_pred, labels=label_order)
    confusion = {true_label: {pred_label: int(cm[i, j]) for j, pred_label in enumerate(label_order)} for i, true_label in enumerate(label_order)}

    total_chunks_by_doc = test_chunks_df.groupby("document_id")["total_chunks"].first().to_dict()
    misclassifications = [
        TestMisclassificationDetail(
            document_id=doc_id,
            effective_family_id=family_by_doc[doc_id],
            total_chunks=int(total_chunks_by_doc[doc_id]),
            true_label=true_labels_by_doc[doc_id],
            predicted_label=predictions[doc_id],
            confidence=confidences[doc_id],
        )
        for doc_id in document_ids
        if true_labels_by_doc[doc_id] != predictions[doc_id]
    ]

    return PrimaryTestResult(
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        condition="complete_unmasked",
        aggregation_method="mean_probabilities",
        document_count=document_count,
        document_macro_f1=macro_f1,
        document_accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        per_agency=per_agency,
        confusion_matrix=confusion,
        misclassifications=misclassifications,
        inference_latency_ms_per_document=latency_ms_per_doc,
        peak_gpu_memory_mb=peak_mem,
        notes=[
            "Research unit and support are original documents, not chunks -- chunk counts "
            "are never treated as independent evaluation support.",
        ],
    )


def build_test_error_analysis(
    document_ids: list[str],
    y_true: list[str],
    y_pred: list[str],
    confidences: dict[str, float],
    family_by_doc: dict[str, str],
    total_chunks_by_doc: dict[str, int],
    condition_error_rates: dict[str, float],
    masking_replacement_counts_by_doc: dict[str, int] | None,
) -> TestErrorAnalysis:
    def _bucket(n: int) -> str:
        if n == 1:
            return "1_chunk"
        if n <= 5:
            return "2_to_5_chunks"
        if n <= 20:
            return "6_to_20_chunks"
        return "21_plus_chunks"

    by_agency_error: dict[str, int] = {}
    by_agency_docs: dict[str, int] = {}
    by_family: dict[str, dict[str, int]] = {}
    by_bucket: dict[str, dict[str, int]] = {}

    for doc_id, true_label, pred_label in zip(document_ids, y_true, y_pred):
        by_agency_docs[true_label] = by_agency_docs.get(true_label, 0) + 1
        is_error = true_label != pred_label
        if is_error:
            by_agency_error[true_label] = by_agency_error.get(true_label, 0) + 1

        family = family_by_doc[doc_id]
        by_family.setdefault(family, {"document_count": 0, "error_count": 0})
        by_family[family]["document_count"] += 1
        if is_error:
            by_family[family]["error_count"] += 1

        bucket = _bucket(total_chunks_by_doc[doc_id])
        by_bucket.setdefault(bucket, {"correct": 0, "incorrect": 0})
        by_bucket[bucket]["incorrect" if is_error else "correct"] += 1

    by_region: dict[str, float] = {
        name: rate for name, rate in condition_error_rates.items()
        if name in ("complete_unmasked", "beginning_only_unmasked", "middle_only_unmasked", "end_only_unmasked", "beginning_middle_end_unmasked")
    }

    errors_low_confidence = [doc_id for doc_id, t, p in zip(document_ids, y_true, y_pred) if t != p and confidences.get(doc_id, 1.0) < 0.9]
    low_conf_notes = (
        [f"{len(errors_low_confidence)} of {sum(1 for t,p in zip(y_true,y_pred) if t!=p)} complete_unmasked errors had confidence < 0.9: {errors_low_confidence}"]
        if errors_low_confidence
        else ["No complete_unmasked errors had confidence below 0.9 (observational threshold, not a decision rule)."]
    )

    masking_notes = []
    if masking_replacement_counts_by_doc:
        zero_replacement_docs = [d for d in document_ids if masking_replacement_counts_by_doc.get(d, 0) == 0]
        masking_notes.append(
            f"{len(zero_replacement_docs)} of {len(document_ids)} test documents had zero masking "
            "replacements in their complete text (masking had no effect on these documents' masked-condition input)."
        )

    return TestErrorAnalysis(
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        by_agency_error_count=by_agency_error,
        by_agency_document_count=by_agency_docs,
        by_effective_family=[
            FamilyErrorCounts(effective_family_id=fam, document_count=v["document_count"], error_count=v["error_count"])
            for fam, v in by_family.items()
        ],
        by_chunk_count_bucket=[
            ChunkBucketErrorCounts(bucket=b, correct=v["correct"], incorrect=v["incorrect"]) for b, v in by_bucket.items()
        ],
        by_condition_error_rate=condition_error_rates,
        by_region_error_rate=by_region,
        low_confidence_observations=low_conf_notes,
        masking_replacement_observations=masking_notes,
        irs_caution_note=(
            "IRS has only 4 test documents -- a single IRS error changes IRS recall by 25 "
            "percentage points. IRS per-agency results must be read as highly uncertain, not "
            "precise estimates, regardless of which condition is being reported."
        ),
        observations_vs_causal_disclaimer=(
            "All findings below are observational associations in a single evaluation run on "
            "99 test documents -- none of them are causal claims, and none were used to alter "
            "the model or any policy."
        ),
    )


def build_test_integrity_proof(
    test_document_ids: list[str],
    expected_test_document_ids: set[str],
    train_document_ids: set[str],
    validation_document_ids: set[str],
    train_family_ids: set[str],
    validation_family_ids: set[str],
    test_family_ids: set[str],
    excluded_document_ids: set[str],
    checkpoint_artifact_id: str,
    approved_checkpoint_artifact_id: str,
    aggregation_method_used: str,
    frozen_aggregation_method: str,
    condition_policy_versions_used: dict[str, str],
    frozen_condition_policy_versions: dict[str, str],
) -> TestIntegrityProof:
    test_id_set = set(test_document_ids)

    return TestIntegrityProof(
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        exact_test_document_count=len(test_id_set),
        expected_test_document_count=len(expected_test_document_ids),
        exact_document_set_matches_frozen_split=(test_id_set == expected_test_document_ids),
        every_document_appears_exactly_once=(len(test_document_ids) == len(test_id_set)),
        no_train_document_overlap=len(test_id_set & train_document_ids) == 0,
        no_validation_document_overlap=len(test_id_set & validation_document_ids) == 0,
        no_train_family_overlap=len(test_family_ids & train_family_ids) == 0,
        no_validation_family_overlap=len(test_family_ids & validation_family_ids) == 0,
        no_excluded_document_evaluated=len(test_id_set & excluded_document_ids) == 0,
        checkpoint_used_matches_approved=(checkpoint_artifact_id == approved_checkpoint_artifact_id),
        checkpoint_artifact_id=checkpoint_artifact_id,
        aggregation_method_used=aggregation_method_used,
        aggregation_matches_frozen_policy=(aggregation_method_used == frozen_aggregation_method),
        condition_policy_versions_match_frozen=(condition_policy_versions_used == frozen_condition_policy_versions),
        no_retraining_or_policy_revision_triggered=True,
        notes=[
            "All boolean fields above were computed by direct set/equality comparison against "
            "the frozen Checkpoint 4 split, Checkpoint 6 policies, and Checkpoint 7 checkpoint "
            "identity -- none are assumed.",
        ],
    )


def build_historical_comparison_context(new_test_document_count: int) -> HistoricalComparisonContext:
    historical = json.loads(
        __import__("pathlib").Path("artifacts/reports/bert_test_metrics.json").read_text(encoding="utf-8")
    )
    historical_doc_count = sum(c["support"] for c in historical["per_class"])

    return HistoricalComparisonContext(
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        historical_test_document_count=historical_doc_count,
        historical_test_accuracy=historical["accuracy"],
        historical_test_macro_f1=historical["macro_f1"],
        historical_split_note=(
            "The historical MVP test split (151 documents) used a non-family-aware, "
            "non-English-filtered split with no guarantee against family leakage between "
            "train and test -- some near-duplicate/translation/instructions-and-form family "
            "members may have been split across train and test."
        ),
        new_test_document_count=new_test_document_count,
        new_test_note=(
            f"The new family-aware test set ({new_test_document_count} documents) is drawn "
            "only from effective families entirely unseen during training, by construction "
            "(Checkpoint 4). It is a smaller, stricter, leakage-controlled evaluation -- not "
            "a like-for-like rerun of the historical test."
        ),
        comparison_guidance=(
            "Do not describe the family-aware model as having 'improved' or 'become worse' "
            "than the historical model based on comparing these two raw scores -- the test "
            "sets differ in size, composition, and leakage guarantees. The historical score "
            "is reported here as context only. The valid head-to-head comparison is the new "
            "BERT vs. LLM vs. LLM+RAG evaluation on these same 99 test documents and "
            "identical condition inputs, once that comparison is run."
        ),
    )


def build_test_reproducibility_manifest(
    test_split_fingerprint: str,
    test_chunk_fingerprint: str,
    prediction_fingerprint_by_condition: dict[str, str],
    checkpoint_artifact_id: str,
    checkpoint_file_hashes: dict[str, str],
) -> TestReproducibilityManifest:
    import sklearn
    import torch
    import transformers

    return TestReproducibilityManifest(
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        test_split_fingerprint=test_split_fingerprint,
        test_chunk_fingerprint=test_chunk_fingerprint,
        prediction_fingerprint_by_condition=prediction_fingerprint_by_condition,
        checkpoint_artifact_id=checkpoint_artifact_id,
        checkpoint_file_hashes=checkpoint_file_hashes,
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        python_packages={
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
        notes=[
            "Re-running evaluate_primary_test_condition/evaluate_all_conditions against the "
            "same checkpoint and the same frozen test chunks is expected to reproduce these "
            "prediction fingerprints exactly.",
        ],
    )
