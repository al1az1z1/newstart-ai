"""Read-only summaries of the real, frozen research artifacts.

Every function here opens an already-saved file at its real (unpatched) location and
returns a small dict describing what's there -- row/document counts, key config values,
and (where a comparable frozen metric exists) a metric recomputed straight from raw
predictions so a caller can see it agrees with what was reported. Nothing in this module
ever writes a file, trains a model, or calls an external API. This is the function set every
stage module's default (no-flag) CLI mode calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from newstart_ai_mvp.config import Settings


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def describe_language_audit(settings: Settings) -> dict[str, Any]:
    reports_dir = settings.resolve_path("artifacts/family_aware/reports")
    audit = pd.read_csv(reports_dir / "language_audit_v1.csv")
    manifest = _read_json(reports_dir / "language_audit_manifest_v1.json")
    return {
        "artifact": "language_audit_v1.csv",
        "document_count": len(audit),
        "language_status_counts": audit["status"].value_counts().to_dict(),
        "policy_version": manifest.get("policy_version"),
        "created_at": manifest.get("created_at"),
    }


def describe_family_audit(settings: Settings) -> dict[str, Any]:
    reports_dir = settings.resolve_path("artifacts/family_aware/reports")
    manifests_dir = settings.resolve_path("artifacts/family_aware/manifests")
    audit = pd.read_csv(reports_dir / "family_audit_v1.csv")
    manifest = _read_json(manifests_dir / "family_audit_manifest_v1.json")
    return {
        "artifact": "family_audit_v1.csv",
        "document_count": len(audit),
        "final_modeling_eligibility_counts": audit["final_modeling_eligibility"].value_counts().to_dict(),
        "effective_agency_counts": audit["effective_agency"].value_counts().to_dict(),
        "created_at": manifest.get("created_at"),
    }


def describe_split(settings: Settings) -> dict[str, Any]:
    split_dir = settings.resolve_path(settings.family_aware.split.output_dir)
    manifest = _read_json(split_dir / "family_split_manifest_v1.json")
    counts = {}
    for split_name in ("train", "validation", "test"):
        df = pd.read_csv(split_dir / f"{split_name}.csv")
        counts[split_name] = {
            "documents": len(df),
            "families": df["effective_family_id"].nunique(),
            "by_agency": df["effective_agency"].value_counts().to_dict(),
        }
    return {
        "artifact": "family_aware_splits/{train,validation,test}.csv",
        "counts": counts,
        "created_at": manifest.get("created_at"),
    }


def describe_chunks(settings: Settings) -> dict[str, Any]:
    chunks_dir = settings.resolve_path(settings.family_aware.chunking.output_dir)
    manifest = _read_json(settings.resolve_path("artifacts/family_aware/manifests") / "chunk_manifest_v1.json")
    counts = {}
    for split_name in ("train", "validation", "test"):
        df = pd.read_csv(chunks_dir / f"{split_name}_chunks.csv")
        counts[split_name] = {"chunks": len(df), "documents": df["document_id"].nunique()}
    return {
        "artifact": "family_aware_chunks/{split}_chunks.csv",
        "counts": counts,
        "chunking_policy_version": manifest.get("chunking_policy_version"),
        "created_at": manifest.get("created_at"),
    }


def describe_masked(settings: Settings) -> dict[str, Any]:
    masked_dir = settings.resolve_path(settings.family_aware.masking.output_dir)
    manifest = _read_json(settings.resolve_path("artifacts/family_aware/manifests") / "masking_policy_v1.json")
    counts = {}
    for split_name in ("train", "validation", "test"):
        docs = pd.read_csv(masked_dir / f"{split_name}_masked_documents.csv")
        counts[split_name] = {
            "documents": len(docs),
            "documents_with_replacements": int((docs["total_replacements"] > 0).sum()),
        }
    return {
        "artifact": "family_aware_masked/{split}_masked_documents.csv",
        "counts": counts,
        "policy_version": manifest.get("policy_version"),
    }


def describe_conditions(settings: Settings) -> dict[str, Any]:
    conditions_dir = settings.resolve_path(settings.family_aware.conditions.output_dir)
    registry_test = pd.read_csv(conditions_dir / "condition_registry_test.csv")
    registry_train_val = pd.read_csv(conditions_dir / "condition_registry_train_validation.csv")
    return {
        "artifact": "family_aware_conditions/condition_registry_{train_validation,test}.csv",
        "test_rows": len(registry_test),
        "test_documents": registry_test["document_id"].nunique(),
        "test_conditions": sorted(registry_test["condition"].unique().tolist()),
        "train_validation_rows": len(registry_train_val),
    }


def describe_bert_checkpoint(settings: Settings) -> dict[str, Any]:
    from newstart_ai_mvp.bert_pipeline import (
        latest_ready_family_aware_artifact_id,
        load_family_aware_artifact_metadata,
    )

    artifact_id = latest_ready_family_aware_artifact_id(settings)
    if artifact_id is None:
        return {"artifact": "family_aware BERT checkpoint", "status": "none found"}
    metadata = load_family_aware_artifact_metadata(settings, artifact_id)
    return {
        "artifact": f"family_aware/models/{artifact_id}/metadata.json",
        "artifact_id": artifact_id,
        "base_model": metadata["base_model"],
        "training_config": metadata["training_config"],
        "history_epochs": [h["epoch"] for h in metadata["history"]],
        "best_epoch": metadata["best_epoch"],
        "best_validation_document_macro_f1": metadata["best_validation_document_macro_f1"],
        "status": metadata["status"],
        "created_at": metadata["created_at"],
    }


def describe_bert_test_results(settings: Settings) -> dict[str, Any]:
    reports_dir = settings.resolve_path("artifacts/family_aware/reports")
    manifests_dir = settings.resolve_path("artifacts/family_aware/manifests")
    predictions = pd.read_csv(reports_dir / "checkpoint8_test_predictions.csv")
    primary_result = _read_json(manifests_dir / "checkpoint8_primary_test_result_v1.json")

    primary = predictions[predictions["condition"] == "complete_unmasked"]
    label_order = sorted(primary["true_label"].unique().tolist())
    recomputed_macro_f1 = float(
        f1_score(primary["true_label"], primary["predicted_label"], average="macro", labels=label_order)
    )
    recomputed_accuracy = float(accuracy_score(primary["true_label"], primary["predicted_label"]))

    return {
        "artifact": "checkpoint8_test_predictions.csv (complete_unmasked slice, n=%d)" % len(primary),
        "recomputed_macro_f1": recomputed_macro_f1,
        "reported_macro_f1": primary_result.get("document_macro_f1"),
        "recomputed_accuracy": recomputed_accuracy,
        "reported_accuracy": primary_result.get("document_accuracy"),
        "agrees_with_report": abs(recomputed_macro_f1 - primary_result.get("document_macro_f1", -1)) < 1e-9,
    }


def describe_rag_index(settings: Settings) -> dict[str, Any]:
    import chromadb

    from newstart_ai_mvp.rag_pipeline import MASKED_COLLECTION_NAME, UNMASKED_COLLECTION_NAME

    result: dict[str, Any] = {"artifact": "family_aware/vector_stores/routing_index_{masked,unmasked}"}
    for masked, persist_dir_key, collection_name in (
        (False, "persist_dir_unmasked", UNMASKED_COLLECTION_NAME),
        (True, "persist_dir_masked", MASKED_COLLECTION_NAME),
    ):
        persist_dir = settings.resolve_path(getattr(settings.family_aware.rag, persist_dir_key))
        client = chromadb.PersistentClient(path=str(persist_dir))
        try:
            collection = client.get_collection(collection_name)
            count = collection.count()
        except Exception as exc:  # collection missing/unreadable -- report, don't crash
            count = f"unavailable ({exc})"
        result["masked" if masked else "unmasked"] = {"collection": collection_name, "chunk_count": count}
    return result


def _describe_prediction_file(settings: Settings, filename: str, method: str) -> dict[str, Any]:
    reports_dir = settings.resolve_path("artifacts/family_aware/reports")
    manifests_dir = settings.resolve_path("artifacts/family_aware/manifests")
    predictions = pd.read_json(reports_dir / filename, lines=True)
    metrics = _read_json(manifests_dir / "checkpoint10_method_condition_metrics_v1.json")

    primary = predictions[predictions["condition"] == "complete_unmasked"]
    label_order = sorted(primary["true_label"].unique().tolist())
    recomputed_macro_f1 = float(
        f1_score(primary["true_label"], primary["predicted_label"], average="macro", labels=label_order)
    )
    reported = next(
        (m for m in metrics if m["method"] == method and m["condition"] == "complete_unmasked"), None
    )
    reported_macro_f1 = reported["document_macro_f1"] if reported else None

    return {
        "artifact": filename,
        "total_cases": len(predictions),
        "status_counts": predictions["status"].value_counts().to_dict(),
        "recomputed_primary_macro_f1": recomputed_macro_f1,
        "reported_primary_macro_f1": reported_macro_f1,
        "agrees_with_report": (
            reported_macro_f1 is not None and abs(recomputed_macro_f1 - reported_macro_f1) < 1e-9
        ),
    }


def describe_llm_predictions(settings: Settings) -> dict[str, Any]:
    return _describe_prediction_file(settings, "checkpoint10_llm_predictions.jsonl", "llm")


def describe_rag_predictions(settings: Settings) -> dict[str, Any]:
    return _describe_prediction_file(settings, "checkpoint10_llm_rag_predictions.jsonl", "llm_rag")


def print_report(title: str, report: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(report, indent=2, default=str))
