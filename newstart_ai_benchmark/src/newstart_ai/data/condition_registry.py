"""Shared evaluation-condition registry (Version 6, Checkpoint 6).

Guarantees that BERT, LLM, and LLM+RAG evaluations (Checkpoint 7+) all consume the exact
same text for a given (document_id, condition) pair -- the registry's only job is to freeze
that mapping once, deterministically, so no method can silently receive different text for
what is supposed to be the same evaluation condition.

"Complete" conditions carry the full document text (masked or unmasked); each method's
runner decides how to consume it (BERT chunks it via the frozen Checkpoint 5 policy and
aggregates; LLM/LLM+RAG send it directly). "Partial" conditions carry the already-selected,
already-(un)masked chunk text for that condition (Checkpoint 6's partial_input selections),
concatenated in beginning/middle/end order -- identical substrings for every method.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pandas as pd

from newstart_ai.schemas.checkpoint6 import ConditionDefinition, ConditionRegistryManifest

_CONDITION_SPECS: dict[str, tuple[str, bool]] = {
    "complete_unmasked": ("complete", False),
    "beginning_only_unmasked": ("beginning", False),
    "middle_only_unmasked": ("middle", False),
    "end_only_unmasked": ("end", False),
    "beginning_middle_end_unmasked": ("beginning_middle_end", False),
    "complete_masked": ("complete", True),
    "beginning_only_masked": ("beginning", True),
    "middle_only_masked": ("middle", True),
    "end_only_masked": ("end", True),
    "beginning_middle_end_masked": ("beginning_middle_end", True),
}

_PARTIAL_TO_SELECTION_CONDITION = {
    "beginning": "beginning_only",
    "middle": "middle_only",
    "end": "end_only",
    "beginning_middle_end": "beginning_middle_end",
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_condition_registry(
    split_df: pd.DataFrame,
    masked_documents_df: pd.DataFrame,
    chunks_df: pd.DataFrame,
    masked_chunks_df: pd.DataFrame,
    selections_df: pd.DataFrame,
    split_name: str,
    settings,
) -> pd.DataFrame:
    """Builds one row per (document_id, condition) for every document in this split."""
    policy_version = settings.family_aware.conditions.policy_version

    complete_unmasked_text = split_df.set_index(split_df["document_id"].astype(str))["text"]
    complete_masked_text = masked_documents_df.set_index("document_id")["masked_text"]
    effective_agency = split_df.set_index(split_df["document_id"].astype(str))["effective_agency"]

    unmasked_chunk_text_by_chunk_id = chunks_df.set_index("chunk_id")["chunk_text"]
    masked_chunk_text_by_chunk_id = masked_chunks_df.set_index("chunk_id")["masked_chunk_text"]
    chunks_by_document = {doc_id: group for doc_id, group in chunks_df.groupby("document_id")}

    selections_by_doc_condition = {
        (row.document_id, row.condition): row for row in selections_df.itertuples(index=False)
    }

    rows = []
    for document_id in complete_unmasked_text.index:
        agency = effective_agency.loc[document_id]
        for condition_name, (region, masked) in _CONDITION_SPECS.items():
            if region == "complete":
                text = complete_masked_text.loc[document_id] if masked else complete_unmasked_text.loc[document_id]
                source_chunk_ids: list[str] = []
                fallback_reason = None
            else:
                selection_condition = _PARTIAL_TO_SELECTION_CONDITION[region]
                selection = selections_by_doc_condition[(document_id, selection_condition)]
                doc_chunks = chunks_by_document[document_id]
                text_lookup = masked_chunk_text_by_chunk_id if masked else unmasked_chunk_text_by_chunk_id
                by_index = doc_chunks.set_index("chunk_index")["chunk_id"]
                source_chunk_ids = [str(by_index.loc[i]) for i in selection.selected_chunk_indices]
                text = "\n\n".join(str(text_lookup.loc[cid]) for cid in source_chunk_ids)
                fallback_reason = selection.fallback_reason

            rows.append(
                {
                    "document_id": document_id,
                    "split": split_name,
                    "effective_agency": agency,
                    "condition": condition_name,
                    "region": region,
                    "masked": masked,
                    "text": text,
                    "text_fingerprint": _sha256(text),
                    "source_chunk_ids": ",".join(source_chunk_ids),
                    "fallback_reason": fallback_reason,
                    "policy_version": policy_version,
                }
            )
    return pd.DataFrame(rows)


def build_condition_definitions(policy_version: str) -> list[ConditionDefinition]:
    definitions_text = {
        "complete": "The document's full text (all content, no chunk selection).",
        "beginning": "The single chunk at chunk_index=0.",
        "middle": "The single chunk at chunk_index = total_chunks // 2.",
        "end": "The single chunk at chunk_index = total_chunks - 1.",
        "beginning_middle_end": "The distinct, deduplicated union of the beginning, middle, "
        "and end chunks, concatenated in that order.",
    }
    return [
        ConditionDefinition(
            name=name,
            definition=(f"[MASKED] {definitions_text[region]}" if masked else definitions_text[region]),
            masked=masked,
            region=region,
            policy_version=policy_version,
        )
        for name, (region, masked) in _CONDITION_SPECS.items()
    ]


def build_condition_registry_manifest(registry_df: pd.DataFrame, settings) -> ConditionRegistryManifest:
    cfg = settings.family_aware.conditions
    definitions = build_condition_definitions(cfg.policy_version)

    per_condition_counts = registry_df.groupby("condition").size().to_dict()
    ordered = registry_df[["document_id", "split", "condition", "text_fingerprint"]].astype(str).sort_values(
        ["document_id", "split", "condition"]
    )
    payload = "\n".join("|".join(row) for row in ordered.itertuples(index=False))
    registry_fingerprint = _sha256(payload)

    return ConditionRegistryManifest(
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        policy_version=cfg.policy_version,
        conditions=definitions,
        total_documents=int(registry_df["document_id"].nunique()),
        total_rows=int(len(registry_df)),
        per_condition_row_counts={str(k): int(v) for k, v in per_condition_counts.items()},
        registry_fingerprint=registry_fingerprint,
        notes=[
            "Each (document_id, condition) pair maps to exactly one frozen text string and "
            "fingerprint here -- any evaluation method consuming this registry for the same "
            "document_id and condition receives byte-identical text.",
            "'Complete' conditions carry the full document text; each method's runner "
            "decides how to consume it (BERT chunks + aggregates via the frozen Checkpoint "
            "5/6 policies; LLM/LLM+RAG send it directly).",
        ],
    )
