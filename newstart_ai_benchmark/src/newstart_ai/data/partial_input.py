"""Deterministic partial-document conditions (Version 6, Checkpoint 6).

Operates only on each document's already-frozen, ordered chunk sequence from Checkpoint 5
(`chunk_index` 0..total_chunks-1) -- never on raw text or test data. Every selection is
reproducible from (document_id, condition, the document's chunk sequence) alone.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pandas as pd

from newstart_ai.schemas.checkpoint6 import PartialInputManifest

CONDITIONS = ("beginning_only", "middle_only", "end_only", "beginning_middle_end")


def _middle_index(total_chunks: int) -> int:
    """Deterministic middle-chunk index. For total_chunks==1 this is 0 (coincides with
    beginning/end); for total_chunks==2 this is also 1 (coincides with end) -- both are
    documented, expected fallback cases, never silent duplication."""
    return total_chunks // 2


def select_partial_chunks(document_id: str, total_chunks: int, condition: str, policy_version: str) -> dict:
    """Returns the selection record for one (document, condition) pair.

    `selected_chunk_indices` is always deduplicated, order-preserving (beginning, then
    middle, then end for the combined condition) -- a chunk is never selected twice unless
    explicitly justified via `fallback_reason`.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown partial-input condition: {condition!r}")
    if total_chunks <= 0:
        raise ValueError(f"document_id={document_id!r} has total_chunks={total_chunks}, cannot select a region.")

    beginning_index = 0
    end_index = total_chunks - 1
    middle_index = _middle_index(total_chunks)

    if condition == "beginning_only":
        requested_regions = ["beginning"]
        requested_indices = [beginning_index]
    elif condition == "middle_only":
        requested_regions = ["middle"]
        requested_indices = [middle_index]
    elif condition == "end_only":
        requested_regions = ["end"]
        requested_indices = [end_index]
    else:  # beginning_middle_end
        requested_regions = ["beginning", "middle", "end"]
        requested_indices = [beginning_index, middle_index, end_index]

    selected_indices = list(dict.fromkeys(requested_indices))  # dedupe, preserve first occurrence order

    fallback_reason = None
    if len(selected_indices) < len(requested_indices):
        if total_chunks == 1:
            fallback_reason = (
                f"document has only 1 chunk; all requested regions ({', '.join(requested_regions)}) "
                "collapse to that single chunk"
            )
        else:
            fallback_reason = (
                f"document has only {total_chunks} chunks; requested regions "
                f"({', '.join(requested_regions)}) collapsed from {len(requested_indices)} to "
                f"{len(selected_indices)} distinct chunk(s): {selected_indices}"
            )

    selection_hash = hashlib.sha256(
        f"{policy_version}|{document_id}|{condition}|{','.join(map(str, selected_indices))}".encode("utf-8")
    ).hexdigest()

    return {
        "document_id": str(document_id),
        "condition": condition,
        "requested_regions": requested_regions,
        "selected_chunk_indices": selected_indices,
        "total_chunks": total_chunks,
        "fallback_reason": fallback_reason,
        "selection_hash": selection_hash,
        "policy_version": policy_version,
    }


def build_partial_input_selections(chunks_df: pd.DataFrame, split_name: str, settings) -> pd.DataFrame:
    """Builds one selection row per (document_id, condition) for every document in
    `chunks_df` (a single split's chunk DataFrame, e.g. train_chunks or validation_chunks)."""
    policy_version = settings.family_aware.partial_input.policy_version
    doc_totals = chunks_df.groupby("document_id")["total_chunks"].first()

    rows = []
    for document_id, total_chunks in doc_totals.items():
        for condition in CONDITIONS:
            row = select_partial_chunks(document_id, int(total_chunks), condition, policy_version)
            row["split"] = split_name
            rows.append(row)
    return pd.DataFrame(rows)


def resolve_selection_text(document_chunks: pd.DataFrame, selected_chunk_indices: list[int], text_column: str = "chunk_text") -> str:
    """Joins the selected chunks' text (in the given order) with a deterministic separator.
    `document_chunks` must be one document's chunk rows, indexed/filterable by chunk_index."""
    by_index = document_chunks.set_index("chunk_index")[text_column]
    return "\n\n".join(str(by_index.loc[i]) for i in selected_chunk_indices)


def build_partial_input_manifest(selections_df: pd.DataFrame, settings) -> PartialInputManifest:
    cfg = settings.family_aware.partial_input

    fallback_counts = (
        selections_df.assign(has_fallback=selections_df["fallback_reason"].notna())
        .groupby("condition")["has_fallback"]
        .sum()
        .astype(int)
        .to_dict()
    )

    # Prove no chunk was selected more than once per (document, condition) row.
    no_unjustified_duplicates = True
    for _, row in selections_df.iterrows():
        indices = row["selected_chunk_indices"]
        if len(indices) != len(set(indices)):
            no_unjustified_duplicates = False
            break

    return PartialInputManifest(
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        policy_version=cfg.policy_version,
        chunks_per_region=cfg.chunks_per_region,
        total_documents=int(selections_df["document_id"].nunique()),
        fallback_document_counts_by_condition={k: int(v) for k, v in fallback_counts.items()},
        no_unjustified_duplicate_selection=no_unjustified_duplicates,
        notes=[
            "middle_index = total_chunks // 2 by construction -- for 2-chunk documents this "
            "coincides with the end chunk (documented fallback, not an error).",
            "beginning_middle_end deduplicates requested indices, preserving "
            "beginning-then-middle-then-end order; a chunk is only ever listed once in "
            "selected_chunk_indices per row.",
        ],
    )
