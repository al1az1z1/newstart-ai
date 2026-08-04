"""Tests for Version 6 Checkpoint 6 deterministic partial-input selection."""

from __future__ import annotations

import pandas as pd
import pytest

from newstart_ai.config import load_settings
from newstart_ai.data.partial_input import (
    build_partial_input_manifest,
    build_partial_input_selections,
    resolve_selection_text,
    select_partial_chunks,
)


@pytest.fixture(scope="module")
def settings():
    return load_settings()


def _chunks_df(document_id: str, total_chunks: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "document_id": [document_id] * total_chunks,
            "chunk_id": [f"{document_id}_{i}" for i in range(total_chunks)],
            "chunk_index": list(range(total_chunks)),
            "total_chunks": [total_chunks] * total_chunks,
            "chunk_text": [f"chunk-{i}-text" for i in range(total_chunks)],
        }
    )


def test_single_chunk_document_all_regions_collapse_with_fallback():
    result = select_partial_chunks("d1", 1, "beginning_middle_end", "v1")
    assert result["selected_chunk_indices"] == [0]
    assert result["fallback_reason"] is not None
    assert "1 chunk" in result["fallback_reason"]


def test_beginning_only_selects_first_chunk():
    result = select_partial_chunks("d1", 10, "beginning_only", "v1")
    assert result["selected_chunk_indices"] == [0]
    assert result["fallback_reason"] is None


def test_end_only_selects_last_chunk():
    result = select_partial_chunks("d1", 10, "end_only", "v1")
    assert result["selected_chunk_indices"] == [9]
    assert result["fallback_reason"] is None


def test_middle_only_selects_floor_division_index():
    result = select_partial_chunks("d1", 7, "middle_only", "v1")
    assert result["selected_chunk_indices"] == [3]


def test_beginning_middle_end_distinct_for_long_document():
    result = select_partial_chunks("d1", 100, "beginning_middle_end", "v1")
    assert result["selected_chunk_indices"] == [0, 50, 99]
    assert result["fallback_reason"] is None
    assert len(set(result["selected_chunk_indices"])) == 3


def test_two_chunk_document_middle_collapses_with_documented_fallback():
    result = select_partial_chunks("d1", 2, "beginning_middle_end", "v1")
    # middle_index = 2 // 2 = 1, colliding with end (index 1) -- not silently duplicated.
    assert result["selected_chunk_indices"] == [0, 1]
    assert len(result["selected_chunk_indices"]) == len(set(result["selected_chunk_indices"]))
    assert result["fallback_reason"] is not None


def test_zero_chunks_raises():
    with pytest.raises(ValueError):
        select_partial_chunks("d1", 0, "beginning_only", "v1")


def test_unknown_condition_raises():
    with pytest.raises(ValueError):
        select_partial_chunks("d1", 10, "not_a_real_condition", "v1")


def test_selection_hash_is_deterministic_and_position_derived():
    a = select_partial_chunks("d1", 50, "end_only", "v1")
    b = select_partial_chunks("d1", 50, "end_only", "v1")
    assert a["selection_hash"] == b["selection_hash"]

    different_doc = select_partial_chunks("d2", 50, "end_only", "v1")
    assert different_doc["selection_hash"] != a["selection_hash"]

    different_policy = select_partial_chunks("d1", 50, "end_only", "v2")
    assert different_policy["selection_hash"] != a["selection_hash"]


def test_resolve_selection_text_joins_in_selection_order():
    chunks = _chunks_df("d1", 5)
    text = resolve_selection_text(chunks, [0, 2, 4])
    assert text == "chunk-0-text\n\nchunk-2-text\n\nchunk-4-text"


def test_build_partial_input_selections_covers_every_document_and_condition(settings):
    chunks = pd.concat([_chunks_df("d1", 1), _chunks_df("d2", 3), _chunks_df("d3", 50)], ignore_index=True)
    selections = build_partial_input_selections(chunks, "train", settings)
    assert set(selections["document_id"].unique()) == {"d1", "d2", "d3"}
    assert set(selections["condition"].unique()) == {"beginning_only", "middle_only", "end_only", "beginning_middle_end"}
    assert len(selections) == 3 * 4  # 3 documents x 4 conditions


def test_partial_input_manifest_reports_fallbacks_and_no_duplicates(settings):
    chunks = pd.concat([_chunks_df("d1", 1), _chunks_df("d2", 2), _chunks_df("d3", 50)], ignore_index=True)
    selections = build_partial_input_selections(chunks, "train", settings)
    manifest = build_partial_input_manifest(selections, settings)
    assert manifest.total_documents == 3
    assert manifest.no_unjustified_duplicate_selection is True
    # d1 (1 chunk) and d2 (2 chunks) both require a beginning_middle_end fallback.
    assert manifest.fallback_document_counts_by_condition["beginning_middle_end"] == 2
    assert manifest.fallback_document_counts_by_condition["beginning_only"] == 0
