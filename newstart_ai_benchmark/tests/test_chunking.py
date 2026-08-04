"""Tests for Version 6 tokenizer-aware chunking with provenance (Checkpoint 5).

Most tests call `build_chunk_rows_for_document`/`compute_chunk_token_ranges` directly with
short synthetic texts against the real configured bert-base-uncased tokenizer (loaded once,
module-scoped, no network calls needed since it is already cached locally from Phase 1
training) -- fast and deterministic. The final test reads the real generated chunk CSVs
under data/family_aware_chunks/ to prove real-data completeness, per Checkpoint 5's explicit
requirement.
"""

from __future__ import annotations

import pandas as pd
import pytest

from newstart_ai.config import load_settings
from newstart_ai.data.chunking import (
    assert_chunk_indices_contiguous_and_unique,
    assert_every_chunk_inherits_parent_split,
    assert_no_cross_split_leakage,
    assert_no_duplicate_chunk_ids,
    build_chunk_rows_for_document,
    compute_chunk_token_ranges,
    get_bert_tokenizer,
)


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def tokenizer(settings):
    return get_bert_tokenizer(settings)


@pytest.fixture(scope="module")
def cfg(settings):
    return settings.family_aware.chunking


def _build(tokenizer, cfg, **overrides):
    defaults = dict(
        document_id="doc",
        text="placeholder",
        effective_family_id="FAM:X",
        agency="USCIS",
        effective_agency="USCIS",
        split="train",
        tokenizer=tokenizer,
        tokenizer_name="bert-base-uncased",
        cfg=cfg,
    )
    defaults.update(overrides)
    return build_chunk_rows_for_document(**defaults)


def test_short_document_produces_one_chunk(tokenizer, cfg):
    rows = _build(tokenizer, cfg, document_id="doc1", text="A short English sentence about USCIS forms.")
    assert len(rows) == 1
    assert rows[0]["chunk_index"] == 0
    assert rows[0]["total_chunks"] == 1
    assert rows[0]["content_token_count"] > 0


def test_long_document_produces_multiple_overlapping_chunks(tokenizer, cfg):
    text = " ".join(f"word{i}" for i in range(3000))
    rows = _build(tokenizer, cfg, document_id="doc2", text=text, split="validation")
    assert len(rows) > 1
    for i, row in enumerate(rows):
        assert row["chunk_index"] == i
        assert row["total_chunks"] == len(rows)
    # Consecutive windows must share tokens (overlap), not be disjoint.
    assert rows[1]["token_start"] < rows[0]["token_end"]


def test_exact_boundary_produces_exactly_one_chunk(cfg):
    window = cfg.max_seq_length - cfg.num_special_tokens
    ranges = compute_chunk_token_ranges(window, window, window - cfg.chunk_overlap_tokens)
    assert ranges == [(0, window)]


def test_one_token_above_limit_produces_two_chunks_with_tail_preserved(cfg):
    window = cfg.max_seq_length - cfg.num_special_tokens
    step = window - cfg.chunk_overlap_tokens
    ranges = compute_chunk_token_ranges(window + 1, window, step)
    assert len(ranges) == 2
    assert ranges[-1][1] == window + 1  # tail token retained, not truncated


def test_empty_text_produces_zero_ranges(cfg):
    window = cfg.max_seq_length - cfg.num_special_tokens
    assert compute_chunk_token_ranges(0, window, window - cfg.chunk_overlap_tokens) == []


def test_extremely_long_document_final_range_reaches_the_end(cfg):
    window = cfg.max_seq_length - cfg.num_special_tokens
    step = window - cfg.chunk_overlap_tokens
    total = 50_000
    ranges = compute_chunk_token_ranges(total, window, step)
    assert ranges[-1][1] == total
    assert ranges[-1][1] - ranges[-1][0] == window


@pytest.mark.parametrize("total", [0, 1, 2, 30, 50_000])
def test_no_empty_or_negative_length_ranges_ever_produced(cfg, total):
    window = cfg.max_seq_length - cfg.num_special_tokens
    step = window - cfg.chunk_overlap_tokens
    for start, end in compute_chunk_token_ranges(total, window, step):
        assert end > start


def test_deterministic_chunk_ids_and_hashes_across_independent_calls(tokenizer, cfg):
    text = "Repeated deterministic content. " * 200
    rows_a = _build(tokenizer, cfg, document_id="doc3", text=text, agency="SSA", effective_agency="SSA", split="test")
    rows_b = _build(tokenizer, cfg, document_id="doc3", text=text, agency="SSA", effective_agency="SSA", split="test")
    assert rows_a == rows_b
    assert [r["chunk_id"] for r in rows_a] == [r["chunk_id"] for r in rows_b]


def test_no_empty_chunks_end_to_end(tokenizer, cfg):
    text = "Another moderately long piece of text used to verify chunk validity. " * 50
    rows = _build(tokenizer, cfg, document_id="doc_valid", text=text)
    df = pd.DataFrame(rows)
    assert (df["content_token_count"] > 0).all()
    assert (df["encoded_sequence_length"] <= cfg.max_seq_length).all()


def test_parent_split_and_family_inheritance(tokenizer, cfg):
    rows = _build(
        tokenizer, cfg,
        document_id="doc5", text="Some inherited-metadata test document.",
        effective_family_id="FAM:5", agency="DMV", effective_agency="IRS", split="validation",
    )
    assert rows[0]["effective_family_id"] == "FAM:5"
    assert rows[0]["agency"] == "DMV"
    assert rows[0]["effective_agency"] == "IRS"
    assert rows[0]["split"] == "validation"


def test_contiguous_chunk_indices_for_a_multi_chunk_document(tokenizer, cfg):
    text = "Long text needing several chunks. " * 400
    rows = _build(tokenizer, cfg, document_id="doc6", text=text, split="test")
    df = pd.DataFrame(rows)
    assert_chunk_indices_contiguous_and_unique(df)


def test_no_cross_split_leakage_and_no_duplicate_chunk_ids(tokenizer, cfg):
    train_df = pd.DataFrame(_build(tokenizer, cfg, document_id="t1", text="Train doc.", effective_family_id="FAM:T1", agency="DMV", effective_agency="DMV", split="train"))
    val_df = pd.DataFrame(_build(tokenizer, cfg, document_id="v1", text="Validation doc.", effective_family_id="FAM:V1", agency="SSA", effective_agency="SSA", split="validation"))
    test_df = pd.DataFrame(_build(tokenizer, cfg, document_id="te1", text="Test doc.", effective_family_id="FAM:TE1", agency="IRS", effective_agency="IRS", split="test"))

    assert_no_cross_split_leakage(train_df, val_df, test_df)
    all_chunks = pd.concat([train_df, val_df, test_df], ignore_index=True)
    assert_no_duplicate_chunk_ids(all_chunks)


def test_split_inheritance_assertion_catches_a_real_mismatch(tokenizer, cfg):
    df = pd.DataFrame(_build(tokenizer, cfg, document_id="doc7", text="Mismatch check.", split="train"))
    with pytest.raises(ValueError):
        assert_every_chunk_inherits_parent_split(df, {"doc7": "validation"})


def test_rerun_from_scratch_produces_identical_chunks(tokenizer, cfg):
    text = "Reproducibility check across independent runs. " * 150
    first = _build(tokenizer, cfg, document_id="doc8", text=text, split="test")
    second = _build(tokenizer, cfg, document_id="doc8", text=text, split="test")
    assert first == second


def test_real_data_all_eligible_documents_represented_across_chunk_files(settings):
    """Regression guard against the real, already-generated Checkpoint 5 output: every one
    of the 659 eligible documents from the Checkpoint 4 split must appear in exactly one of
    the three chunk files, with no cross-split duplication."""
    chunk_dir = settings.resolve_path(settings.family_aware.chunking.output_dir)
    train_path = chunk_dir / "train_chunks.csv"
    val_path = chunk_dir / "validation_chunks.csv"
    test_path = chunk_dir / "test_chunks.csv"
    for path in (train_path, val_path, test_path):
        assert path.exists(), f"Expected generated chunk file missing: {path}"

    train_docs = set(pd.read_csv(train_path)["document_id"].astype(str))
    val_docs = set(pd.read_csv(val_path)["document_id"].astype(str))
    test_docs = set(pd.read_csv(test_path)["document_id"].astype(str))

    assert not (train_docs & val_docs)
    assert not (train_docs & test_docs)
    assert not (val_docs & test_docs)

    split_dir = settings.resolve_path(settings.family_aware.split.output_dir)
    eligible_docs = set()
    for name in ("train.csv", "validation.csv", "test.csv"):
        eligible_docs |= set(pd.read_csv(split_dir / name)["document_id"].astype(str))

    all_chunked_docs = train_docs | val_docs | test_docs
    assert all_chunked_docs == eligible_docs
    assert len(eligible_docs) == 659
