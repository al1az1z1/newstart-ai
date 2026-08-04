"""Tests for the Version 6 family-aware split (Checkpoint 4).

Uses a small synthetic eligible-corpus DataFrame (shaped like the real data: several
multi-document families per agency, a tight IRS-like agency with only pair/singleton
families) so these tests never touch the real dataset or call Gemini.
"""

from __future__ import annotations

import pandas as pd
import pytest

from newstart_ai.config import load_settings
from newstart_ai.data.family_split import (
    assert_every_eligible_document_assigned_exactly_once,
    assert_no_document_overlap,
    assert_no_excluded_document_in_splits,
    assert_no_family_overlap,
    assign_families_to_splits,
    create_family_aware_split,
    find_agencies_missing_by_split,
    fingerprint_split,
)


@pytest.fixture(scope="module")
def settings():
    return load_settings()


def _make_family(agency: str, family_id: str, size: int, start_id: int) -> list[dict]:
    return [
        {
            "document_id": str(start_id + i),
            "effective_agency": agency,
            "effective_family_id": family_id,
            "final_modeling_eligibility": "include_english_corpus",
        }
        for i in range(size)
    ]


@pytest.fixture()
def synthetic_eligible_df():
    rows = []
    doc_id = 1
    # A "large" agency with plenty of families of varying size (like DMV/USCIS/SSA).
    for i in range(30):
        size = 1 if i % 3 else 2
        rows += _make_family("BIGAGENCY", f"BIGAGENCY:F{i}", size, doc_id)
        doc_id += size
    # A tight agency with few, small families (like IRS).
    for i in range(6):
        size = 2 if i % 2 == 0 else 1
        rows += _make_family("TINYAGENCY", f"TINYAGENCY:F{i}", size, doc_id)
        doc_id += size
    return pd.DataFrame(rows)


@pytest.fixture()
def excluded_df(synthetic_eligible_df):
    """Adds a few excluded (non-eligible) documents that must never appear in a split."""
    excluded = pd.DataFrame(
        [
            {
                "document_id": "9001",
                "effective_agency": "BIGAGENCY",
                "effective_family_id": "BIGAGENCY:EXCLUDED",
                "final_modeling_eligibility": "exclude_non_english",
            }
        ]
    )
    return pd.concat([synthetic_eligible_df, excluded], ignore_index=True)


def test_every_family_stays_in_one_split(settings, synthetic_eligible_df):
    train_df, val_df, test_df, family_map = create_family_aware_split(synthetic_eligible_df, settings)
    assert_no_family_overlap(train_df, val_df, test_df)


def test_zero_document_overlap(settings, synthetic_eligible_df):
    train_df, val_df, test_df, _ = create_family_aware_split(synthetic_eligible_df, settings)
    assert_no_document_overlap(train_df, val_df, test_df)


def test_every_eligible_document_assigned_exactly_once(settings, synthetic_eligible_df):
    train_df, val_df, test_df, _ = create_family_aware_split(synthetic_eligible_df, settings)
    assert_every_eligible_document_assigned_exactly_once(synthetic_eligible_df, train_df, val_df, test_df)


def test_excluded_documents_never_appear_in_a_split(settings, excluded_df):
    eligible_only = excluded_df[excluded_df["final_modeling_eligibility"] == "include_english_corpus"]
    train_df, val_df, test_df, _ = create_family_aware_split(eligible_only, settings)
    assert_no_excluded_document_in_splits(excluded_df, train_df, val_df, test_df)
    all_split_ids = set(train_df["document_id"]) | set(val_df["document_id"]) | set(test_df["document_id"])
    assert "9001" not in all_split_ids


def test_tight_agency_still_appears_in_every_split(settings, synthetic_eligible_df):
    """TINYAGENCY has only 6 families (like IRS's 18) -- the coverage-guarantee step must
    still place it in all three splits."""
    train_df, val_df, test_df, _ = create_family_aware_split(synthetic_eligible_df, settings)
    missing = find_agencies_missing_by_split(train_df, val_df, test_df, ["BIGAGENCY", "TINYAGENCY"])
    assert missing == {}


def test_split_percentages_are_reasonably_close_to_configured_ratios(settings, synthetic_eligible_df):
    train_df, val_df, test_df, _ = create_family_aware_split(synthetic_eligible_df, settings)
    total = len(train_df) + len(val_df) + len(test_df)
    train_pct = len(train_df) / total
    # Family-integrity constraints mean this won't be exact -- just sanity-check it's in
    # the right neighborhood of the configured 0.70 train ratio.
    assert 0.55 <= train_pct <= 0.85


def test_split_assignment_is_deterministic(settings, synthetic_eligible_df):
    first = assign_families_to_splits(synthetic_eligible_df, settings)
    second = assign_families_to_splits(synthetic_eligible_df, settings)
    assert first == second


def test_rerun_with_identical_inputs_produces_identical_split_fingerprints(settings, synthetic_eligible_df):
    train_a, val_a, test_a, _ = create_family_aware_split(synthetic_eligible_df, settings)
    train_b, val_b, test_b, _ = create_family_aware_split(synthetic_eligible_df, settings)
    assert fingerprint_split(train_a) == fingerprint_split(train_b)
    assert fingerprint_split(val_a) == fingerprint_split(val_b)
    assert fingerprint_split(test_a) == fingerprint_split(test_b)


def test_no_family_is_split_across_partitions_even_under_deficit_pressure(settings):
    """Regression guard: a single oversized family must land entirely in one split, never
    partially moved to another split to chase the target ratio."""
    rows = _make_family("SOLOAGENCY", "SOLOAGENCY:BIGFAMILY", 20, 1)
    rows += _make_family("SOLOAGENCY", "SOLOAGENCY:SMALL1", 1, 21)
    rows += _make_family("SOLOAGENCY", "SOLOAGENCY:SMALL2", 1, 22)
    df = pd.DataFrame(rows)

    train_df, val_df, test_df, family_map = create_family_aware_split(df, settings)
    # The 20-document family must appear in exactly one split's document set.
    big_family_splits = set()
    for name, split_df in (("train", train_df), ("validation", val_df), ("test", test_df)):
        if (split_df["effective_family_id"] == "SOLOAGENCY:BIGFAMILY").any():
            big_family_splits.add(name)
    assert len(big_family_splits) == 1
