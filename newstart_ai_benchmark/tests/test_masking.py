"""Tests for Version 6 Checkpoint 6 identifier masking."""

from __future__ import annotations

import pandas as pd
import pytest

from newstart_ai.config import load_settings
from newstart_ai.data.masking import (
    apply_masking,
    build_masked_chunks,
    build_masked_documents,
    build_masking_rules,
    mask_document,
)


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def rules(settings):
    return build_masking_rules(settings.family_aware.masking)


def test_agency_full_name_and_abbreviation_are_masked(rules):
    text = "This document was issued by the United States Citizenship and Immigration Services (USCIS)."
    masked, counts = apply_masking(text, rules)
    assert "United States Citizenship and Immigration Services" not in masked
    assert "USCIS" not in masked
    assert counts["agency_identifier:USCIS"] == 2


def test_form_number_is_masked_but_word_form_is_preserved(rules):
    text = "Please complete Form I-9 and submit it to your employer."
    masked, counts = apply_masking(text, rules)
    assert "I-9" not in masked
    assert "Form" in masked
    assert counts["form_number"] == 1


def test_omb_number_is_masked(rules):
    text = "OMB No. 1615-0047 appears in the top right corner."
    masked, counts = apply_masking(text, rules)
    assert "1615-0047" not in masked
    assert counts["omb_number"] == 1


def test_gov_url_is_masked(rules):
    text = "Visit www.uscis.gov/i-9 for more information."
    masked, counts = apply_masking(text, rules)
    assert "uscis.gov" not in masked
    assert counts["agency_url"] == 1


def test_no_identifiers_found_case(rules):
    text = "Please provide your full legal name, date of birth, and current mailing address below."
    masked, counts = apply_masking(text, rules)
    assert masked == text
    assert sum(counts.values()) == 0


def test_multiple_identifiers_in_one_document(rules):
    text = (
        "Form SS-4 OMB No. 1545-0003. This form is issued by the Internal Revenue Service "
        "(IRS), Department of the Treasury. See www.irs.gov/forms for details."
    )
    masked, counts = apply_masking(text, rules)
    assert counts["form_number"] == 1
    assert counts["omb_number"] == 1
    assert counts["agency_identifier:IRS"] == 3  # Internal Revenue Service, IRS, Department of the Treasury
    assert counts["agency_url"] == 1
    assert "SS-4" not in masked and "1545-0003" not in masked and "Treasury" not in masked


def test_ordinary_semantic_content_is_never_falsely_masked(rules):
    """False-positive protection: generic references to a Social Security NUMBER (not the
    agency itself), or an unrelated 'Form of' phrase, or plain digits, must survive intact."""
    text = "Enter your social security number below. This form of identification is required."
    masked, counts = apply_masking(text, rules)
    assert masked == text
    assert sum(counts.values()) == 0


def test_masking_never_changes_ground_truth_label_column():
    # Masking operates on text only; the caller's agency/effective_agency columns are
    # copied through unmodified by build_masked_documents (never derived from text).
    df = pd.DataFrame(
        {
            "document_id": ["1"],
            "text": ["Issued by USCIS."],
            "effective_family_id": ["FAM:1"],
            "agency": ["USCIS"],
            "effective_agency": ["USCIS"],
        }
    )
    from newstart_ai.config import load_settings as _load

    settings = _load()
    masked_df = build_masked_documents(df, "train", settings)
    assert masked_df.loc[0, "agency"] == "USCIS"
    assert masked_df.loc[0, "effective_agency"] == "USCIS"


def test_mask_document_preserves_original_and_produces_separate_masked_derivative(rules):
    original = "Issued by the Social Security Administration (SSA)."
    record = mask_document("1", original, rules, "v1")
    assert record["masked_text"] != original
    assert "original_text_hash" in record and "masked_text_hash" in record
    assert record["original_text_hash"] != record["masked_text_hash"]


def test_masking_deterministic_across_independent_calls(rules):
    text = "Form N-400, OMB No. 1615-0052, issued by USCIS. Visit uscis.gov."
    masked_a, counts_a = apply_masking(text, rules)
    masked_b, counts_b = apply_masking(text, rules)
    assert masked_a == masked_b
    assert counts_a == counts_b


def test_build_masked_chunks_preserves_chunk_provenance_columns(settings):
    chunks_df = pd.DataFrame(
        {
            "chunk_id": ["c1", "c2"],
            "document_id": ["1", "1"],
            "chunk_index": [0, 1],
            "total_chunks": [2, 2],
            "token_start": [0, 400],
            "token_end": [400, 800],
            "split": ["train", "train"],
            "chunk_text": ["Issued by USCIS.", "No identifiers in this second chunk at all."],
        }
    )
    masked_chunks = build_masked_chunks(chunks_df, settings)
    assert list(masked_chunks["chunk_id"]) == ["c1", "c2"]
    assert list(masked_chunks["chunk_index"]) == [0, 1]
    assert masked_chunks.loc[0, "total_replacements"] == 1
    assert masked_chunks.loc[1, "total_replacements"] == 0


def test_identical_rules_applied_regardless_of_documents_own_agency_label(rules):
    """Apply-identically requirement: the same rule set masks an SSA mention inside a DMV
    document exactly as it would inside an SSA document -- no label-conditional branching."""
    text_in_dmv_doc = "This DMV form also references the Social Security Administration."
    masked, counts = apply_masking(text_in_dmv_doc, rules)
    assert counts["agency_identifier:SSA"] == 1
    assert counts["agency_identifier:DMV"] == 1
