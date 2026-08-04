"""Tests for the Version 6 family-discovery module (Checkpoint 3).

These exercise `assign_family()`/`derive_filename_code()` directly against small synthetic
inputs modeled on real filenames observed in `final_dataset.csv` -- they never read the real
dataset, call Gemini, or write to disk. Every required scenario from
Robustness_v6_Family_Aware_Chunked_BERT.md's Checkpoint 3 review notes is covered.
"""

from __future__ import annotations

import pandas as pd

from newstart_ai.data.family_grouping import (
    assign_family,
    build_family_assignments,
    derive_filename_code,
    find_conflicting_agency_families,
)


def test_form_and_instructions_share_a_family_via_form_number():
    form = assign_family("1", "USCIS", "i-765.pdf", "I-765")
    instr = assign_family("2", "USCIS", "i-765instr.pdf", "I-765")
    assert form.family_key == instr.family_key
    assert form.evidence_type == "form_number_exact"


def test_form_and_supplement_share_a_family_via_form_number():
    form = assign_family("1", "USCIS", "i-914.pdf", "I-914")
    supplement = assign_family("2", "USCIS", "i-914supa.pdf", "I-914")
    assert form.family_key == supplement.family_key


def test_translated_and_english_versions_linked_when_supported():
    # Modeled on the real fss4.pdf / fss4sp.pdf pair -- no form_number for either, linked
    # only via the filename code after stripping the trailing Spanish-language marker.
    english = assign_family("540", "SSA", "fss4.pdf", None)
    spanish = assign_family("541", "SSA", "fss4sp.pdf", None)
    assert english.family_key == spanish.family_key
    assert english.evidence_type == "filename_code_match"


def test_unrelated_forms_are_not_grouped_by_broad_agency_or_topic():
    a = assign_family("1", "USCIS", "i-765.pdf", "I-765")
    b = assign_family("2", "USCIS", "i-129.pdf", "I-129")
    assert a.family_key != b.family_key


def test_dmv_filename_based_grouping_uses_trailing_code_pattern():
    # Modeled on the real DL-93 translation family (base + several language variants).
    base = assign_family("1", "DMV", "verification-authorization-dl-93.pdf", None)
    thai = assign_family("2", "DMV", "verification-authorization-thai-dl-93-th.pdf", None)
    khmer = assign_family("3", "DMV", "verification-authorization-khmer-dl-93-km.pdf", None)
    assert base.family_key == thai.family_key == khmer.family_key
    assert base.evidence_type == "filename_code_match"


def test_dmv_unrelated_descriptive_filenames_are_not_merged():
    a = assign_family("1", "DMV", "renewal-list.pdf", None)
    b = assign_family("2", "DMV", "notice-of-change-of-address.pdf", None)
    assert a.family_key != b.family_key
    assert a.evidence_type == "singleton_no_evidence"
    assert b.evidence_type == "singleton_no_evidence"


def test_singleton_preservation_when_no_evidence():
    result = assign_family("99", "DMV", "certificate-of-facts-re-unsatisfied-judgment.pdf", None)
    assert result.evidence_type == "singleton_no_evidence"
    assert result.family_key == "SINGLETON:DMV:99"


def test_family_ids_are_deterministic():
    first = assign_family("1", "USCIS", "i-765.pdf", "I-765")
    second = assign_family("1", "USCIS", "i-765.pdf", "I-765")
    assert first.family_key == second.family_key
    assert derive_filename_code("verification-authorization-dl-93.pdf") == derive_filename_code(
        "verification-authorization-dl-93.pdf"
    )


def test_conflicting_agency_detection_invariant_holds():
    # assign_family() always namespaces by the document's own agency, so two documents
    # with the same code but different agencies can never land in the same family_id.
    df = pd.DataFrame(
        [
            {"document_id": "1", "agency": "USCIS", "filename": "x-1.pdf", "form_number": "X-1"},
            {"document_id": "2", "agency": "DMV", "filename": "x-1.pdf", "form_number": "X-1"},
        ]
    )

    class _DatasetCfg:
        id_column = "document_id"
        label_column = "agency"

    class _BaseCfg:
        dataset = _DatasetCfg()

    class _Settings:
        base = _BaseCfg()

    assignments = build_family_assignments(df, _Settings())
    conflicts = find_conflicting_agency_families(assignments)
    assert len(conflicts) == 0


def test_manual_override_validation_document_540_and_541_share_family():
    """Regression test pinned to the real Checkpoint 3 finding: documents 540 (fss4.pdf)
    and 541 (fss4sp.pdf), both labeled SSA with no form_number, must be linked into one
    family so any agency-label override applies consistently to both."""
    doc_540 = assign_family("540", "SSA", "fss4.pdf", None)
    doc_541 = assign_family("541", "SSA", "fss4sp.pdf", None)
    assert doc_540.family_key == doc_541.family_key == "SSA:SS4"


def test_weak_filename_evidence_does_not_confidently_link_document_131():
    """Regression test: document 131 (i9-INS-Spanish.pdf, no form_number) is NOT
    confidently linked to the USCIS:I9 family -- it's a documented miss, correctly left as
    its own singleton and flagged for manual review rather than force-merged."""
    result = assign_family("131", "USCIS", "i9-INS-Spanish.pdf", None)
    assert result.family_key != "USCIS:I9"
