"""Regression tests for Checkpoint 3's revised policy: family_id/effective_family_id,
agency/effective_agency, and modeling_eligibility are kept separate, and family members may
have different eligibility. These use a small synthetic DataFrame shaped like the real
fss4/fss4sp/fw4v/i9-INS-Spanish rows -- no Gemini calls, no real dataset I/O.
"""

from __future__ import annotations

import pandas as pd
import pytest

from newstart_ai.config import load_settings
from newstart_ai.data.family_audit import build_full_family_audit


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture()
def synthetic_df():
    return pd.DataFrame(
        [
            {
                "document_id": 540,
                "filename": "fss4.pdf",
                "agency": "SSA",
                "form_number": None,
                "document_type": "form",
                "text": "SS-4 Application for Employer Identification Number. "
                "Department of the Treasury. Internal Revenue Service. " * 3,
                "text_length": 300,
            },
            {
                "document_id": 541,
                "filename": "fss4sp.pdf",
                "agency": "SSA",
                "form_number": None,
                "document_type": "form",
                "text": "SS-4 Solicitud de Numero de Identificacion del Empleador. "
                "Departamento del Tesoro. Servicio de Impuestos Internos. " * 3,
                "text_length": 300,
            },
            {
                "document_id": 542,
                "filename": "fw4v.pdf",
                "agency": "SSA",
                "form_number": None,
                "document_type": "form",
                "text": "W-4V Voluntary Withholding Request. Department of the Treasury. "
                "Internal Revenue Service. Give this form to the payer. " * 3,
                "text_length": 300,
            },
            {
                "document_id": 131,
                "filename": "i9-INS-Spanish.pdf",
                "agency": "USCIS",
                "form_number": None,
                "document_type": "translated_form",
                "text": "Instrucciones para el Formulario I-9. Departamento de Seguridad "
                "Nacional. Servicio de Ciudadania e Inmigracion. " * 3,
                "text_length": 300,
            },
            {
                "document_id": 231,
                "filename": "i-9.pdf",
                "agency": "USCIS",
                "form_number": "I-9",
                "document_type": "form",
                "text": "Form I-9 Employment Eligibility Verification. " * 5,
                "text_length": 300,
            },
        ]
    )


@pytest.fixture()
def synthetic_language_audit(synthetic_df):
    # Mirrors the real Checkpoint 2 findings for these exact documents.
    return pd.DataFrame(
        [
            {"document_id": "540", "status": "confidently_english", "detected_language": "en"},
            {"document_id": "541", "status": "confidently_non_english", "detected_language": "es"},
            {"document_id": "542", "status": "confidently_english", "detected_language": "en"},
            {"document_id": "131", "status": "confidently_non_english", "detected_language": "es"},
            {"document_id": "231", "status": "confidently_english", "detected_language": "en"},
        ]
    )


def _row(audit_df, document_id):
    return audit_df[audit_df["document_id"] == str(document_id)].iloc[0]


def test_family_members_can_have_different_eligibility(settings, synthetic_df, synthetic_language_audit):
    """The core policy fix: 540 and 541 share a family but must NOT share eligibility."""
    audit_df = build_full_family_audit(synthetic_df, synthetic_language_audit, settings)
    row_540 = _row(audit_df, 540)
    row_541 = _row(audit_df, 541)

    assert row_540["family_id"] == row_541["family_id"]  # same source family
    assert row_540["final_modeling_eligibility"] == "include_english_corpus"
    assert row_541["final_modeling_eligibility"] == "exclude_non_english"


def test_effective_family_id_recomputed_after_agency_override(settings, synthetic_df, synthetic_language_audit):
    """After the SSA->IRS override, the effective family id must use the corrected
    agency, while the original family_id is preserved in the audit trail."""
    audit_df = build_full_family_audit(synthetic_df, synthetic_language_audit, settings)
    row_540 = _row(audit_df, 540)

    assert row_540["family_id"] == "SSA:SS4"
    assert row_540["effective_agency"] == "IRS"
    assert row_540["effective_family_id"] == "IRS:SS4"


def test_document_542_gets_agency_override_and_english_eligibility(settings, synthetic_df, synthetic_language_audit):
    row_542 = _row(build_full_family_audit(synthetic_df, synthetic_language_audit, settings), 542)
    assert row_542["effective_agency"] == "IRS"
    assert row_542["effective_family_id"] == "IRS:W4V"
    assert row_542["final_modeling_eligibility"] == "include_english_corpus"


def test_document_131_gets_manual_family_override_not_agency_change(settings, synthetic_df, synthetic_language_audit):
    audit_df = build_full_family_audit(synthetic_df, synthetic_language_audit, settings)
    row_131 = _row(audit_df, 131)
    row_231 = _row(audit_df, 231)  # the real I-9 form, form_number="I-9"

    assert row_131["agency"] == "USCIS"
    assert row_131["effective_agency"] == "USCIS"  # no agency change
    assert row_131["effective_family_id"] == "USCIS:I9"
    assert row_131["effective_family_id"] == row_231["family_id"]  # now matches the real family
    assert row_131["final_modeling_eligibility"] == "exclude_non_english"


def test_override_proposals_include_effective_family_id_change(settings, synthetic_df, synthetic_language_audit):
    from newstart_ai.data.family_audit import build_override_proposals

    audit_df = build_full_family_audit(synthetic_df, synthetic_language_audit, settings)
    proposals = build_override_proposals(audit_df)
    by_id = {p.document_id: p for p in proposals}

    assert "540" in by_id and "541" in by_id and "542" in by_id and "131" in by_id

    fields_540 = {c.field for c in by_id["540"].field_changes}
    assert "agency" in fields_540
    assert "effective_family_id" in fields_540

    fields_131 = {c.field for c in by_id["131"].field_changes}
    assert "agency" not in fields_131  # no agency change for 131
    assert "effective_family_id" in fields_131
