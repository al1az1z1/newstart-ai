"""Tests for the Version 6 English-filtering module (Checkpoint 2).

These tests exercise `classify_document_language()` directly against small synthetic
strings -- they never touch the real `final_dataset.csv`, never call Gemini, and never
write to disk. Every required case from Robustness_v6_Family_Aware_Chunked_BERT.md's
Checkpoint 2 review notes is covered: English, clearly non-English, mixed-language, very
short, numeric/form-like, and empty text -- plus determinism, since that's a hard
requirement for the audit to be reproducible.
"""

from __future__ import annotations

import pytest

from newstart_ai.config.settings import LanguageFilterSettings
from newstart_ai.data.language_filter import classify_document_language, get_language_identifier

CONFIG = LanguageFilterSettings(
    detector_name="py3langid",
    target_language="en",
    min_text_length=30,
    min_alphabetic_ratio=0.5,
    confident_english_threshold=0.90,
    confident_non_english_threshold=0.90,
    mixed_language_window_count=2,
)


@pytest.fixture(scope="module")
def identifier():
    return get_language_identifier()


def test_clean_english_is_confidently_english(identifier):
    text = "This form must be completed in full before submission to the appropriate office."
    lang, confidence, status, reason = classify_document_language(text, identifier, CONFIG)
    assert status == "confidently_english"
    assert lang == "en"
    assert confidence >= CONFIG.confident_english_threshold


def test_clean_spanish_is_confidently_non_english(identifier):
    text = "Este formulario debe completarse en su totalidad antes de enviarlo a la oficina."
    lang, confidence, status, reason = classify_document_language(text, identifier, CONFIG)
    assert status == "confidently_non_english"
    assert lang == "es"
    assert confidence >= CONFIG.confident_non_english_threshold


def test_mixed_language_document_is_uncertain_not_deleted(identifier):
    english_half = (
        "This form must be completed in full before submission to the appropriate "
        "office for processing and review by an authorized agent."
    )
    spanish_half = (
        "Este formulario debe completarse en su totalidad antes de enviarlo a la "
        "oficina correspondiente para su procesamiento y revision."
    )
    text = english_half + " " + spanish_half
    lang, confidence, status, reason = classify_document_language(text, identifier, CONFIG)
    # The whole-document signal alone would call this confidently one language or the
    # other (verified manually to be ~0.999 confidence) -- window disagreement must
    # override that and mark it for review instead.
    assert status == "uncertain_review"
    assert reason == "mixed_language_signal_across_document"


def test_very_short_text_is_uncertain_not_deleted(identifier):
    lang, confidence, status, reason = classify_document_language("OK", identifier, CONFIG)
    assert status == "uncertain_review"
    assert reason == "text_too_short_for_reliable_detection"
    assert lang is None and confidence is None


def test_numeric_form_like_text_is_uncertain_not_deleted(identifier):
    text = "1040 2021 0001 998877 33 44 55 66 77 88 99 00 11 22 33 44 55 66 77 88 99 00"
    lang, confidence, status, reason = classify_document_language(text, identifier, CONFIG)
    assert status == "uncertain_review"
    assert reason == "insufficient_alphabetic_content"


def test_empty_text_is_uncertain_not_deleted(identifier):
    lang, confidence, status, reason = classify_document_language("", identifier, CONFIG)
    assert status == "uncertain_review"
    assert reason == "empty_text"
    assert lang is None and confidence is None


def test_whitespace_only_text_is_treated_as_empty(identifier):
    lang, confidence, status, reason = classify_document_language("   \n\t  ", identifier, CONFIG)
    assert status == "uncertain_review"
    assert reason == "empty_text"


def test_none_text_is_treated_as_empty(identifier):
    lang, confidence, status, reason = classify_document_language(None, identifier, CONFIG)
    assert status == "uncertain_review"
    assert reason == "empty_text"


def test_classification_is_deterministic(identifier):
    text = "This form must be completed in full before submission to the appropriate office."
    first = classify_document_language(text, identifier, CONFIG)
    second = classify_document_language(text, identifier, CONFIG)
    third = classify_document_language(text, identifier, CONFIG)
    assert first == second == third


def test_real_document_541_text_is_confidently_non_english(identifier):
    """Regression test pinned to the actual Checkpoint 2 investigation: document 541
    (filename fss4sp.pdf, labeled SSA) is IRS Form SS-4 in Spanish."""
    text = (
        "SS-4 Solicitud de Numero de Identificacion del Empleador (EIN) "
        "Formulario OMB No. 1545-0003 Departamento del Tesoro Servicio de "
        "Impuestos Internos Vea las instrucciones por separado para cada linea."
    )
    lang, confidence, status, reason = classify_document_language(text, identifier, CONFIG)
    assert status == "confidently_non_english"
    assert lang == "es"
