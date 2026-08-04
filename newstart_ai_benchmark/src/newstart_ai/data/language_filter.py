"""Deterministic English-language filtering for the Version 6 family-aware robustness
research (Robustness_v6_Family_Aware_Chunked_BERT.md, Checkpoint 2).

This module never deletes or silently excludes a record. Every document is classified into
exactly one of three states -- confidently_english, confidently_non_english, or
uncertain_review -- and uncertain/mixed cases are always kept pending manual review rather
than dropped. The historical `final_dataset.csv` is only ever read here, never modified;
audit outputs go to new artifacts/family_aware/ paths.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path

import pandas as pd
import py3langid.langid as langid_module

from newstart_ai.config.settings import LanguageFilterSettings, Settings
from newstart_ai.data.fingerprinting import dataset_fingerprint
from newstart_ai.schemas.language import LanguageAuditRow, LanguageFilterManifest

# Any Unicode letter, in any script -- used to detect form/numeric-code text that has too
# little real word content for language detection to be meaningful.
_ALPHA_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def get_language_identifier() -> langid_module.LanguageIdentifier:
    """Builds a py3langid identifier configured for normalized (0-1) confidence scores.

    A fresh instance is built here rather than reusing py3langid's module-level singleton,
    so enabling normalized probabilities can never affect any other code that happens to
    import py3langid elsewhere in the process.
    """
    return langid_module.LanguageIdentifier.from_pickled_model(
        langid_module.MODEL_FILE, norm_probs=True
    )


def _alphabetic_ratio(text: str) -> float:
    non_space = re.sub(r"\s+", "", text)
    if not non_space:
        return 0.0
    return len(_ALPHA_RE.findall(non_space)) / len(non_space)


def classify_document_language(
    text: str, identifier, config: LanguageFilterSettings
) -> tuple[str | None, float | None, str, str]:
    """Classifies one document's text. Returns (detected_language, confidence, status, reason).

    Deterministic: py3langid uses a fixed pretrained Naive Bayes model with no random
    initialization and no network calls, so identical text always produces identical output
    (verified in tests/test_language_filter.py).

    `status` is always one of "confidently_english", "confidently_non_english", or
    "uncertain_review" -- the last of these means "keep pending manual review," never
    "exclude."
    """
    stripped = (text or "").strip()

    if not stripped:
        return None, None, "uncertain_review", "empty_text"

    if len(stripped) < config.min_text_length:
        return None, None, "uncertain_review", "text_too_short_for_reliable_detection"

    if _alphabetic_ratio(stripped) < config.min_alphabetic_ratio:
        return None, None, "uncertain_review", "insufficient_alphabetic_content"

    # Mixed-language check: split into equal windows and classify each independently. A
    # whole-document classification can be extremely (and misleadingly) confident even when
    # the document is genuinely half one language and half another, because the underlying
    # Naive Bayes model's probabilities are dominated by whichever language has slightly
    # more character n-gram evidence overall.
    if config.mixed_language_window_count >= 2:
        window_count = config.mixed_language_window_count
        window_size = len(stripped) // window_count
        windows = [
            stripped[i * window_size : (i + 1) * window_size if i < window_count - 1 else None]
            for i in range(window_count)
        ]
        if all(len(w) >= config.min_text_length for w in windows):
            window_langs = {identifier.classify(w)[0] for w in windows}
            if len(window_langs) > 1:
                whole_lang, whole_confidence = identifier.classify(stripped)
                return (
                    whole_lang,
                    float(whole_confidence),
                    "uncertain_review",
                    "mixed_language_signal_across_document",
                )

    detected_language, confidence = identifier.classify(stripped)
    confidence = float(confidence)

    if detected_language == config.target_language and confidence >= config.confident_english_threshold:
        return detected_language, confidence, "confidently_english", "detector_high_confidence"

    if (
        detected_language != config.target_language
        and confidence >= config.confident_non_english_threshold
    ):
        return detected_language, confidence, "confidently_non_english", "detector_high_confidence"

    return detected_language, confidence, "uncertain_review", "low_detection_confidence"


def build_language_audit(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Runs language classification over every row of df, returning one audit row per
    document. Never modifies df or the source CSV."""
    config = settings.family_aware.language_filter
    identifier = get_language_identifier()
    ds_cfg = settings.base.dataset

    rows: list[dict] = []
    for _, record in df.iterrows():
        text = record[ds_cfg.text_column]
        lang, confidence, status, reason = classify_document_language(text, identifier, config)
        form_number = record.get("form_number")
        rows.append(
            LanguageAuditRow(
                document_id=str(record[ds_cfg.id_column]),
                agency=record[ds_cfg.label_column],
                filename=str(record.get("filename", "")),
                form_number=None if pd.isna(form_number) else str(form_number),
                text_length=int(record.get("text_length") or len(str(text))),
                detected_language=lang,
                confidence=confidence,
                status=status,
                reason=reason,
            ).model_dump()
        )
    return pd.DataFrame(rows)


def build_language_filter_manifest(
    audit_df: pd.DataFrame, source_df: pd.DataFrame, settings: Settings
) -> LanguageFilterManifest:
    config = settings.family_aware.language_filter

    counts_by_status = audit_df["status"].value_counts().to_dict()
    counts_by_status_and_agency = {
        status: group["agency"].value_counts().to_dict()
        for status, group in audit_df.groupby("status")
    }

    try:
        detector_version = package_version(config.detector_name)
    except Exception:
        detector_version = "unknown"

    return LanguageFilterManifest(
        detector_name=config.detector_name,
        detector_version=detector_version,
        detector_config=config.model_dump(),
        created_at=datetime.now(timezone.utc).isoformat(),
        source_dataset_path=settings.base.dataset.path,
        source_dataset_fingerprint=dataset_fingerprint(source_df, settings),
        total_documents=len(source_df),
        counts_by_status=counts_by_status,
        counts_by_status_and_agency=counts_by_status_and_agency,
        notes=[
            "uncertain_review documents are retained pending manual review, never deleted.",
            "This audit never modifies the source dataset; see language_audit_v1.csv for "
            "per-document decisions.",
        ],
    )


def save_language_audit(
    audit_df: pd.DataFrame, manifest: LanguageFilterManifest, settings: Settings
) -> tuple[Path, Path]:
    """Saves the audit CSV and manifest JSON under artifacts/family_aware/reports/ --
    never under the historical artifacts/reports/ path."""
    reports_dir = settings.resolve_path("artifacts/family_aware/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    audit_path = reports_dir / "language_audit_v1.csv"
    manifest_path = reports_dir / "language_audit_manifest_v1.json"

    audit_df.to_csv(audit_path, index=False)
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))

    return audit_path, manifest_path
