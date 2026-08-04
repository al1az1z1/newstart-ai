"""Orchestrates Checkpoint 3's family audit: combines the Checkpoint 2 language audit with
family discovery (family_grouping.py) into one per-document audit table, produces every
report category the checkpoint requires, and proposes -- but never applies -- agency,
family, and eligibility overrides.

Four concepts are kept deliberately separate throughout this module (see
schemas/family.py's FamilyAuditRow docstring): `family_id` vs `effective_family_id`,
`agency` vs `effective_agency`, `language_status` (a fact) vs `modeling_eligibility` (a
decision, made per document -- never forced identical across a family).

Nothing in this module modifies `final_dataset.csv`, historical splits, or historical
artifacts. Every output is written under artifacts/family_aware/.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from newstart_ai.config.settings import Settings
from newstart_ai.data.family_grouping import (
    assign_family,
    build_family_assignments,
    find_conflicting_agency_families,
    find_cross_agency_code_conflicts,
    find_exact_duplicate_candidates,
    find_near_duplicate_candidates,
)
from newstart_ai.data.fingerprinting import dataset_fingerprint
from newstart_ai.data.language_filter import build_language_audit
from newstart_ai.schemas.family import FamilyAuditManifest, ManualReviewFlag, OverrideFieldChange, OverrideProposal

_KNOWN_DOCTYPES = {"form", "instructions", "supplement", "checklist", "translated_form"}

# Manually inspected findings from this checkpoint's revision (direct reading of each
# document's substantive text, not filename alone -- see the Checkpoint 3 revision report
# for the full evidence behind every entry). `None` means "no change from the automated
# result" for that field.
#
# agency_override:            proposed effective_agency, or None
# family_override:            a manually forced effective_family_id, or None (when set,
#                              this takes precedence over agency-based recomputation --
#                              used for document 131, which needs no agency change but does
#                              need a family fix the automated code-matcher couldn't make)
# final_modeling_eligibility:  the per-document decision, independent of family membership
# manual_language_notes:      what direct inspection of the substantive text found
# confidence:                 "confident_override" (ready for approval) or
#                              "flagged_for_review" (evidence recorded, no override proposed)
_MANUAL_FINDINGS: dict[str, dict] = {
    "540": {
        "agency_override": "IRS",
        "family_override": None,
        "final_modeling_eligibility": "include_english_corpus",
        "manual_language_notes": (
            "Direct inspection: substantive text is English ('SS-4 Application for "
            "Employer Identification Number... Department of the Treasury / Internal "
            "Revenue Service'). Not excluded merely for matching its Spanish sibling "
            "(document 541) -- each document's eligibility is judged independently."
        ),
        "evidence": [
            "filename='fss4.pdf'",
            "Literal strings 'Department of the Treasury' and 'Internal Revenue Service' "
            "and OMB control number 1545-0003 (the real OMB number for IRS Form SS-4) "
            "appear verbatim in the extracted text.",
            "Shares family SS4 with document 541 (fss4sp.pdf, the Spanish translation).",
        ],
        "confidence": "confident_override",
    },
    "541": {
        "agency_override": "IRS",
        "family_override": None,
        "final_modeling_eligibility": "exclude_non_english",
        "manual_language_notes": (
            "Direct inspection: substantive text is Spanish ('SS-4 Solicitud de Numero de "
            "Identificacion del Empleador (EIN)'). Retained as exclude_non_english "
            "unchanged from Checkpoint 2 -- the agency correction does not change this."
        ),
        "evidence": [
            "filename='fss4sp.pdf'",
            "Same OMB number (1545-0003) and issuing-agency strings as document 540.",
            "Shares family SS4 with document 540 (fss4.pdf, the English original).",
        ],
        "confidence": "confident_override",
    },
    "542": {
        "agency_override": "IRS",
        "family_override": None,
        "final_modeling_eligibility": "include_english_corpus",
        "manual_language_notes": "Direct inspection: substantive text is English.",
        "evidence": [
            "filename='fw4v.pdf' -- IRS Form W-4V, Voluntary Withholding Request.",
            "Text reads 'W-4V Voluntary Withholding Request ... OMB No. 1545-0074 "
            "... Department of the Treasury ... Internal Revenue Service' -- OMB "
            "prefix 1545 is IRS's own series, confirming IRS as the issuing/publishing "
            "agency.",
            "The instruction 'Give Form W-4V to the payer of your payments. Do not send "
            "it to the IRS' describes where the COMPLETED form is submitted (e.g. SSA, "
            "as a payer), not who authored/publishes the form. Per the experiment's "
            "target definition (the agency responsible for/publishing the document), "
            "this does not make SSA the correct label.",
            "No other document in the dataset already carries a W-4V code under any "
            "agency, so the recomputed family (IRS:W4V) remains a singleton, not a merge.",
        ],
        "confidence": "confident_override",
    },
    "131": {
        "agency_override": None,
        "family_override": "USCIS:I9",
        "final_modeling_eligibility": "exclude_non_english",
        "manual_language_notes": (
            "Direct inspection: substantive text is Spanish ('Instrucciones para el "
            "Formulario I-9, Verificacion de Elegibilidad de Empleo'), confirming this is "
            "a Spanish translation of the I-9 instructions, not a different form."
        ),
        "evidence": [
            "filename='i9-INS-Spanish.pdf', document_type=translated_form.",
            "Text explicitly names 'Formulario I-9', 'OMB No. 1615-0047' (the real OMB "
            "number for USCIS Form I-9), 'U S C I S', and 'Departamento de Seguridad "
            "Nacional' (Department of Homeland Security) / 'Servicio de Ciudadania e "
            "Inmigracion de Estados Unidos' (USCIS's full name in Spanish).",
            "Automated filename-code matching produced USCIS:I9INS (a documented miss, "
            "since 'INS' -- the historical Immigration and Naturalization Service "
            "abbreviation -- is not a recognized language/doctype suffix); this manual "
            "override corrects it to the true family USCIS:I9 (form 231, instructions "
            "77, this document).",
        ],
        "confidence": "confident_override",
    },
    "210": {
        "agency_override": None,
        "family_override": None,
        "final_modeling_eligibility": "exclude_non_english",
        "manual_language_notes": (
            "Direct inspection of the body text (not just the filename's '_PSH' suffix) "
            "confirms Pashto-script content throughout, e.g. the opening line "
            "'ترازو تينامئ "
            "ينروک' alongside 'OMB No. 1615-0067'. English "
            "appears only in boilerplate ('Page 11 Form I-589 Supplement A Edition "
            "07/26/22'), not as substantive content."
        ),
        "evidence": [
            "Already correctly grouped in family USCIS:I589 (26 members) via form_number "
            "-- family membership needs no change, only the eligibility decision.",
        ],
        "confidence": "confident_override",
    },
    "261": {
        "agency_override": None,
        "family_override": None,
        "final_modeling_eligibility": "exclude_insufficient_text",
        "manual_language_notes": (
            "Direct inspection: the extracted text is English but consists entirely of "
            "repeated form-field labels (STATE OF CALIFORNIA, RENEWAL LIST, CONTACT "
            "PERSON, TELEPHONE NUMBER, ...), the same short block duplicated several "
            "times -- no substantive prose. Excluded for insufficient substantive text, "
            "not for being non-English."
        ),
        "evidence": ["text_length=401, entirely field-label boilerplate, no sentences."],
        "confidence": "confident_override",
    },
    "361": {
        "agency_override": None,
        "family_override": None,
        "final_modeling_eligibility": "exclude_non_english",
        "manual_language_notes": (
            "Direct inspection confirms Khmer-script content throughout (matches the "
            "filename's 'khmer'/'-km' markers), referencing 'Department of Homeland "
            "Security (DHS)' and 'U.S. Citizenship and Immigration Services (USCIS)' in "
            "English within otherwise Khmer text."
        ),
        "evidence": [
            "Already correctly grouped in family DMV:DL94 (5 members) via filename code "
            "-- family membership needs no change, only the eligibility decision.",
        ],
        "confidence": "confident_override",
    },
    "397": {
        "agency_override": None,
        "family_override": None,
        "final_modeling_eligibility": "include_english_corpus",
        "manual_language_notes": (
            "Direct inspection: substantive English content is present and clear at the "
            "start ('VESSEL VERIFICATION BOATS, THIS FORM MUST BE COMPLETED IN FULL') and "
            "end ('under penalty of perjury under the laws of the State of California "
            "that the foregoing is true and correct...'). A middle section is garbled "
            "(character-scrambled table/checkbox extraction, not a foreign language) -- "
            "this is a text-EXTRACTION-quality issue, not a language issue, and is noted "
            "for awareness rather than excluded on language grounds."
        ),
        "evidence": ["Originally flagged mixed-language; garbled table text, not a real second language."],
        "confidence": "confident_override",
    },
    "634": {
        "agency_override": None,
        "family_override": None,
        "final_modeling_eligibility": "include_english_corpus",
        "manual_language_notes": (
            "Direct inspection: clear, substantive, well-formed English text throughout "
            "('Social Security Administration OMB No. 0960-0681, FUNCTION REPORT - "
            "ADULT...'). The automated insufficient_alphabetic_content flag was driven by "
            "the form's many blank-line/checkbox fields diluting the alphabetic ratio, "
            "not by an actual lack of substantive English content."
        ),
        "evidence": ["text_length=21799, real narrative instructions confirmed by direct reading."],
        "confidence": "confident_override",
    },
}


def build_full_family_audit(
    df: pd.DataFrame, language_audit_df: pd.DataFrame, settings: Settings
) -> pd.DataFrame:
    """Merges the Checkpoint 2 language audit with this checkpoint's family assignments,
    then layers on this revision's manually-confirmed findings. Never modifies df."""
    assignments = build_family_assignments(df, settings)

    merged = assignments.merge(
        language_audit_df[["document_id", "status", "detected_language"]],
        on="document_id",
        how="left",
    ).rename(columns={"status": "language_status"})

    def relationship_type(row):
        if row["document_type"] in _KNOWN_DOCTYPES:
            return row["document_type"]
        return "singleton" if row["family_size"] == 1 else "unclassified"

    def recommended_modeling_eligibility(row):
        if row["language_status"] == "confidently_non_english":
            return "exclude_non_english"
        if row["language_status"] == "uncertain_review":
            return "pending_review"
        return "include_english_corpus"

    merged["relationship_type"] = merged.apply(relationship_type, axis=1)
    merged["recommended_modeling_eligibility"] = merged.apply(recommended_modeling_eligibility, axis=1)

    # --- Layer on this revision's manual findings ---
    merged["agency_override_proposed"] = None
    merged["family_override_proposed"] = None
    merged["manual_language_notes"] = None
    merged["final_modeling_eligibility"] = merged["recommended_modeling_eligibility"]

    for document_id, finding in _MANUAL_FINDINGS.items():
        row_mask = merged["document_id"] == document_id
        if not row_mask.any():
            continue
        merged.loc[row_mask, "agency_override_proposed"] = finding["agency_override"]
        merged.loc[row_mask, "family_override_proposed"] = finding["family_override"]
        merged.loc[row_mask, "manual_language_notes"] = finding["manual_language_notes"]
        merged.loc[row_mask, "final_modeling_eligibility"] = finding["final_modeling_eligibility"]

    # effective_agency: the override if proposed, else the original label.
    merged["effective_agency"] = merged["agency_override_proposed"].fillna(merged["agency"])

    # effective_family_id: a manual family override wins if present; otherwise, if the
    # agency changed, recompute using the SAME tested assign_family() logic (reused, not
    # re-implemented) with the corrected agency; otherwise it's unchanged.
    def effective_family_id(row):
        if row["family_override_proposed"]:
            return row["family_override_proposed"]
        if row["effective_agency"] != row["agency"]:
            recomputed = assign_family(
                document_id=row["document_id"],
                agency=row["effective_agency"],
                filename=row["filename"],
                form_number=row["form_number"],
            )
            return recomputed.family_key
        return row["family_id"]

    merged["effective_family_id"] = merged.apply(effective_family_id, axis=1)

    def review_status(row):
        if row["document_id"] in _MANUAL_FINDINGS:
            return "resolved_manual_review"
        if row["language_status"] == "uncertain_review":
            return "needs_review"
        if row["family_size"] == 1:
            return "singleton_confirmed"
        return "auto_grouped"

    merged["review_status"] = merged.apply(review_status, axis=1)
    merged["conflict_reason"] = None

    conflicting_families = find_conflicting_agency_families(merged)
    if len(conflicting_families):
        conflicting_ids = set(sum(conflicting_families["document_ids"].tolist(), []))
        merged.loc[merged["document_id"].isin(conflicting_ids), "conflict_reason"] = (
            "family_id contains more than one agency label (should never happen by "
            "construction -- see conflicting_agency_families report)"
        )

    return merged


def build_category_reports(audit_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Produces every report category Checkpoint 3 requires (Section 8)."""
    reports: dict[str, pd.DataFrame] = {}

    reports["non_singleton_families"] = (
        audit_df[audit_df["family_size"] > 1]
        .sort_values(["family_size", "family_id"], ascending=[False, True])
    )
    reports["singleton_families"] = audit_df[audit_df["family_size"] == 1]

    lang_diversity = audit_df.groupby("family_id")["detected_language"].apply(
        lambda s: s.dropna().nunique()
    )
    has_non_english_member = audit_df.groupby("family_id")["language_status"].apply(
        lambda s: (s == "confidently_non_english").any() and (s == "confidently_english").any()
    )
    cross_language_family_ids = sorted(
        set(lang_diversity[lang_diversity > 1].index) | set(has_non_english_member[has_non_english_member].index)
    )
    reports["cross_language_families"] = audit_df[audit_df["family_id"].isin(cross_language_family_ids)]

    doctype_diversity = audit_df.groupby("family_id")["document_type"].apply(
        lambda s: set(s) & _KNOWN_DOCTYPES
    )
    form_bundle_family_ids = [
        fam for fam, types in doctype_diversity.items()
        if "form" in types and len(types & {"instructions", "supplement", "checklist"}) > 0
    ]
    reports["form_plus_instructions_or_supplement_families"] = audit_df[
        audit_df["family_id"].isin(form_bundle_family_ids)
    ]

    reports["ambiguous_manual_review"] = audit_df[
        audit_df["review_status"].isin(["needs_review", "resolved_manual_review"])
    ]

    reports["cross_agency_code_conflicts"] = find_cross_agency_code_conflicts(audit_df)
    reports["conflicting_agency_families"] = find_conflicting_agency_families(audit_df)

    uncertain_family_ids = audit_df.loc[
        audit_df["language_status"] == "uncertain_review", "family_id"
    ].unique()
    reports["families_with_uncertain_language_records"] = audit_df[
        audit_df["family_id"].isin(uncertain_family_ids)
    ]

    return reports


def build_override_proposals(audit_df: pd.DataFrame) -> list[OverrideProposal]:
    """Confident, evidence-backed override proposals only. Ambiguous cases (none remain
    after this revision -- all seven previously-ambiguous documents now have a confident,
    evidence-backed finding) go through build_manual_review_flags() instead."""
    proposals = []

    for document_id, finding in _MANUAL_FINDINGS.items():
        if finding["confidence"] != "confident_override":
            continue
        matches = audit_df[audit_df["document_id"] == document_id]
        if matches.empty:
            continue
        row = matches.iloc[0]

        field_changes = []
        if finding["agency_override"]:
            field_changes.append(
                OverrideFieldChange(field="agency", before=row["agency"], after=finding["agency_override"])
            )
        if finding["family_override"] or row["family_id"] != row["effective_family_id"]:
            field_changes.append(
                OverrideFieldChange(
                    field="effective_family_id",
                    before=row["family_id"],
                    after=row["effective_family_id"],
                )
            )
        field_changes.append(
            OverrideFieldChange(
                field="modeling_eligibility",
                before=row["recommended_modeling_eligibility"],
                after=finding["final_modeling_eligibility"],
            )
        )

        proposals.append(
            OverrideProposal(
                document_id=document_id,
                field_changes=field_changes,
                evidence=finding["evidence"] + [finding["manual_language_notes"]],
            )
        )
    return proposals


def build_manual_review_flags(audit_df: pd.DataFrame) -> list[ManualReviewFlag]:
    """Anything still genuinely unresolved after this revision's manual inspection. Empty
    unless a future record turns up evidence that doesn't resolve cleanly."""
    flags = []
    still_uncertain = audit_df[
        (audit_df["language_status"] == "uncertain_review")
        & (~audit_df["document_id"].isin(_MANUAL_FINDINGS))
    ]
    for _, row in still_uncertain.iterrows():
        flags.append(
            ManualReviewFlag(
                document_id=row["document_id"],
                reason="Uncertain-language record not yet manually inspected.",
                evidence=[f"filename={row['filename']!r}", f"family_id={row['family_id']!r}"],
            )
        )
    return flags


def build_family_audit_manifest(
    audit_df: pd.DataFrame, source_df: pd.DataFrame, settings: Settings
) -> FamilyAuditManifest:
    family_first = audit_df.drop_duplicates("family_id")
    singleton_count = int((family_first["family_size"] == 1).sum())
    non_singleton_count = int((family_first["family_size"] > 1).sum())

    doctype_diversity = audit_df.groupby("family_id")["document_type"].apply(
        lambda s: set(s) & _KNOWN_DOCTYPES
    )
    form_bundle_count = sum(
        1 for types in doctype_diversity
        if "form" in types and len(types & {"instructions", "supplement", "checklist"}) > 0
    )

    lang_diversity = audit_df.groupby("family_id")["detected_language"].apply(
        lambda s: s.dropna().nunique()
    )
    cross_language_count = int((lang_diversity > 1).sum())

    exact_dupes = find_exact_duplicate_candidates(source_df, settings)
    near_dupes = find_near_duplicate_candidates(source_df, settings)
    cross_agency_conflicts = find_cross_agency_code_conflicts(audit_df)
    ambiguous_count = int(
        audit_df["review_status"].isin(["needs_review", "resolved_manual_review"]).sum()
    )

    return FamilyAuditManifest(
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        source_dataset_fingerprint=dataset_fingerprint(source_df, settings),
        total_documents=len(source_df),
        total_families=int(family_first["family_id"].nunique()),
        singleton_family_count=singleton_count,
        non_singleton_family_count=non_singleton_count,
        families_by_agency=family_first.groupby("agency")["family_id"].count().to_dict(),
        documents_by_agency=audit_df.groupby("agency")["document_id"].count().to_dict(),
        form_instruction_or_supplement_family_count=form_bundle_count,
        cross_language_family_count=cross_language_count,
        duplicate_candidate_count=len(exact_dupes) + len(near_dupes),
        cross_agency_conflict_count=len(cross_agency_conflicts),
        ambiguous_review_count=ambiguous_count,
        notes=[
            "Family assignment is agency-scoped: a family_id can never span two agency "
            "labels by construction (see conflicting_agency_families report, expected "
            "empty). Cross-agency code collisions are reported separately as "
            "cross_agency_code_conflicts and never auto-merged.",
            "Near-duplicate TF-IDF candidates mostly reflect shared boilerplate language "
            "between DIFFERENT forms in the same agency, not the same form -- they are a "
            "diagnostic signal for manual review, not evidence used for family assignment.",
            "family_id is the original, source-label-derived grouping; effective_family_id "
            "is what actually governs leakage-safe splitting once approved overrides are "
            "applied. modeling_eligibility is decided per document, never forced identical "
            "across a family -- an English form and its translation may share a family "
            "while only the English document enters the modeling corpus.",
        ],
    )


def save_family_audit(
    audit_df: pd.DataFrame,
    category_reports: dict[str, pd.DataFrame],
    manifest: FamilyAuditManifest,
    override_proposals: list[OverrideProposal],
    manual_review_flags: list[ManualReviewFlag],
    exact_duplicates: pd.DataFrame,
    near_duplicates: pd.DataFrame,
    settings: Settings,
    override_version: str = "v2",
    supersedes_version: str | None = "v1",
) -> Path:
    reports_dir = settings.resolve_path("artifacts/family_aware/reports")
    manifests_dir = settings.resolve_path("artifacts/family_aware/manifests")
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    audit_df.to_csv(reports_dir / "family_audit_v1.csv", index=False)

    for name, report_df in category_reports.items():
        report_df.to_csv(reports_dir / f"family_report_{name}.csv", index=False)

    exact_duplicates.to_csv(reports_dir / "family_duplicate_candidates_exact.csv", index=False)
    near_duplicates.to_csv(reports_dir / "family_duplicate_candidates_near.csv", index=False)

    with open(manifests_dir / "family_audit_manifest_v1.json", "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))

    # Overrides files are versioned and never overwritten in place: each call to this
    # function targets a NEW version filename, so earlier versions stay on disk unchanged
    # for auditability (see docs/BLUEPRINT.md and the Checkpoint 3 revision report).
    overrides_payload = {
        "version": override_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "supersedes": (
            f"family_overrides_{supersedes_version}.json ({supersedes_version} preserved "
            "unchanged for auditability -- see this file's evidence for what changed)"
            if supersedes_version
            else None
        ),
        "overrides": [proposal.model_dump() for proposal in override_proposals],
        "manual_review_flags": [flag.model_dump() for flag in manual_review_flags],
    }
    overrides_path = manifests_dir / f"family_overrides_{override_version}.json"
    if overrides_path.exists():
        raise FileExistsError(
            f"{overrides_path} already exists -- versioned override files are never "
            "overwritten. Pass a new override_version to save_family_audit()."
        )
    with open(overrides_path, "w", encoding="utf-8") as f:
        json.dump(overrides_payload, f, indent=2, ensure_ascii=False)

    return reports_dir
