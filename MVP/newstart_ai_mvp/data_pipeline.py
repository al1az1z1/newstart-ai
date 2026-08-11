"""Self-contained data-preparation pipeline: dataset validation, language filtering, family
grouping/audit, the family-aware split, tokenizer-aware chunking, identifier masking,
partial-input selection, and the shared condition registry.

This is a copy of the original project's research logic (Checkpoints 2-6), reorganized into
one module and simplified where the original used elaborate Pydantic manifest schemas --
manifests here are plain dicts, written straight to JSON. The actual computational
methodology (family-assignment regex evidence, the seeded deficit-greedy split algorithm,
the sliding-window token-chunking math, the masking regex rules, and the condition
definitions) is unchanged from the original.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SPLIT_NAMES = ("train", "validation", "test")


def _sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


# =========================================================================================
# Fingerprinting
# =========================================================================================


def dataset_fingerprint(df: pd.DataFrame, settings) -> str:
    """Hashes (id, label, text) for every row, sorted by id -- same content, same
    fingerprint regardless of row order."""
    ds_cfg = settings.base.dataset
    id_col, text_col, label_col = ds_cfg.id_column, ds_cfg.text_column, ds_cfg.label_column
    ordered = df[[id_col, label_col, text_col]].astype(str).sort_values(id_col)
    hasher = hashlib.sha256()
    for _, row in ordered.iterrows():
        hasher.update(row[id_col].encode("utf-8") + b"|" + row[label_col].encode("utf-8") + b"|" + row[text_col].encode("utf-8") + b"\n")
    return hasher.hexdigest()


def fingerprint_records(df: pd.DataFrame, columns: list[str]) -> str:
    ordered = df[columns].astype(str).sort_values(columns).reset_index(drop=True)
    payload = "\n".join("|".join(row) for row in ordered.itertuples(index=False))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprint_split(df: pd.DataFrame) -> str:
    return fingerprint_records(df, ["document_id", "effective_agency", "effective_family_id"])


def fingerprint_chunks(chunks_df: pd.DataFrame) -> str:
    columns = ["chunk_id", "document_id", "effective_family_id", "token_start", "token_end", "chunk_text_hash"]
    return fingerprint_records(chunks_df, columns)


def fingerprint_file(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# =========================================================================================
# Stage: dataset validation (non-destructive; reports only, never cleans or drops rows)
# =========================================================================================

MIN_ROWS_PER_CLASS_FOR_SPLIT = 5
IMBALANCE_WARNING_RATIO = 4.0


def load_dataset(settings) -> pd.DataFrame:
    path = settings.resolve_path(settings.base.dataset.path)
    return pd.read_csv(path, encoding_errors="replace")


def validate_dataset(df: pd.DataFrame, settings) -> dict:
    ds_cfg = settings.base.dataset
    id_col, text_col, label_col = ds_cfg.id_column, ds_cfg.text_column, ds_cfg.label_column
    allowed_labels = set(settings.base.labels)

    row_count = len(df)
    id_series = df[id_col].astype(str)
    duplicate_document_id_count = int(id_series.duplicated().sum())
    text_series = df[text_col].fillna("").astype(str)
    empty_text_count = int((text_series.str.strip() == "").sum())
    label_series = df[label_col].astype(str)
    invalid_label_values = sorted(label_series[~label_series.isin(allowed_labels)].unique().tolist())
    counts = label_series.value_counts()
    imbalance_ratio = float(counts.max() / counts.min()) if len(counts) and counts.min() > 0 else 0.0

    warnings = []
    if duplicate_document_id_count:
        warnings.append(f"{duplicate_document_id_count} duplicate '{id_col}' values found.")
    if empty_text_count:
        warnings.append(f"{empty_text_count} rows have empty or whitespace-only text.")
    if invalid_label_values:
        warnings.append(f"Found label values outside {sorted(allowed_labels)}: {invalid_label_values}")
    if imbalance_ratio >= IMBALANCE_WARNING_RATIO:
        warnings.append(f"Class imbalance ratio is {imbalance_ratio:.1f}x -- class-weighted loss will apply.")

    return {
        "row_count": row_count,
        "document_id_column_unique": duplicate_document_id_count == 0,
        "duplicate_document_id_count": duplicate_document_id_count,
        "empty_text_count": empty_text_count,
        "valid_labels": not invalid_label_values,
        "invalid_label_values": invalid_label_values,
        "class_counts": counts.to_dict(),
        "imbalance_ratio": imbalance_ratio,
        "has_critical_errors": duplicate_document_id_count > 0 or bool(invalid_label_values),
        "warnings": warnings,
    }


# =========================================================================================
# Stage: language filtering (Checkpoint 2)
# =========================================================================================

_ALPHA_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def get_language_identifier():
    import py3langid.langid as langid_module

    return langid_module.LanguageIdentifier.from_pickled_model(langid_module.MODEL_FILE, norm_probs=True)


def _alphabetic_ratio(text: str) -> float:
    non_space = re.sub(r"\s+", "", text)
    return len(_ALPHA_RE.findall(non_space)) / len(non_space) if non_space else 0.0


def classify_document_language(text: str, identifier, config) -> tuple[str | None, float | None, str, str]:
    stripped = (text or "").strip()
    if not stripped:
        return None, None, "uncertain_review", "empty_text"
    if len(stripped) < config.min_text_length:
        return None, None, "uncertain_review", "text_too_short_for_reliable_detection"
    if _alphabetic_ratio(stripped) < config.min_alphabetic_ratio:
        return None, None, "uncertain_review", "insufficient_alphabetic_content"

    if config.mixed_language_window_count >= 2:
        window_count = config.mixed_language_window_count
        window_size = len(stripped) // window_count
        windows = [stripped[i * window_size: (i + 1) * window_size if i < window_count - 1 else None] for i in range(window_count)]
        if all(len(w) >= config.min_text_length for w in windows):
            window_langs = {identifier.classify(w)[0] for w in windows}
            if len(window_langs) > 1:
                whole_lang, whole_confidence = identifier.classify(stripped)
                return whole_lang, float(whole_confidence), "uncertain_review", "mixed_language_signal_across_document"

    detected_language, confidence = identifier.classify(stripped)
    confidence = float(confidence)
    if detected_language == config.target_language and confidence >= config.confident_english_threshold:
        return detected_language, confidence, "confidently_english", "detector_high_confidence"
    if detected_language != config.target_language and confidence >= config.confident_non_english_threshold:
        return detected_language, confidence, "confidently_non_english", "detector_high_confidence"
    return detected_language, confidence, "uncertain_review", "low_detection_confidence"


def build_language_audit(df: pd.DataFrame, settings) -> pd.DataFrame:
    config = settings.family_aware.language_filter
    identifier = get_language_identifier()
    ds_cfg = settings.base.dataset

    rows = []
    for _, record in df.iterrows():
        text = record[ds_cfg.text_column]
        lang, confidence, status, reason = classify_document_language(text, identifier, config)
        form_number = record.get("form_number")
        rows.append({
            "document_id": str(record[ds_cfg.id_column]),
            "agency": record[ds_cfg.label_column],
            "filename": str(record.get("filename", "")),
            "form_number": None if pd.isna(form_number) else str(form_number),
            "text_length": int(record.get("text_length") or len(str(text))),
            "detected_language": lang,
            "confidence": confidence,
            "status": status,
            "reason": reason,
        })
    return pd.DataFrame(rows)


def build_language_filter_manifest(audit_df: pd.DataFrame, source_df: pd.DataFrame, settings) -> dict:
    config = settings.family_aware.language_filter
    return {
        "version": "v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "detector_name": config.detector_name,
        "detector_config": config.model_dump(),
        "source_dataset_fingerprint": dataset_fingerprint(source_df, settings),
        "total_documents": len(source_df),
        "counts_by_status": audit_df["status"].value_counts().to_dict(),
        "notes": ["uncertain_review documents are retained pending manual review, never deleted."],
    }


def save_language_audit(audit_df: pd.DataFrame, manifest: dict, settings) -> tuple[Path, Path]:
    reports_dir = settings.resolve_path("artifacts/family_aware/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    audit_path = reports_dir / "language_audit_v1.csv"
    manifest_path = reports_dir / "language_audit_manifest_v1.json"
    audit_df.to_csv(audit_path, index=False)
    _write_json(manifest_path, manifest)
    return audit_path, manifest_path


# =========================================================================================
# Stage: family grouping (Checkpoint 3)
# =========================================================================================

LANGUAGE_SUFFIX_TOKENS = {
    "sp", "es", "fr", "vi", "ru", "so", "tur", "hc", "ch", "pt", "psh", "ar", "dar",
    "km", "th", "sw", "ht", "ja", "hi", "ko", "tl", "uk", "pl", "el", "hy", "fa", "zh", "kr",
    "spanish", "chinese", "vietnamese", "russian", "somali", "turkish", "thai",
    "portuguese", "arabic", "dari", "khmer", "swahili", "haitian", "korean",
    "tagalog", "creole", "farsi", "armenian", "greek", "polish", "ukrainian",
}
_SHORT_LANGUAGE_SUFFIXES = sorted((tok for tok in LANGUAGE_SUFFIX_TOKENS if len(tok) <= 3), key=len, reverse=True)
DOCTYPE_SUFFIX_TOKENS = {
    "instr", "instruction", "instructions", "ws", "worksheet", "sup", "supa", "supb",
    "supc", "sup1", "sup2", "sup3", "supplement", "checklist", "form", "rev", "revised",
    "translated", "watermark", "pdf",
}
_ALPHA_TOKEN_RE = re.compile(r"^[a-z]{1,6}$")
_KNOWN_DOCTYPES = {"form", "instructions", "supplement", "checklist", "translated_form"}


def normalize_code(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def normalize_filename_stem(filename: str) -> list[str]:
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    stem = urllib.parse.unquote(stem).lower()
    stem = re.sub(r"[\s_]+", "-", stem)
    stem = re.sub(r"-+", "-", stem).strip("-")
    return [t for t in stem.split("-") if t]


def _strip_trailing_noise_tokens(tokens: list[str]) -> list[str]:
    tokens = list(tokens)
    while tokens and (tokens[-1] in DOCTYPE_SUFFIX_TOKENS or tokens[-1] in LANGUAGE_SUFFIX_TOKENS):
        tokens.pop()
    return tokens


def _strip_trailing_language_chars(joined: str) -> str:
    changed = True
    while changed:
        changed = False
        for suffix in _SHORT_LANGUAGE_SUFFIXES:
            if len(joined) > len(suffix) + 1 and joined.endswith(suffix):
                joined = joined[: -len(suffix)]
                changed = True
                break
    return joined


def derive_filename_code(filename: str) -> str | None:
    tokens = _strip_trailing_noise_tokens(normalize_filename_stem(filename))
    if not tokens:
        return None
    if len(tokens) >= 2:
        for i in range(len(tokens) - 1, 0, -1):
            alpha_tok, digit_tok = tokens[i - 1], tokens[i]
            if _ALPHA_TOKEN_RE.match(alpha_tok) and digit_tok[:1].isdigit():
                return normalize_code(alpha_tok + digit_tok)
    joined = _strip_trailing_language_chars("".join(tokens))
    if not any(ch.isdigit() for ch in joined):
        return None
    if len(joined) > 2 and joined[0] in ("f", "i") and joined[1].isalpha():
        joined = joined[1:]
    return normalize_code(joined)


@dataclass
class FamilyAssignment:
    family_key: str
    evidence_type: str
    evidence_detail: str
    confidence: float


def assign_family(document_id: str, agency: str, filename: str, form_number: str | None) -> FamilyAssignment:
    if form_number and str(form_number).strip() and str(form_number).lower() != "nan":
        code = normalize_code(str(form_number))
        if code:
            return FamilyAssignment(f"{agency}:{code}", "form_number_exact", f"form_number={form_number!r}", 1.0)
    code = derive_filename_code(filename)
    if code:
        return FamilyAssignment(f"{agency}:{code}", "filename_code_match", f"filename={filename!r} -> code={code}", 0.75)
    return FamilyAssignment(f"SINGLETON:{agency}:{document_id}", "singleton_no_evidence", "no form_number and no confident filename code", 1.0)


def build_family_assignments(df: pd.DataFrame, settings) -> pd.DataFrame:
    ds_cfg = settings.base.dataset
    rows = []
    for _, record in df.iterrows():
        form_number = record.get("form_number")
        form_number = None if pd.isna(form_number) else str(form_number)
        a = assign_family(str(record[ds_cfg.id_column]), record[ds_cfg.label_column], str(record.get("filename", "")), form_number)
        rows.append({
            "document_id": str(record[ds_cfg.id_column]), "agency": record[ds_cfg.label_column],
            "filename": record.get("filename", ""), "form_number": form_number,
            "document_type": record.get("document_type", ""), "family_id": a.family_key,
            "evidence_type": a.evidence_type, "evidence_detail": a.evidence_detail, "confidence": a.confidence,
        })
    result = pd.DataFrame(rows)
    result["family_size"] = result.groupby("family_id")["document_id"].transform("count")
    return result


def find_cross_agency_code_conflicts(assignments: pd.DataFrame) -> pd.DataFrame:
    coded = assignments[assignments["evidence_type"] != "singleton_no_evidence"].copy()
    coded["code"] = coded["family_id"].str.split(":", n=1).str[1]
    rows = []
    for code, group in coded.groupby("code"):
        agencies = sorted(group["agency"].unique())
        if len(agencies) > 1:
            rows.append({"code": code, "agencies": agencies, "document_ids": sorted(group["document_id"].tolist())})
    return pd.DataFrame(rows)


def find_conflicting_agency_families(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family_id, group in assignments.groupby("family_id"):
        agencies = group["agency"].unique()
        if len(agencies) > 1:
            rows.append({"family_id": family_id, "agencies": sorted(agencies.tolist()), "document_ids": sorted(group["document_id"].tolist())})
    return pd.DataFrame(rows)


def normalized_text_hash(text: str) -> str:
    normalized = re.sub(r"[^\w\s]", "", re.sub(r"\s+", " ", str(text).strip().lower()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def find_exact_duplicate_candidates(df: pd.DataFrame, settings) -> pd.DataFrame:
    ds_cfg = settings.base.dataset
    hashes = df[ds_cfg.text_column].apply(normalized_text_hash)
    ids = df[ds_cfg.id_column].astype(str)
    rows = []
    for _, group in pd.DataFrame({"id": ids, "hash": hashes}).groupby("hash"):
        if len(group) > 1:
            group_ids = sorted(group["id"].tolist())
            for i in range(len(group_ids)):
                for j in range(i + 1, len(group_ids)):
                    rows.append({"document_id_a": group_ids[i], "document_id_b": group_ids[j], "similarity": 1.0, "method": "exact_normalized_text_hash"})
    return pd.DataFrame(rows, columns=["document_id_a", "document_id_b", "similarity", "method"])


def find_near_duplicate_candidates(df: pd.DataFrame, settings, threshold: float = 0.92, max_features: int = 20000) -> pd.DataFrame:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    ds_cfg = settings.base.dataset
    texts = df[ds_cfg.text_column].fillna("").astype(str).tolist()
    ids = df[ds_cfg.id_column].astype(str).tolist()
    matrix = TfidfVectorizer(max_features=max_features, stop_words="english").fit_transform(texts)
    sim = cosine_similarity(matrix)
    rows = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            s = float(sim[i, j])
            if s >= threshold:
                rows.append({"document_id_a": ids[i], "document_id_b": ids[j], "similarity": s, "method": "tfidf_cosine"})
    return pd.DataFrame(rows, columns=["document_id_a", "document_id_b", "similarity", "method"])


# =========================================================================================
# Stage: family audit (Checkpoint 3) -- manual findings, effective_agency/family_id
# =========================================================================================

# Manually inspected findings from the original project's Checkpoint 3 revision (direct
# reading of each document's substantive text). Reproduced verbatim -- these are real,
# evidence-backed research decisions, not something this refactor invented.
_MANUAL_FINDINGS: dict[str, dict] = {
    "540": {"agency_override": "IRS", "family_override": None, "final_modeling_eligibility": "include_english_corpus",
            "manual_language_notes": "Direct inspection: substantive text is English (SS-4 Application for EIN, Dept. of the Treasury / IRS)."},
    "541": {"agency_override": "IRS", "family_override": None, "final_modeling_eligibility": "exclude_non_english",
            "manual_language_notes": "Direct inspection: Spanish translation of SS-4 (same OMB number as document 540)."},
    "542": {"agency_override": "IRS", "family_override": None, "final_modeling_eligibility": "include_english_corpus",
            "manual_language_notes": "Direct inspection: IRS Form W-4V, Voluntary Withholding Request -- OMB 1545-0074 (IRS series)."},
    "131": {"agency_override": None, "family_override": "USCIS:I9", "final_modeling_eligibility": "exclude_non_english",
            "manual_language_notes": "Direct inspection: Spanish translation of I-9 instructions; corrected family from an automated filename-code miss."},
    "210": {"agency_override": None, "family_override": None, "final_modeling_eligibility": "exclude_non_english",
            "manual_language_notes": "Direct inspection confirms Pashto-script content throughout."},
    "261": {"agency_override": None, "family_override": None, "final_modeling_eligibility": "exclude_insufficient_text",
            "manual_language_notes": "Direct inspection: text is entirely repeated form-field labels, no substantive prose."},
    "361": {"agency_override": None, "family_override": None, "final_modeling_eligibility": "exclude_non_english",
            "manual_language_notes": "Direct inspection confirms Khmer-script content throughout."},
    "397": {"agency_override": None, "family_override": None, "final_modeling_eligibility": "include_english_corpus",
            "manual_language_notes": "Direct inspection: substantive English content present; garbled middle section is an extraction artifact, not a language issue."},
    "634": {"agency_override": None, "family_override": None, "final_modeling_eligibility": "include_english_corpus",
            "manual_language_notes": "Direct inspection: clear English text; low alphabetic ratio was caused by blank-line/checkbox fields, not non-English content."},
}


def build_full_family_audit(df: pd.DataFrame, language_audit_df: pd.DataFrame, settings) -> pd.DataFrame:
    assignments = build_family_assignments(df, settings)
    merged = assignments.merge(
        language_audit_df[["document_id", "status", "detected_language"]], on="document_id", how="left"
    ).rename(columns={"status": "language_status"})

    def recommended_modeling_eligibility(row):
        if row["language_status"] == "confidently_non_english":
            return "exclude_non_english"
        if row["language_status"] == "uncertain_review":
            return "pending_review"
        return "include_english_corpus"

    merged["recommended_modeling_eligibility"] = merged.apply(recommended_modeling_eligibility, axis=1)
    merged["agency_override_proposed"] = None
    merged["final_modeling_eligibility"] = merged["recommended_modeling_eligibility"]

    for document_id, finding in _MANUAL_FINDINGS.items():
        mask = merged["document_id"] == document_id
        if not mask.any():
            continue
        merged.loc[mask, "agency_override_proposed"] = finding["agency_override"]
        merged.loc[mask, "final_modeling_eligibility"] = finding["final_modeling_eligibility"]

    merged["effective_agency"] = merged["agency_override_proposed"].fillna(merged["agency"])

    def effective_family_id(row):
        finding = _MANUAL_FINDINGS.get(row["document_id"])
        if finding and finding["family_override"]:
            return finding["family_override"]
        if row["effective_agency"] != row["agency"]:
            return assign_family(row["document_id"], row["effective_agency"], row["filename"], row["form_number"]).family_key
        return row["family_id"]

    merged["effective_family_id"] = merged.apply(effective_family_id, axis=1)
    return merged


def build_category_reports(audit_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    reports: dict[str, pd.DataFrame] = {
        "non_singleton_families": audit_df[audit_df["family_size"] > 1].sort_values(["family_size", "family_id"], ascending=[False, True]),
        "singleton_families": audit_df[audit_df["family_size"] == 1],
        "cross_agency_code_conflicts": find_cross_agency_code_conflicts(audit_df),
        "conflicting_agency_families": find_conflicting_agency_families(audit_df),
    }
    return reports


def build_family_audit_manifest(audit_df: pd.DataFrame, source_df: pd.DataFrame, settings) -> dict:
    family_first = audit_df.drop_duplicates("family_id")
    exact_dupes = find_exact_duplicate_candidates(source_df, settings)
    near_dupes = find_near_duplicate_candidates(source_df, settings)
    return {
        "version": "v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset_fingerprint": dataset_fingerprint(source_df, settings),
        "total_documents": len(source_df),
        "total_families": int(family_first["family_id"].nunique()),
        "singleton_family_count": int((family_first["family_size"] == 1).sum()),
        "non_singleton_family_count": int((family_first["family_size"] > 1).sum()),
        "documents_by_agency": audit_df.groupby("agency")["document_id"].count().to_dict(),
        "duplicate_candidate_count": len(exact_dupes) + len(near_dupes),
        "notes": ["Family assignment is agency-scoped: a family_id never spans two agency labels by construction."],
    }


def save_family_audit(
    audit_df: pd.DataFrame, category_reports: dict[str, pd.DataFrame], manifest: dict, settings,
    override_version: str = "v2", supersedes_version: str | None = "v1",
) -> Path:
    reports_dir = settings.resolve_path("artifacts/family_aware/reports")
    manifests_dir = settings.resolve_path("artifacts/family_aware/manifests")
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    audit_df.to_csv(reports_dir / "family_audit_v1.csv", index=False)
    for name, report_df in category_reports.items():
        report_df.to_csv(reports_dir / f"family_report_{name}.csv", index=False)

    _write_json(manifests_dir / "family_audit_manifest_v1.json", manifest)

    overrides_path = manifests_dir / f"family_overrides_{override_version}.json"
    if overrides_path.exists():
        raise FileExistsError(f"{overrides_path} already exists -- versioned override files are never overwritten.")
    overrides_payload = {
        "version": override_version, "created_at": datetime.now(timezone.utc).isoformat(),
        "supersedes": f"family_overrides_{supersedes_version}.json" if supersedes_version else None,
        "manual_findings": _MANUAL_FINDINGS,
    }
    _write_json(overrides_path, overrides_payload)
    return reports_dir


# =========================================================================================
# Stage: family-aware split (Checkpoint 4)
# =========================================================================================


def build_eligible_corpus(audit_df: pd.DataFrame, source_df: pd.DataFrame, settings) -> pd.DataFrame:
    ds_cfg = settings.base.dataset
    text_lookup = source_df.set_index(source_df[ds_cfg.id_column].astype(str))[ds_cfg.text_column]
    eligible = audit_df[audit_df["final_modeling_eligibility"] == "include_english_corpus"].copy()
    eligible["text"] = eligible["document_id"].map(text_lookup)
    return eligible.reset_index(drop=True)


def _seeded_sort_key(family_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{family_id}".encode("utf-8")).hexdigest()


def assign_families_to_splits(eligible_df: pd.DataFrame, settings) -> dict[str, str]:
    split_cfg = settings.family_aware.split
    ratios = {"train": split_cfg.train, "validation": split_cfg.validation, "test": split_cfg.test}
    seed = split_cfg.random_seed

    family_sizes = eligible_df.groupby(["effective_agency", "effective_family_id"]).size().reset_index(name="doc_count")
    assignment: dict[str, str] = {}

    for agency in sorted(family_sizes["effective_agency"].unique()):
        agency_families = family_sizes[family_sizes["effective_agency"] == agency].copy()
        agency_families["sort_key"] = agency_families["effective_family_id"].apply(lambda fid: _seeded_sort_key(fid, seed))
        agency_families = agency_families.sort_values(["doc_count", "sort_key"], ascending=[False, True])
        remaining = list(agency_families.itertuples(index=False))

        total_docs = int(agency_families["doc_count"].sum())
        targets = {s: total_docs * ratios[s] for s in SPLIT_NAMES}
        assigned_counts = {s: 0 for s in SPLIT_NAMES}

        if len(remaining) >= 3:
            for split_name in ("test", "validation"):
                fam = remaining.pop(0)
                assignment[fam.effective_family_id] = split_name
                assigned_counts[split_name] += fam.doc_count
        elif len(remaining) == 2:
            fam = remaining.pop(0)
            assignment[fam.effective_family_id] = "test"
            assigned_counts["test"] += fam.doc_count

        for family in remaining:
            deficits = {s: targets[s] - assigned_counts[s] for s in SPLIT_NAMES}
            best_split = max(deficits, key=lambda s: deficits[s])
            assignment[family.effective_family_id] = best_split
            assigned_counts[best_split] += family.doc_count

    return assignment


def create_family_aware_split(eligible_df: pd.DataFrame, settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    family_to_split = assign_families_to_splits(eligible_df, settings)
    working = eligible_df.copy()
    working["_split"] = working["effective_family_id"].map(family_to_split)
    if working["_split"].isna().any():
        unmapped = working.loc[working["_split"].isna(), "effective_family_id"].unique()
        raise RuntimeError(f"No split assignment produced for families: {unmapped.tolist()}")
    train_df = working[working["_split"] == "train"].drop(columns=["_split"]).reset_index(drop=True)
    val_df = working[working["_split"] == "validation"].drop(columns=["_split"]).reset_index(drop=True)
    test_df = working[working["_split"] == "test"].drop(columns=["_split"]).reset_index(drop=True)
    return train_df, val_df, test_df, family_to_split


def assert_no_document_overlap(train_df, val_df, test_df) -> None:
    ids = [set(df["document_id"]) for df in (train_df, val_df, test_df)]
    leaking = {n: v for n, v in zip(("train/validation", "train/test", "validation/test"), (ids[0] & ids[1], ids[0] & ids[2], ids[1] & ids[2])) if v}
    if leaking:
        raise ValueError(f"Document overlap detected across splits: {leaking}")


def assert_no_family_overlap(train_df, val_df, test_df) -> None:
    fams = [set(df["effective_family_id"]) for df in (train_df, val_df, test_df)]
    leaking = {n: v for n, v in zip(("train/validation", "train/test", "validation/test"), (fams[0] & fams[1], fams[0] & fams[2], fams[1] & fams[2])) if v}
    if leaking:
        raise ValueError(f"Effective family overlap detected across splits: {leaking}")


def assert_every_eligible_document_assigned_exactly_once(eligible_df, train_df, val_df, test_df) -> None:
    expected = set(eligible_df["document_id"])
    assigned = list(train_df["document_id"]) + list(val_df["document_id"]) + list(test_df["document_id"])
    if len(assigned) != len(set(assigned)) or set(assigned) != expected:
        raise ValueError("Split coverage mismatch: a document is missing, duplicated, or unexpected.")


def assert_no_excluded_document_in_splits(audit_df, train_df, val_df, test_df) -> None:
    excluded_ids = set(audit_df.loc[audit_df["final_modeling_eligibility"] != "include_english_corpus", "document_id"])
    in_splits = set(train_df["document_id"]) | set(val_df["document_id"]) | set(test_df["document_id"])
    leaking = excluded_ids & in_splits
    if leaking:
        raise ValueError(f"Excluded/unresolved documents found inside a split: {leaking}")


def find_agencies_missing_by_split(train_df, val_df, test_df, all_agencies: list[str]) -> dict[str, list[str]]:
    missing = {}
    for name, df in (("train", train_df), ("validation", val_df), ("test", test_df)):
        gap = sorted(set(all_agencies) - set(df["effective_agency"]))
        if gap:
            missing[name] = gap
    return missing


def fingerprint_eligibility_manifest(audit_df: pd.DataFrame) -> str:
    return fingerprint_records(audit_df, ["document_id", "effective_agency", "effective_family_id", "final_modeling_eligibility"])


def build_split_report(eligible_df, train_df, val_df, test_df, audit_df, source_df, override_artifact_path, override_artifact_version, settings) -> dict:
    assert_no_document_overlap(train_df, val_df, test_df)
    assert_no_family_overlap(train_df, val_df, test_df)
    assert_every_eligible_document_assigned_exactly_once(eligible_df, train_df, val_df, test_df)
    assert_no_excluded_document_in_splits(audit_df, train_df, val_df, test_df)

    all_agencies = sorted(eligible_df["effective_agency"].unique())
    agencies_missing = find_agencies_missing_by_split(train_df, val_df, test_df, all_agencies)
    total_docs = len(eligible_df)

    splits_report = []
    split_fingerprints = {}
    for name, df in (("train", train_df), ("validation", val_df), ("test", test_df)):
        splits_report.append({
            "split": name, "document_count": len(df), "family_count": int(df["effective_family_id"].nunique()),
            "percentage_of_eligible_documents": round(100 * len(df) / total_docs, 2) if total_docs else 0.0,
            "by_agency": {a: int((df["effective_agency"] == a).sum()) for a in all_agencies},
        })
        split_fingerprints[name] = fingerprint_split(df)

    split_cfg = settings.family_aware.split
    return {
        "version": "v1", "created_at": datetime.now(timezone.utc).isoformat(), "random_seed": split_cfg.random_seed,
        "source_dataset_fingerprint": dataset_fingerprint(source_df, settings),
        "eligibility_manifest_fingerprint": fingerprint_eligibility_manifest(audit_df),
        "override_artifact_fingerprint": fingerprint_file(override_artifact_path),
        "override_artifact_version": override_artifact_version,
        "total_eligible_documents": total_docs, "total_eligible_families": int(eligible_df["effective_family_id"].nunique()),
        "splits": splits_report, "split_fingerprints": split_fingerprints,
        "all_agencies_in_every_split": len(agencies_missing) == 0, "agencies_missing_by_split": agencies_missing,
        "zero_document_overlap": True, "zero_family_overlap": True,
        "every_eligible_document_assigned_exactly_once": True, "no_excluded_or_unresolved_document_in_any_split": True,
        "notes": ["Grouping key is effective_family_id -- family integrity and agency coverage were prioritized over exact 70/15/15 percentages."],
    }


def save_family_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, report: dict, settings) -> Path:
    output_dir = settings.resolve_path(settings.family_aware.split.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    _write_json(output_dir / "family_split_manifest_v1.json", report)
    return output_dir


# =========================================================================================
# Stage: tokenizer-aware chunking (Checkpoint 5)
# =========================================================================================


def get_bert_tokenizer(settings):
    from transformers import AutoTokenizer

    cfg = settings.family_aware.chunking
    return AutoTokenizer.from_pretrained(settings.bert.base_model, revision=cfg.tokenizer_revision)


def compute_chunk_token_ranges(total_tokens: int, window: int, step: int) -> list[tuple[int, int]]:
    """Deterministic sliding-window ranges. A document within one window produces exactly
    one range; longer documents slide by `step` tokens, with the final range snapped back
    to end exactly at total_tokens so the tail is always retained in full, never truncated."""
    if total_tokens <= 0:
        return []
    if total_tokens <= window:
        return [(0, total_tokens)]
    if step <= 0:
        raise ValueError("chunk_overlap_tokens must be smaller than the content window")
    ranges: list[tuple[int, int]] = []
    start = 0
    while True:
        end = start + window
        if end >= total_tokens:
            end = total_tokens
            start = end - window
            if ranges and ranges[-1] == (start, end):
                break
            ranges.append((start, end))
            break
        ranges.append((start, end))
        start += step
    return ranges


def build_chunk_rows_for_document(document_id, text, effective_family_id, agency, effective_agency, split, tokenizer, tokenizer_name, cfg) -> list[dict]:
    parent_text_hash = _sha256(str(text))
    token_ids = tokenizer.encode(str(text), add_special_tokens=False)
    window = cfg.max_seq_length - cfg.num_special_tokens
    step = window - cfg.chunk_overlap_tokens
    ranges = compute_chunk_token_ranges(len(token_ids), window, step)
    total_chunks = len(ranges)

    rows = []
    for chunk_index, (start, end) in enumerate(ranges):
        chunk_token_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(chunk_token_ids, skip_special_tokens=True)
        rows.append({
            "chunk_id": _sha256(f"{cfg.chunking_policy_version}|{document_id}|{chunk_index}|{start}|{end}"),
            "document_id": str(document_id), "effective_family_id": effective_family_id, "agency": agency,
            "effective_agency": effective_agency, "split": split, "chunk_index": chunk_index, "total_chunks": total_chunks,
            "tokenizer_name": tokenizer_name, "tokenizer_revision": cfg.tokenizer_revision,
            "token_start": start, "token_end": end, "content_token_count": len(chunk_token_ids),
            "encoded_sequence_length": len(chunk_token_ids) + cfg.num_special_tokens,
            "chunk_text": chunk_text, "chunk_text_hash": _sha256(chunk_text),
            "parent_text_hash": parent_text_hash, "chunking_policy_version": cfg.chunking_policy_version,
        })
    return rows


def build_chunks_for_split(split_df: pd.DataFrame, split_name: str, tokenizer, tokenizer_name: str, settings) -> pd.DataFrame:
    cfg = settings.family_aware.chunking
    all_rows: list[dict] = []
    for row in split_df.itertuples(index=False):
        rows = build_chunk_rows_for_document(str(row.document_id), row.text, row.effective_family_id, row.agency, row.effective_agency, split_name, tokenizer, tokenizer_name, cfg)
        if not rows:
            raise RuntimeError(f"document_id={row.document_id!r} produced zero chunks.")
        all_rows.extend(rows)
    return pd.DataFrame(all_rows)


def build_all_split_chunks(train_df, val_df, test_df, settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tokenizer = get_bert_tokenizer(settings)
    tokenizer_name = settings.bert.base_model
    return (
        build_chunks_for_split(train_df, "train", tokenizer, tokenizer_name, settings),
        build_chunks_for_split(val_df, "validation", tokenizer, tokenizer_name, settings),
        build_chunks_for_split(test_df, "test", tokenizer, tokenizer_name, settings),
    )


def assert_every_chunk_maps_to_one_eligible_parent(chunks_df, eligible_df) -> None:
    unmapped = set(chunks_df["document_id"].astype(str)) - set(eligible_df["document_id"].astype(str))
    if unmapped:
        raise ValueError(f"Chunks exist for documents outside the eligible corpus: {unmapped}")


def assert_every_chunk_inherits_parent_split(chunks_df, document_to_split) -> None:
    mismatches = [(d, s) for d, s in zip(chunks_df["document_id"], chunks_df["split"]) if document_to_split.get(str(d)) != s]
    if mismatches:
        raise ValueError(f"Chunk split does not match parent document's split: {mismatches[:5]}")


def assert_no_cross_split_leakage(train_chunks, val_chunks, test_chunks) -> None:
    for column in ("document_id", "effective_family_id", "chunk_id"):
        sets = [set(df[column]) for df in (train_chunks, val_chunks, test_chunks)]
        leaking = {n: v for n, v in zip(("train/validation", "train/test", "validation/test"), (sets[0] & sets[1], sets[0] & sets[2], sets[1] & sets[2])) if v}
        if leaking:
            raise ValueError(f"{column} crosses splits: {leaking}")


def assert_no_excluded_document_chunked(chunks_df, audit_df) -> None:
    excluded_ids = set(audit_df.loc[audit_df["final_modeling_eligibility"] != "include_english_corpus", "document_id"].astype(str))
    leaking = excluded_ids & set(chunks_df["document_id"].astype(str))
    if leaking:
        raise ValueError(f"Excluded documents produced chunks: {leaking}")


def assert_no_duplicate_chunk_ids(chunks_df) -> None:
    dup = chunks_df["chunk_id"][chunks_df["chunk_id"].duplicated()]
    if not dup.empty:
        raise ValueError(f"Duplicate chunk_id values found: {dup.tolist()[:5]}")


def assert_chunk_indices_contiguous_and_unique(chunks_df) -> None:
    for doc_id, group in chunks_df.groupby("document_id"):
        indices = sorted(group["chunk_index"].tolist())
        if indices != list(range(len(indices))):
            raise ValueError(f"document_id={doc_id!r} has non-contiguous chunk indices: {indices}")


def assert_no_empty_chunks(chunks_df) -> None:
    empty = chunks_df[chunks_df["content_token_count"] <= 0]
    if not empty.empty:
        raise ValueError(f"Empty chunks found: {empty['chunk_id'].tolist()}")


def assert_every_eligible_document_has_at_least_one_chunk(eligible_df, chunks_df) -> None:
    missing = set(eligible_df["document_id"].astype(str)) - set(chunks_df["document_id"].astype(str))
    if missing:
        raise ValueError(f"Eligible documents with zero chunks: {missing}")


def build_chunk_report(eligible_df, train_df, val_df, test_df, train_chunks, val_chunks, test_chunks, audit_df, split_fingerprints, settings) -> dict:
    all_chunks = pd.concat([train_chunks, val_chunks, test_chunks], ignore_index=True)
    document_to_split = {str(d): s for s, df in (("train", train_df), ("validation", val_df), ("test", test_df)) for d in df["document_id"]}

    assert_every_chunk_maps_to_one_eligible_parent(all_chunks, eligible_df)
    assert_every_chunk_inherits_parent_split(all_chunks, document_to_split)
    assert_no_cross_split_leakage(train_chunks, val_chunks, test_chunks)
    assert_no_excluded_document_chunked(all_chunks, audit_df)
    assert_no_duplicate_chunk_ids(all_chunks)
    assert_chunk_indices_contiguous_and_unique(all_chunks)
    assert_no_empty_chunks(all_chunks)
    assert_every_eligible_document_has_at_least_one_chunk(eligible_df, all_chunks)

    cfg = settings.family_aware.chunking
    return {
        "version": "v1", "created_at": datetime.now(timezone.utc).isoformat(), "chunking_policy_version": cfg.chunking_policy_version,
        "tokenizer_name": settings.bert.base_model, "tokenizer_revision": cfg.tokenizer_revision,
        "max_seq_length": cfg.max_seq_length, "num_special_tokens": cfg.num_special_tokens,
        "chunk_overlap_tokens": cfg.chunk_overlap_tokens,
        "source_split_fingerprints": split_fingerprints,
        "chunk_fingerprints": {"train": fingerprint_chunks(train_chunks), "validation": fingerprint_chunks(val_chunks), "test": fingerprint_chunks(test_chunks)},
        "splits": [
            {"split": n, "document_count": int(df["document_id"].nunique()), "chunk_count": len(df)}
            for n, df in (("train", train_chunks), ("validation", val_chunks), ("test", test_chunks))
        ],
        "all_invariants_verified": True,
        "notes": ["All chunking invariant assertions were called before this manifest was built and raise on failure."],
    }


def save_family_aware_chunks(train_chunks: pd.DataFrame, val_chunks: pd.DataFrame, test_chunks: pd.DataFrame, report: dict, settings) -> Path:
    output_dir = settings.resolve_path(settings.family_aware.chunking.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_chunks.to_csv(output_dir / "train_chunks.csv", index=False)
    val_chunks.to_csv(output_dir / "validation_chunks.csv", index=False)
    test_chunks.to_csv(output_dir / "test_chunks.csv", index=False)
    _write_json(settings.resolve_path("artifacts/family_aware/manifests") / "chunk_manifest_v1.json", report)
    return output_dir


# =========================================================================================
# Stage: identifier masking (Checkpoint 6)
# =========================================================================================


@dataclass
class MaskingRule:
    name: str
    pattern: re.Pattern
    replacement: object


def build_masking_rules(config) -> list[MaskingRule]:
    """Rule order matters (corrected after a manual audit finding, real document 692): .gov
    URLs first, then "Form <code>" spans, then agency names/abbreviations, then OMB numbers
    -- avoids one rule partially consuming text a later rule needs intact."""
    rules: list[MaskingRule] = []
    url_pattern = re.compile(r"\b(?:https?://)?(?:www\.)?[\w-]+(?:\.[\w-]+)*\.gov(?:/[\w\-./?%&=]*)?", re.IGNORECASE)
    rules.append(MaskingRule("agency_url", url_pattern, config.url_placeholder))

    form_pattern = re.compile(r"\bForm\s+([A-Za-z]{1,5}-?\d{1,5}[A-Za-z]{0,3})\b", re.IGNORECASE)
    rules.append(MaskingRule("form_number", form_pattern, lambda m: f"{m.group(0).split()[0]} {config.form_number_placeholder}"))

    for agency, phrases in config.agency_identifier_phrases.items():
        ordered = sorted(phrases, key=len, reverse=True)
        pattern = re.compile(rf"\b(?:{'|'.join(re.escape(p) for p in ordered)})\b", re.IGNORECASE)
        rules.append(MaskingRule(f"agency_identifier:{agency}", pattern, config.agency_name_placeholder))

    omb_pattern = re.compile(r"\bOMB\s*(?:No\.?|Number|#)?\s*\d{4}-\d{4}\b", re.IGNORECASE)
    rules.append(MaskingRule("omb_number", omb_pattern, config.omb_number_placeholder))
    return rules


def apply_masking(text: str, rules: list[MaskingRule]) -> tuple[str, dict[str, int]]:
    masked = str(text)
    counts: dict[str, int] = {}
    for rule in rules:
        masked, n = rule.pattern.subn(rule.replacement, masked)
        counts[rule.name] = n
    return masked, counts


def mask_document(document_id: str, text: str, rules: list[MaskingRule], policy_version: str) -> dict:
    masked_text, counts = apply_masking(text, rules)
    return {
        "document_id": str(document_id), "original_text_hash": _sha256(text), "masked_text": masked_text,
        "masked_text_hash": _sha256(masked_text), "rule_match_counts_json": json.dumps(counts),
        "total_replacements": sum(counts.values()), "policy_version": policy_version,
    }


def build_masked_documents(split_df: pd.DataFrame, split_name: str, settings) -> pd.DataFrame:
    cfg = settings.family_aware.masking
    rules = build_masking_rules(cfg)
    rows = []
    for row in split_df.itertuples(index=False):
        record = mask_document(row.document_id, row.text, rules, cfg.policy_version)
        record.update(effective_family_id=row.effective_family_id, agency=row.agency, effective_agency=row.effective_agency, split=split_name)
        rows.append(record)
    return pd.DataFrame(rows)


def build_masked_chunks(chunks_df: pd.DataFrame, settings) -> pd.DataFrame:
    cfg = settings.family_aware.masking
    rules = build_masking_rules(cfg)
    rows = []
    for row in chunks_df.itertuples(index=False):
        masked_text, counts = apply_masking(row.chunk_text, rules)
        rows.append({
            "chunk_id": row.chunk_id, "document_id": row.document_id, "chunk_index": row.chunk_index,
            "total_chunks": row.total_chunks, "token_start": row.token_start, "token_end": row.token_end,
            "split": row.split, "masked_chunk_text": masked_text, "masked_chunk_text_hash": _sha256(masked_text),
            "rule_match_counts_json": json.dumps(counts), "total_replacements": sum(counts.values()), "policy_version": cfg.policy_version,
        })
    return pd.DataFrame(rows)


def build_masking_manifest(masked_documents_by_split: dict[str, pd.DataFrame], audit_examples: list[dict], settings) -> dict:
    cfg = settings.family_aware.masking
    rule_names = [r.name for r in build_masking_rules(cfg)]
    all_docs = pd.concat(masked_documents_by_split.values(), ignore_index=True)
    return {
        "version": "v1", "created_at": datetime.now(timezone.utc).isoformat(), "policy_version": cfg.policy_version,
        "rule_names": rule_names,
        "per_split_summary": [
            {"split": s, "document_count": len(df), "total_replacements": int(df["total_replacements"].sum()),
             "documents_with_zero_matches": int((df["total_replacements"] == 0).sum())}
            for s, df in masked_documents_by_split.items()
        ],
        "audit_examples": audit_examples,
        "notes": ["Rules target explicit routing shortcuts only -- ordinary semantic content is never touched.",
                   "The same rule set applies to every document regardless of split or agency."],
    }


# =========================================================================================
# Stage: partial-input selection (Checkpoint 6)
# =========================================================================================

CONDITIONS = ("beginning_only", "middle_only", "end_only", "beginning_middle_end")


def _middle_index(total_chunks: int) -> int:
    return total_chunks // 2


def select_partial_chunks(document_id: str, total_chunks: int, condition: str, policy_version: str) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown partial-input condition: {condition!r}")
    if total_chunks <= 0:
        raise ValueError(f"document_id={document_id!r} has total_chunks={total_chunks}")

    beginning_index, end_index, middle_index = 0, total_chunks - 1, _middle_index(total_chunks)
    if condition == "beginning_only":
        requested = [beginning_index]
    elif condition == "middle_only":
        requested = [middle_index]
    elif condition == "end_only":
        requested = [end_index]
    else:
        requested = [beginning_index, middle_index, end_index]

    selected_indices = list(dict.fromkeys(requested))
    fallback_reason = None
    if len(selected_indices) < len(requested):
        fallback_reason = f"document has only {total_chunks} chunk(s); requested regions collapsed to {selected_indices}"

    return {
        "document_id": str(document_id), "condition": condition, "selected_chunk_indices": selected_indices,
        "total_chunks": total_chunks, "fallback_reason": fallback_reason, "policy_version": policy_version,
    }


def build_partial_input_selections(chunks_df: pd.DataFrame, split_name: str, settings) -> pd.DataFrame:
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
    by_index = document_chunks.set_index("chunk_index")[text_column]
    return "\n\n".join(str(by_index.loc[i]) for i in selected_chunk_indices)


def build_partial_input_manifest(selections_df: pd.DataFrame, settings) -> dict:
    cfg = settings.family_aware.partial_input
    fallback_counts = selections_df.assign(has_fallback=selections_df["fallback_reason"].notna()).groupby("condition")["has_fallback"].sum().astype(int).to_dict()
    return {
        "version": "v1", "created_at": datetime.now(timezone.utc).isoformat(), "policy_version": cfg.policy_version,
        "chunks_per_region": cfg.chunks_per_region, "total_documents": int(selections_df["document_id"].nunique()),
        "fallback_document_counts_by_condition": {k: int(v) for k, v in fallback_counts.items()},
        "notes": ["middle_index = total_chunks // 2; for 2-chunk documents this coincides with the end chunk (documented fallback, not an error)."],
    }


# =========================================================================================
# Stage: shared condition registry (Checkpoint 6)
# =========================================================================================

_CONDITION_SPECS: dict[str, tuple[str, bool]] = {
    "complete_unmasked": ("complete", False), "beginning_only_unmasked": ("beginning", False),
    "middle_only_unmasked": ("middle", False), "end_only_unmasked": ("end", False),
    "beginning_middle_end_unmasked": ("beginning_middle_end", False), "complete_masked": ("complete", True),
    "beginning_only_masked": ("beginning", True), "middle_only_masked": ("middle", True),
    "end_only_masked": ("end", True), "beginning_middle_end_masked": ("beginning_middle_end", True),
}
_PARTIAL_TO_SELECTION_CONDITION = {"beginning": "beginning_only", "middle": "middle_only", "end": "end_only", "beginning_middle_end": "beginning_middle_end"}


def build_condition_registry(split_df, masked_documents_df, chunks_df, masked_chunks_df, selections_df, split_name, settings) -> pd.DataFrame:
    policy_version = settings.family_aware.conditions.policy_version
    complete_unmasked_text = split_df.set_index(split_df["document_id"].astype(str))["text"]
    complete_masked_text = masked_documents_df.set_index("document_id")["masked_text"]
    effective_agency = split_df.set_index(split_df["document_id"].astype(str))["effective_agency"]
    unmasked_chunk_text = chunks_df.set_index("chunk_id")["chunk_text"]
    masked_chunk_text = masked_chunks_df.set_index("chunk_id")["masked_chunk_text"]
    chunks_by_document = {doc_id: group for doc_id, group in chunks_df.groupby("document_id")}
    selections_by = {(row.document_id, row.condition): row for row in selections_df.itertuples(index=False)}

    rows = []
    for document_id in complete_unmasked_text.index:
        agency = effective_agency.loc[document_id]
        for condition_name, (region, masked) in _CONDITION_SPECS.items():
            if region == "complete":
                text = complete_masked_text.loc[document_id] if masked else complete_unmasked_text.loc[document_id]
                source_chunk_ids, fallback_reason = [], None
            else:
                selection = selections_by[(document_id, _PARTIAL_TO_SELECTION_CONDITION[region])]
                by_index = chunks_by_document[document_id].set_index("chunk_index")["chunk_id"]
                source_chunk_ids = [str(by_index.loc[i]) for i in selection.selected_chunk_indices]
                text_lookup = masked_chunk_text if masked else unmasked_chunk_text
                text = "\n\n".join(str(text_lookup.loc[cid]) for cid in source_chunk_ids)
                fallback_reason = selection.fallback_reason
            rows.append({
                "document_id": document_id, "split": split_name, "effective_agency": agency, "condition": condition_name,
                "region": region, "masked": masked, "text": text, "text_fingerprint": _sha256(text),
                "source_chunk_ids": ",".join(source_chunk_ids), "fallback_reason": fallback_reason, "policy_version": policy_version,
            })
    return pd.DataFrame(rows)


def build_condition_definitions(policy_version: str) -> list[dict]:
    definitions_text = {
        "complete": "The document's full text (all content, no chunk selection).",
        "beginning": "The single chunk at chunk_index=0.", "middle": "The single chunk at chunk_index = total_chunks // 2.",
        "end": "The single chunk at chunk_index = total_chunks - 1.",
        "beginning_middle_end": "The distinct, deduplicated union of the beginning, middle, and end chunks, concatenated in that order.",
    }
    return [
        {"name": name, "definition": (f"[MASKED] {definitions_text[region]}" if masked else definitions_text[region]),
         "masked": masked, "region": region, "policy_version": policy_version}
        for name, (region, masked) in _CONDITION_SPECS.items()
    ]


def build_condition_registry_manifest(registry_df: pd.DataFrame, settings) -> dict:
    cfg = settings.family_aware.conditions
    ordered = registry_df[["document_id", "split", "condition", "text_fingerprint"]].astype(str).sort_values(["document_id", "split", "condition"])
    payload = "\n".join("|".join(row) for row in ordered.itertuples(index=False))
    return {
        "version": "v1", "created_at": datetime.now(timezone.utc).isoformat(), "policy_version": cfg.policy_version,
        "conditions": build_condition_definitions(cfg.policy_version), "total_documents": int(registry_df["document_id"].nunique()),
        "total_rows": int(len(registry_df)), "per_condition_row_counts": registry_df.groupby("condition").size().to_dict(),
        "registry_fingerprint": _sha256(payload),
        "notes": ["Each (document_id, condition) pair maps to exactly one frozen text string -- every method consuming this registry receives byte-identical text."],
    }
