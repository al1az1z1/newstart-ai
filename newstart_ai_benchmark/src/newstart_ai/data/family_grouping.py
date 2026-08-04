"""Deterministic document-family discovery for the Version 6 family-aware robustness
research (Robustness_v6_Family_Aware_Chunked_BERT.md, Checkpoint 3).

A "family" is the indivisible future split unit: a main form, its instructions,
supplements, and translated/revised versions must all end up in the same train, validation,
or test partition. Every document keeps its own independent `document_id` -- text is never
merged or nested; `family_id` is grouping metadata only, never model input.

Evidence priority, most to least reliable:
  1. Normalized `form_number` (present for IRS 100%, USCIS 99%, SSA 88%, DMV 0%).
  2. A form-code pattern parsed from the filename, used only when form_number is missing.
  3. No evidence -> a stable singleton family (never grouped by agency/topic alone).

This module is deliberately conservative: when evidence is weak or ambiguous, it prefers a
singleton (or a flag for manual review) over a false merge, per the project's explicit
requirement not to group documents on agency/topic similarity alone.
"""

from __future__ import annotations

import hashlib
import re
import urllib.parse

import pandas as pd

# Full words and short filename abbreviations for languages actually observed in this
# dataset's non-English documents (Checkpoint 2's language audit) plus their common
# filename shorthands. Used to strip translation markers before comparing form codes, so a
# translated document's code matches its English sibling's code.
LANGUAGE_SUFFIX_TOKENS = {
    "sp", "es", "fr", "vi", "ru", "so", "tur", "hc", "ch", "pt", "psh", "ar", "dar",
    "km", "th", "sw", "ht", "ja", "hi", "ko", "tl", "uk", "pl", "el", "hy", "fa", "zh", "kr",
    "spanish", "chinese", "vietnamese", "russian", "somali", "turkish", "thai",
    "portuguese", "arabic", "dari", "khmer", "swahili", "haitian", "korean",
    "tagalog", "creole", "farsi", "armenian", "greek", "polish", "ukrainian",
}

# Only short (2-3 char) codes are used for character-level stripping on concatenated
# (no-hyphen) filenames -- full words are handled by the token-level stripper above.
_SHORT_LANGUAGE_SUFFIXES = sorted(
    (tok for tok in LANGUAGE_SUFFIX_TOKENS if len(tok) <= 3), key=len, reverse=True
)

# Document-type / revision markers that are never part of a form's own identifying code.
DOCTYPE_SUFFIX_TOKENS = {
    "instr", "instruction", "instructions", "ws", "worksheet", "sup", "supa", "supb",
    "supc", "sup1", "sup2", "sup3", "supplement", "checklist", "form", "rev", "revised",
    "translated", "watermark", "pdf",
}

_ALPHA_TOKEN_RE = re.compile(r"^[a-z]{1,6}$")


def normalize_code(value: str) -> str:
    """Uppercase, letters+digits only -- the shared canonical form for both form-number-
    derived and filename-derived codes, so e.g. "I-589" and a filename-derived "i589" match."""
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def normalize_filename_stem(filename: str) -> list[str]:
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    stem = urllib.parse.unquote(stem)
    stem = stem.lower()
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
    """Extracts a form-code candidate from a filename, or None if there isn't enough
    evidence to derive one confidently.

    Two patterns are supported, both requiring the code to sit at the trailing (end)
    position -- a mid-filename word that merely looks like a code is never extracted, to
    avoid false-positive grouping from descriptive titles:

    1. Hyphenated filenames (DMV's convention, e.g. "...-reg-6004.pdf"): the last
       (short alpha token, digit-leading token) pair.
    2. Concatenated filenames with no internal hyphen (IRS/SSA convention, e.g.
       "fss4.pdf", "fw4v.pdf"): the whole stem, after stripping a trailing language marker
       and an optional single leading "f"/"i" document-type marker.
    """
    tokens = normalize_filename_stem(filename)
    tokens = _strip_trailing_noise_tokens(tokens)
    if not tokens:
        return None

    if len(tokens) >= 2:
        for i in range(len(tokens) - 1, 0, -1):
            alpha_tok, digit_tok = tokens[i - 1], tokens[i]
            if _ALPHA_TOKEN_RE.match(alpha_tok) and digit_tok[:1].isdigit():
                return normalize_code(alpha_tok + digit_tok)

    joined = "".join(tokens)
    joined = _strip_trailing_language_chars(joined)
    if not any(ch.isdigit() for ch in joined):
        return None  # no digit evidence anywhere -- too generic to trust
    if len(joined) > 2 and joined[0] in ("f", "i") and joined[1].isalpha():
        joined = joined[1:]  # strip a single leading form/instructions marker letter
    return normalize_code(joined)


class FamilyAssignment:
    __slots__ = ("family_key", "evidence_type", "evidence_detail", "confidence")

    def __init__(self, family_key: str, evidence_type: str, evidence_detail: str, confidence: float):
        self.family_key = family_key
        self.evidence_type = evidence_type
        self.evidence_detail = evidence_detail
        self.confidence = confidence


def assign_family(document_id: str, agency: str, filename: str, form_number: str | None) -> FamilyAssignment:
    """Assigns one document's family key, scoped by its own (current) agency label so a
    coincidental code collision can never silently merge two different agencies' documents.
    Cross-agency code collisions are detected and reported separately, never auto-merged."""
    if form_number and str(form_number).strip() and str(form_number).lower() != "nan":
        code = normalize_code(str(form_number))
        if code:
            return FamilyAssignment(
                family_key=f"{agency}:{code}",
                evidence_type="form_number_exact",
                evidence_detail=f"form_number={form_number!r}",
                confidence=1.0,
            )

    code = derive_filename_code(filename)
    if code:
        return FamilyAssignment(
            family_key=f"{agency}:{code}",
            evidence_type="filename_code_match",
            evidence_detail=f"filename={filename!r} -> code={code}",
            confidence=0.75,
        )

    return FamilyAssignment(
        family_key=f"SINGLETON:{agency}:{document_id}",
        evidence_type="singleton_no_evidence",
        evidence_detail="no form_number and no confident filename code",
        confidence=1.0,
    )


def build_family_assignments(df: pd.DataFrame, settings) -> pd.DataFrame:
    """Runs assign_family() over every row, then computes family_size per resulting group.
    Never modifies df."""
    ds_cfg = settings.base.dataset
    rows = []
    for _, record in df.iterrows():
        form_number = record.get("form_number")
        form_number = None if pd.isna(form_number) else str(form_number)
        assignment = assign_family(
            document_id=str(record[ds_cfg.id_column]),
            agency=record[ds_cfg.label_column],
            filename=str(record.get("filename", "")),
            form_number=form_number,
        )
        rows.append(
            {
                "document_id": str(record[ds_cfg.id_column]),
                "agency": record[ds_cfg.label_column],
                "filename": record.get("filename", ""),
                "form_number": form_number,
                "document_type": record.get("document_type", ""),
                "family_id": assignment.family_key,
                "evidence_type": assignment.evidence_type,
                "evidence_detail": assignment.evidence_detail,
                "confidence": assignment.confidence,
            }
        )
    result = pd.DataFrame(rows)
    family_sizes = result.groupby("family_id")["document_id"].transform("count")
    result["family_size"] = family_sizes
    return result


def find_cross_agency_code_conflicts(assignments: pd.DataFrame) -> pd.DataFrame:
    """Detects the same normalized code appearing under more than one agency label --
    reported for manual review, never auto-merged across agencies.

    Only applies to non-singleton evidence (form_number_exact / filename_code_match), since
    singleton keys already embed the document_id and can never collide.
    """
    coded = assignments[assignments["evidence_type"] != "singleton_no_evidence"].copy()
    coded["code"] = coded["family_id"].str.split(":", n=1).str[1]

    conflicts = []
    for code, group in coded.groupby("code"):
        agencies = sorted(group["agency"].unique())
        if len(agencies) > 1:
            conflicts.append(
                {
                    "code": code,
                    "agencies": agencies,
                    "document_ids": sorted(group["document_id"].tolist()),
                }
            )
    return pd.DataFrame(conflicts)


def normalized_text_hash(text: str) -> str:
    """A whitespace/punctuation-insensitive content hash, used to catch exact or
    near-exact duplicate extractions (e.g. the same PDF crawled twice)."""
    normalized = re.sub(r"\s+", " ", str(text).strip().lower())
    normalized = re.sub(r"[^\w\s]", "", normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def find_exact_duplicate_candidates(df: pd.DataFrame, settings) -> pd.DataFrame:
    """Pairs of documents whose text is identical after whitespace/punctuation
    normalization -- regardless of agency or existing family, since a true duplicate
    extraction could in principle span either."""
    ds_cfg = settings.base.dataset
    hashes = df[ds_cfg.text_column].apply(normalized_text_hash)
    ids = df[ds_cfg.id_column].astype(str)

    rows = []
    for _, group in pd.DataFrame({"id": ids, "hash": hashes}).groupby("hash"):
        if len(group) > 1:
            group_ids = sorted(group["id"].tolist())
            for i in range(len(group_ids)):
                for j in range(i + 1, len(group_ids)):
                    rows.append(
                        {
                            "document_id_a": group_ids[i],
                            "document_id_b": group_ids[j],
                            "similarity": 1.0,
                            "method": "exact_normalized_text_hash",
                        }
                    )
    return pd.DataFrame(rows, columns=["document_id_a", "document_id_b", "similarity", "method"])


def find_near_duplicate_candidates(
    df: pd.DataFrame, settings, threshold: float = 0.92, max_features: int = 20000
) -> pd.DataFrame:
    """TF-IDF cosine similarity between every pair of documents, reporting pairs at or
    above `threshold`. This is a diagnostic content-similarity signal, independent of (and
    reported alongside, not merged into) family assignment -- 754 documents makes an O(n^2)
    similarity matrix computationally trivial."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    ds_cfg = settings.base.dataset
    texts = df[ds_cfg.text_column].fillna("").astype(str).tolist()
    ids = df[ds_cfg.id_column].astype(str).tolist()

    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(matrix)

    rows = []
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            similarity = float(similarity_matrix[i, j])
            if similarity >= threshold:
                rows.append(
                    {
                        "document_id_a": ids[i],
                        "document_id_b": ids[j],
                        "similarity": similarity,
                        "method": "tfidf_cosine",
                    }
                )
    return pd.DataFrame(rows, columns=["document_id_a", "document_id_b", "similarity", "method"])


def find_conflicting_agency_families(assignments: pd.DataFrame) -> pd.DataFrame:
    """Sanity check: a single family_id (already agency-scoped) should never contain more
    than one distinct agency value. This should always be empty by construction -- it exists
    as an explicit, tested invariant rather than an assumption."""
    conflicts = []
    for family_id, group in assignments.groupby("family_id"):
        agencies = group["agency"].unique()
        if len(agencies) > 1:
            conflicts.append(
                {
                    "family_id": family_id,
                    "agencies": sorted(agencies.tolist()),
                    "document_ids": sorted(group["document_id"].tolist()),
                }
            )
    return pd.DataFrame(conflicts)
