"""Narrow, reproducible identifier masking (Version 6, Checkpoint 6).

Targets only explicit routing shortcuts: agency names/unambiguous abbreviations, form
numbers referenced as "Form <code>", OMB control numbers, and .gov URLs. Every rule is a
literal-phrase or narrowly-scoped regex authored from public knowledge of these four
agencies' real letterhead/footer/URL conventions -- never fitted by inspecting any
document's text (train, validation, or test), so it cannot leak test information and applies
identically regardless of which split or agency a document belongs to.

Ordinary semantic content (dates, addresses, unrelated numbers, generic phrases like "social
security number") is deliberately left untouched -- masking only ever replaces a matched
identifier with a stable placeholder token, never deletes or rewrites surrounding text.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from newstart_ai.schemas.checkpoint6 import (
    MaskingAuditExample,
    MaskingManifest,
    MaskingRuleMatchSummary,
    MaskingSplitSummary,
)


@dataclass
class MaskingRule:
    name: str
    pattern: re.Pattern
    replacement: object  # str or callable(match) -> str


def build_masking_rules(config) -> list[MaskingRule]:
    """Rule order matters and was corrected after a manual audit finding (real document 692,
    an SSA form): applied in the order below to avoid one rule partially consuming text that
    a later rule needs intact.

    1. `.gov` URLs first -- an agency abbreviation is frequently a literal substring of its
       own domain (e.g. "irs" inside "www.irs.gov"); masking the URL first prevents that
       substring from being separately (and redundantly) consumed by the agency-identifier
       rule, which would otherwise also corrupt the URL text so the URL rule could no longer
       recognize it on a later pass (its placeholder contains "[" "]" "_", not valid hostname
       characters).
    2. Form numbers second -- several real form codes ARE the agency abbreviation plus a
       number (e.g. "Form SSA-714", "Form SSA-1560"). If the agency-name rule ran first, it
       would consume just the "SSA" token and leave the trailing "-714" digits exposed and
       unmasked (found via manual audit of document 692). Masking the whole "Form <code>"
       span first avoids this partial-consumption gap.
    3. Agency names/abbreviations third, for any standalone mention not already covered.
    4. OMB numbers last (no interaction with the other three).
    """
    rules: list[MaskingRule] = []

    url_pattern = re.compile(r"\b(?:https?://)?(?:www\.)?[\w-]+(?:\.[\w-]+)*\.gov(?:/[\w\-./?%&=]*)?", re.IGNORECASE)
    rules.append(MaskingRule(name="agency_url", pattern=url_pattern, replacement=config.url_placeholder))

    form_pattern = re.compile(r"\bForm\s+([A-Za-z]{1,5}-?\d{1,5}[A-Za-z]{0,3})\b", re.IGNORECASE)
    rules.append(
        MaskingRule(
            name="form_number",
            pattern=form_pattern,
            replacement=lambda m: f"{m.group(0).split()[0]} {config.form_number_placeholder}",
        )
    )

    for agency, phrases in config.agency_identifier_phrases.items():
        ordered_phrases = sorted(phrases, key=len, reverse=True)
        alternation = "|".join(re.escape(p) for p in ordered_phrases)
        pattern = re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)
        rules.append(MaskingRule(name=f"agency_identifier:{agency}", pattern=pattern, replacement=config.agency_name_placeholder))

    omb_pattern = re.compile(r"\bOMB\s*(?:No\.?|Number|#)?\s*\d{4}-\d{4}\b", re.IGNORECASE)
    rules.append(MaskingRule(name="omb_number", pattern=omb_pattern, replacement=config.omb_number_placeholder))

    return rules


def apply_masking(text: str, rules: list[MaskingRule]) -> tuple[str, dict[str, int]]:
    """Applies every rule once, in order, to `text`. Returns (masked_text, {rule_name: count})."""
    masked = str(text)
    counts: dict[str, int] = {}
    for rule in rules:
        masked, n = rule.pattern.subn(rule.replacement, masked)
        counts[rule.name] = n
    return masked, counts


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def mask_document(document_id: str, text: str, rules: list[MaskingRule], policy_version: str) -> dict:
    original_hash = _sha256(str(text))
    masked_text, counts = apply_masking(text, rules)
    return {
        "document_id": str(document_id),
        "original_text_hash": original_hash,
        "masked_text": masked_text,
        "masked_text_hash": _sha256(masked_text),
        "rule_match_counts": counts,
        "total_replacements": sum(counts.values()),
        "policy_version": policy_version,
    }


def build_masked_documents(split_df: pd.DataFrame, split_name: str, settings) -> pd.DataFrame:
    """Builds the masked derivative of every document's full original text in one frozen
    split DataFrame. Never modifies `split_df` -- returns a new DataFrame only."""
    cfg = settings.family_aware.masking
    rules = build_masking_rules(cfg)

    rows = []
    for row in split_df.itertuples(index=False):
        record = mask_document(row.document_id, row.text, rules, cfg.policy_version)
        record["effective_family_id"] = row.effective_family_id
        record["agency"] = row.agency
        record["effective_agency"] = row.effective_agency
        record["split"] = split_name
        rows.append(record)
    return pd.DataFrame(rows)


def build_masked_chunks(chunks_df: pd.DataFrame, settings) -> pd.DataFrame:
    """Builds the masked derivative of every chunk's decoded `chunk_text`, keyed by
    `chunk_id` -- keeps identical chunk_index/token_start/token_end/split provenance so
    partial-input selection (which only ever refers to chunk_index) applies identically to
    masked and unmasked text. Never modifies `chunks_df` -- returns a new DataFrame."""
    cfg = settings.family_aware.masking
    rules = build_masking_rules(cfg)

    rows = []
    for row in chunks_df.itertuples(index=False):
        masked_text, counts = apply_masking(row.chunk_text, rules)
        rows.append(
            {
                "chunk_id": row.chunk_id,
                "document_id": row.document_id,
                "chunk_index": row.chunk_index,
                "total_chunks": row.total_chunks,
                "token_start": row.token_start,
                "token_end": row.token_end,
                "split": row.split,
                "masked_chunk_text": masked_text,
                "masked_chunk_text_hash": _sha256(masked_text),
                "rule_match_counts": counts,
                "total_replacements": sum(counts.values()),
                "policy_version": cfg.policy_version,
            }
        )
    return pd.DataFrame(rows)


def build_masking_manifest(masked_documents_by_split: dict[str, pd.DataFrame], audit_examples: list[dict], settings) -> MaskingManifest:
    cfg = settings.family_aware.masking
    rules = build_masking_rules(cfg)
    rule_names = [r.name for r in rules]

    all_docs = pd.concat(masked_documents_by_split.values(), ignore_index=True)

    rule_summaries = []
    for name in rule_names:
        counts = all_docs["rule_match_counts"].apply(lambda d: d.get(name, 0))
        rule_summaries.append(
            MaskingRuleMatchSummary(
                rule_name=name,
                total_matches=int(counts.sum()),
                documents_with_at_least_one_match=int((counts > 0).sum()),
            )
        )

    split_summaries = []
    for split_name, df in masked_documents_by_split.items():
        split_summaries.append(
            MaskingSplitSummary(
                split=split_name,
                document_count=int(len(df)),
                total_replacements=int(df["total_replacements"].sum()),
                documents_with_zero_matches=int((df["total_replacements"] == 0).sum()),
            )
        )

    return MaskingManifest(
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        policy_version=cfg.policy_version,
        rule_names=rule_names,
        rule_match_summary=rule_summaries,
        per_split_summary=split_summaries,
        audit_examples=[MaskingAuditExample(**example) for example in audit_examples],
        ground_truth_label_unchanged=True,
        fitted_without_examining_test_outcomes=True,
        applies_identically_across_methods=True,
        notes=[
            "Rules target explicit routing shortcuts only (agency names/abbreviations, "
            "\"Form <code>\" identifiers, OMB control numbers, .gov URLs) -- ordinary "
            "semantic content is never touched.",
            "The same rule set and code path apply to every document regardless of split "
            "or effective_agency, so BERT/LLM/LLM+RAG evaluation inputs are masked "
            "identically.",
        ],
    )
