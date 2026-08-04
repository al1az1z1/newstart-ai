"""Family-aware, agency-representative train/validation/test split for the Version 6
corrected experiment (Robustness_v6_Family_Aware_Chunked_BERT.md, Checkpoint 4).

Grouping key is `effective_family_id` (never the original `family_id`, and never an
individual `document_id`) -- every eligible member of an effective family is assigned to the
same split. Family integrity and agency coverage are prioritized over hitting the configured
70/15/15 ratios exactly, per the checkpoint's explicit requirement.

Nothing here modifies `final_dataset.csv`, historical splits, or historical artifacts.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from newstart_ai.config.settings import Settings
from newstart_ai.data.fingerprinting import dataset_fingerprint
from newstart_ai.schemas.family_split import (
    FamilyAwareSplitManifest,
    SplitAgencyCounts,
    SplitCounts,
)

SPLIT_NAMES = ("train", "validation", "test")


def build_eligible_corpus(audit_df: pd.DataFrame, source_df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Selects final_modeling_eligibility == include_english_corpus documents and joins
    back the full document text from source_df. Never modifies source_df or audit_df."""
    ds_cfg = settings.base.dataset
    text_lookup = source_df.set_index(source_df[ds_cfg.id_column].astype(str))[ds_cfg.text_column]

    eligible = audit_df[audit_df["final_modeling_eligibility"] == "include_english_corpus"].copy()
    eligible["text"] = eligible["document_id"].map(text_lookup)
    return eligible.reset_index(drop=True)


def _seeded_sort_key(family_id: str, seed: int) -> str:
    """A deterministic, seed-dependent tie-break for families of equal size -- not
    alphabetical (which would systematically favor certain family_id prefixes) and not
    Python's unseeded `random` (which isn't reproducible across environments/versions)."""
    return hashlib.sha256(f"{seed}:{family_id}".encode("utf-8")).hexdigest()


def assign_families_to_splits(eligible_df: pd.DataFrame, settings: Settings) -> dict[str, str]:
    """Returns {effective_family_id: split_name}, deterministic given the configured seed.

    Per effective_agency, independently:
      1. Coverage guarantee -- when an agency has >=3 (or >=2) distinct families, its
         largest available family is assigned to test, then the largest remaining to
         validation, so every agency appears in every split. Every real family in this
         dataset is well under 15% of its own agency's total, so this never risks
         overshooting a split's target on its own (checked, not assumed).
      2. Deficit-greedy -- every remaining family (largest first, seeded deterministic
         tie-break) is assigned to whichever split currently has the largest remaining gap
         toward its document-count target. A family is never divided to fix an imbalance.
    """
    split_cfg = settings.family_aware.split
    ratios = {"train": split_cfg.train, "validation": split_cfg.validation, "test": split_cfg.test}
    seed = split_cfg.random_seed

    family_sizes = (
        eligible_df.groupby(["effective_agency", "effective_family_id"])
        .size()
        .reset_index(name="doc_count")
    )

    assignment: dict[str, str] = {}

    for agency in sorted(family_sizes["effective_agency"].unique()):
        agency_families = family_sizes[family_sizes["effective_agency"] == agency].copy()
        agency_families["sort_key"] = agency_families["effective_family_id"].apply(
            lambda fid: _seeded_sort_key(fid, seed)
        )
        agency_families = agency_families.sort_values(["doc_count", "sort_key"], ascending=[False, True])
        remaining = list(agency_families.itertuples(index=False))

        total_docs = int(agency_families["doc_count"].sum())
        targets = {split: total_docs * ratios[split] for split in SPLIT_NAMES}
        assigned_counts = {split: 0 for split in SPLIT_NAMES}

        if len(remaining) >= 3:
            test_family = remaining.pop(0)
            assignment[test_family.effective_family_id] = "test"
            assigned_counts["test"] += test_family.doc_count

            validation_family = remaining.pop(0)
            assignment[validation_family.effective_family_id] = "validation"
            assigned_counts["validation"] += validation_family.doc_count
        elif len(remaining) == 2:
            # Only enough distinct families for one guaranteed non-train split; documented
            # in the split report as a coverage gap rather than silently ignored.
            test_family = remaining.pop(0)
            assignment[test_family.effective_family_id] = "test"
            assigned_counts["test"] += test_family.doc_count

        for family in remaining:
            deficits = {split: targets[split] - assigned_counts[split] for split in SPLIT_NAMES}
            best_split = max(deficits, key=lambda split: deficits[split])
            assignment[family.effective_family_id] = best_split
            assigned_counts[best_split] += family.doc_count

    return assignment


def create_family_aware_split(
    eligible_df: pd.DataFrame, settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Splits eligible_df by effective_family_id. Returns (train_df, validation_df,
    test_df, family_to_split_map)."""
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


# --- Invariant proofs -------------------------------------------------------------------


def assert_no_document_overlap(train_df, val_df, test_df) -> None:
    ids = [set(df["document_id"]) for df in (train_df, val_df, test_df)]
    overlaps = {
        "train/validation": ids[0] & ids[1],
        "train/test": ids[0] & ids[2],
        "validation/test": ids[1] & ids[2],
    }
    leaking = {name: found for name, found in overlaps.items() if found}
    if leaking:
        raise ValueError(f"Document overlap detected across splits: {leaking}")


def assert_no_family_overlap(train_df, val_df, test_df) -> None:
    fams = [set(df["effective_family_id"]) for df in (train_df, val_df, test_df)]
    overlaps = {
        "train/validation": fams[0] & fams[1],
        "train/test": fams[0] & fams[2],
        "validation/test": fams[1] & fams[2],
    }
    leaking = {name: found for name, found in overlaps.items() if found}
    if leaking:
        raise ValueError(f"Effective family overlap detected across splits: {leaking}")


def assert_every_eligible_document_assigned_exactly_once(eligible_df, train_df, val_df, test_df) -> None:
    expected = set(eligible_df["document_id"])
    assigned = list(train_df["document_id"]) + list(val_df["document_id"]) + list(test_df["document_id"])
    assigned_set = set(assigned)
    if len(assigned) != len(assigned_set):
        raise ValueError("A document_id appears more than once across the splits combined.")
    if assigned_set != expected:
        raise ValueError(
            f"Split coverage mismatch: {len(expected - assigned_set)} eligible documents "
            f"missing, {len(assigned_set - expected)} unexpected documents present."
        )


def assert_no_excluded_document_in_splits(audit_df, train_df, val_df, test_df) -> None:
    excluded_ids = set(
        audit_df.loc[audit_df["final_modeling_eligibility"] != "include_english_corpus", "document_id"]
    )
    in_splits = set(train_df["document_id"]) | set(val_df["document_id"]) | set(test_df["document_id"])
    leaking = excluded_ids & in_splits
    if leaking:
        raise ValueError(f"Excluded/unresolved documents found inside a split: {leaking}")


def find_agencies_missing_by_split(train_df, val_df, test_df, all_agencies: list[str]) -> dict[str, list[str]]:
    missing = {}
    for name, df in (("train", train_df), ("validation", val_df), ("test", test_df)):
        present = set(df["effective_agency"])
        gap = sorted(set(all_agencies) - present)
        if gap:
            missing[name] = gap
    return missing


# --- Fingerprints ------------------------------------------------------------------------


def _fingerprint_records(df: pd.DataFrame, columns: list[str]) -> str:
    ordered = df[columns].astype(str).sort_values(columns).reset_index(drop=True)
    payload = "\n".join("|".join(row) for row in ordered.itertuples(index=False))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprint_eligibility_manifest(audit_df: pd.DataFrame) -> str:
    """Fingerprints the columns that determine eligibility and grouping -- if this changes,
    the split must be regenerated."""
    columns = ["document_id", "effective_agency", "effective_family_id", "final_modeling_eligibility"]
    return _fingerprint_records(audit_df, columns)


def fingerprint_file(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fingerprint_split(df: pd.DataFrame) -> str:
    return _fingerprint_records(df, ["document_id", "effective_agency", "effective_family_id"])


# --- Report + save -------------------------------------------------------------------------


def build_split_report(
    eligible_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    source_df: pd.DataFrame,
    override_artifact_path: Path,
    override_artifact_version: str,
    settings: Settings,
) -> FamilyAwareSplitManifest:
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
        by_agency = [
            SplitAgencyCounts(
                agency=agency,
                document_count=int((df["effective_agency"] == agency).sum()),
                family_count=int(df.loc[df["effective_agency"] == agency, "effective_family_id"].nunique()),
            )
            for agency in all_agencies
        ]
        splits_report.append(
            SplitCounts(
                split=name,
                document_count=len(df),
                family_count=int(df["effective_family_id"].nunique()),
                percentage_of_eligible_documents=round(100 * len(df) / total_docs, 2) if total_docs else 0.0,
                by_agency=by_agency,
            )
        )
        split_fingerprints[name] = fingerprint_split(df)

    split_cfg = settings.family_aware.split
    return FamilyAwareSplitManifest(
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        random_seed=split_cfg.random_seed,
        configured_ratios={"train": split_cfg.train, "validation": split_cfg.validation, "test": split_cfg.test},
        source_dataset_fingerprint=dataset_fingerprint(source_df, settings),
        eligibility_manifest_fingerprint=fingerprint_eligibility_manifest(audit_df),
        override_artifact_fingerprint=fingerprint_file(override_artifact_path),
        override_artifact_version=override_artifact_version,
        total_eligible_documents=total_docs,
        total_eligible_families=int(eligible_df["effective_family_id"].nunique()),
        splits=splits_report,
        split_fingerprints=split_fingerprints,
        all_agencies_in_every_split=len(agencies_missing) == 0,
        agencies_missing_by_split=agencies_missing,
        zero_document_overlap=True,
        zero_family_overlap=True,
        every_eligible_document_assigned_exactly_once=True,
        no_excluded_or_unresolved_document_in_any_split=True,
        notes=[
            "Grouping key is effective_family_id (post-override), never the original "
            "family_id and never an individual document_id.",
            "Family integrity and agency coverage were prioritized over exact 70/15/15 "
            "percentages, per Checkpoint 4's explicit instruction.",
            "All four assertion functions in family_split.py were called before this "
            "manifest was built and raise on failure -- the boolean fields above reflect "
            "checks that actually ran, not assumptions.",
        ],
    )


def save_family_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    report: FamilyAwareSplitManifest,
    settings: Settings,
) -> Path:
    output_dir = settings.resolve_path(settings.family_aware.split.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    with open(output_dir / "family_split_manifest_v1.json", "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))

    return output_dir
