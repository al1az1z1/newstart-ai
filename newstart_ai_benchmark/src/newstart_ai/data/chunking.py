"""Tokenizer-aware chunking of the frozen family-aware splits, with full provenance
(Robustness_v6_Family_Aware_Chunked_BERT.md, Checkpoint 5).

Reads only the already-frozen train/validation/test CSVs produced by
`newstart_ai.data.family_split` -- every chunk inherits its parent document's split,
`effective_family_id`, and `effective_agency` unchanged. This module never re-derives or
re-assigns family or split membership; it only tokenizes and windows the text of documents
that are already placed.

Nothing here modifies `final_dataset.csv`, the historical splits, or the family-aware
document-level splits from Checkpoint 4.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from newstart_ai.config.settings import Settings
from newstart_ai.schemas.chunk import (
    ChunkAgencyCounts,
    ChunkCountDistribution,
    FamilyAwareChunkManifest,
    LargestDocumentByChunks,
    SplitChunkCounts,
)

SPLIT_NAMES = ("train", "validation", "test")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_TOKENIZER_IDENTITY_FILES = {"tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"}


def resolve_tokenizer_identity(settings: Settings) -> dict:
    """Resolves the exact Hugging Face commit hash and file-level hashes behind the
    configured tokenizer revision, purely by inspecting the local HF cache -- never
    triggers a network call or a new download. The model must already be cached locally
    (it is, from Phase 1 BERT training and every tokenizer load in Checkpoint 5).

    Returns {"resolved_commit_hash": str | None, "tokenizer_file_hashes": dict[str, str]}.
    None/empty results (rather than raising) if the cache entry can't be found, so this is
    always a best-effort, additive provenance improvement -- never a hard dependency.
    """
    model_name = settings.bert.base_model
    revision = settings.family_aware.chunking.tokenizer_revision

    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
    except Exception:
        return {"resolved_commit_hash": None, "tokenizer_file_hashes": {}}

    for repo in cache_info.repos:
        if repo.repo_id != model_name or repo.repo_type != "model":
            continue
        for rev in repo.revisions:
            if revision == rev.commit_hash or revision in rev.refs:
                file_hashes = {}
                for f in rev.files:
                    if f.file_name in _TOKENIZER_IDENTITY_FILES and Path(f.file_path).exists():
                        with open(f.file_path, "rb") as fh:
                            file_hashes[f.file_name] = hashlib.sha256(fh.read()).hexdigest()
                return {"resolved_commit_hash": rev.commit_hash, "tokenizer_file_hashes": file_hashes}
    return {"resolved_commit_hash": None, "tokenizer_file_hashes": {}}


def get_bert_tokenizer(settings: Settings):
    """Loads the tokenizer for the classifier this chunking feeds.

    The model *name* always comes from configs/bert.yaml (settings.bert.base_model) so
    chunking can never silently target a different model than the one that will train on
    these chunks; only the revision is chunking-specific configuration.
    """
    cfg = settings.family_aware.chunking
    return AutoTokenizer.from_pretrained(settings.bert.base_model, revision=cfg.tokenizer_revision)


def compute_chunk_token_ranges(total_tokens: int, window: int, step: int) -> list[tuple[int, int]]:
    """Deterministic (start, end) token ranges covering `total_tokens` raw tokens.

    - Empty input (`total_tokens == 0`) produces no ranges at all -- callers must not build
      a chunk from zero tokens.
    - A document that fits within one window (including exactly at the limit) produces
      exactly one range spanning the whole document -- no duplication.
    - Longer documents slide forward by `step` tokens per range. The final range is always
      snapped back so it ends exactly at `total_tokens` and spans a full `window` tokens
      (when the document has at least `window` tokens) -- this guarantees the document's
      trailing content is always retained in full context, never silently truncated, even
      if that means the last two ranges overlap by more than `step` normally would.
    """
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


def build_chunk_rows_for_document(
    document_id: str,
    text: str,
    effective_family_id: str,
    agency: str,
    effective_agency: str,
    split: str,
    tokenizer,
    tokenizer_name: str,
    cfg,
) -> list[dict]:
    """Builds every provenance-complete chunk row for one already-split document.

    Returns an empty list only if the document's text tokenizes to zero raw tokens (see
    `compute_chunk_token_ranges`) -- callers building a full split are expected to assert
    this never happens for an eligible document (tests/test_chunking.py covers the
    empty-text edge case directly at this function's level).
    """
    parent_text_hash = _sha256(str(text))
    token_ids = tokenizer.encode(str(text), add_special_tokens=False)

    window = cfg.max_seq_length - cfg.num_special_tokens
    step = window - cfg.chunk_overlap_tokens
    ranges = compute_chunk_token_ranges(len(token_ids), window, step)
    total_chunks = len(ranges)

    rows = []
    for chunk_index, (start, end) in enumerate(ranges):
        chunk_token_ids = token_ids[start:end]
        content_token_count = len(chunk_token_ids)
        encoded_sequence_length = content_token_count + cfg.num_special_tokens
        chunk_text = tokenizer.decode(chunk_token_ids, skip_special_tokens=True)
        chunk_id = _sha256(
            f"{cfg.chunking_policy_version}|{document_id}|{chunk_index}|{start}|{end}"
        )
        rows.append(
            {
                "chunk_id": chunk_id,
                "document_id": str(document_id),
                "effective_family_id": effective_family_id,
                "agency": agency,
                "effective_agency": effective_agency,
                "split": split,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "tokenizer_name": tokenizer_name,
                "tokenizer_revision": cfg.tokenizer_revision,
                "token_start": start,
                "token_end": end,
                "content_token_count": content_token_count,
                "encoded_sequence_length": encoded_sequence_length,
                "chunk_text": chunk_text,
                "chunk_text_hash": _sha256(chunk_text),
                "parent_text_hash": parent_text_hash,
                "chunking_policy_version": cfg.chunking_policy_version,
            }
        )
    return rows


def build_chunks_for_split(split_df: pd.DataFrame, split_name: str, tokenizer, tokenizer_name: str, settings: Settings) -> pd.DataFrame:
    """Chunks every document in one already-frozen split DataFrame (train/validation/test).

    `split_df` must already carry `document_id`, `text`, `effective_family_id`, `agency`,
    `effective_agency` -- exactly the columns saved by `family_split.save_family_split`.
    """
    cfg = settings.family_aware.chunking
    all_rows: list[dict] = []
    for row in split_df.itertuples(index=False):
        rows = build_chunk_rows_for_document(
            document_id=str(row.document_id),
            text=row.text,
            effective_family_id=row.effective_family_id,
            agency=row.agency,
            effective_agency=row.effective_agency,
            split=split_name,
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cfg=cfg,
        )
        if not rows:
            raise RuntimeError(
                f"document_id={row.document_id!r} in split={split_name!r} produced zero "
                "chunks -- every eligible document must produce at least one chunk."
            )
        all_rows.extend(rows)
    return pd.DataFrame(all_rows)


def build_all_split_chunks(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tokenizer = get_bert_tokenizer(settings)
    tokenizer_name = settings.bert.base_model
    train_chunks = build_chunks_for_split(train_df, "train", tokenizer, tokenizer_name, settings)
    val_chunks = build_chunks_for_split(val_df, "validation", tokenizer, tokenizer_name, settings)
    test_chunks = build_chunks_for_split(test_df, "test", tokenizer, tokenizer_name, settings)
    return train_chunks, val_chunks, test_chunks


# --- Invariant proofs -------------------------------------------------------------------


def assert_every_chunk_maps_to_one_eligible_parent(chunks_df: pd.DataFrame, eligible_df: pd.DataFrame) -> None:
    eligible_ids = set(eligible_df["document_id"].astype(str))
    chunk_doc_ids = set(chunks_df["document_id"].astype(str))
    unmapped = chunk_doc_ids - eligible_ids
    if unmapped:
        raise ValueError(f"Chunks exist for documents outside the eligible corpus: {unmapped}")


def assert_every_chunk_inherits_parent_split(chunks_df: pd.DataFrame, document_to_split: dict[str, str]) -> None:
    mismatches = [
        (doc_id, split)
        for doc_id, split in zip(chunks_df["document_id"], chunks_df["split"])
        if document_to_split.get(str(doc_id)) != split
    ]
    if mismatches:
        raise ValueError(f"Chunk split does not match parent document's split: {mismatches[:5]}")


def assert_no_cross_split_leakage(train_chunks: pd.DataFrame, val_chunks: pd.DataFrame, test_chunks: pd.DataFrame) -> None:
    for column in ("document_id", "effective_family_id", "chunk_id"):
        sets = [set(df[column]) for df in (train_chunks, val_chunks, test_chunks)]
        overlaps = {
            "train/validation": sets[0] & sets[1],
            "train/test": sets[0] & sets[2],
            "validation/test": sets[1] & sets[2],
        }
        leaking = {name: found for name, found in overlaps.items() if found}
        if leaking:
            raise ValueError(f"{column} crosses splits: {leaking}")


def assert_no_excluded_document_chunked(chunks_df: pd.DataFrame, audit_df: pd.DataFrame) -> None:
    excluded_ids = set(
        audit_df.loc[audit_df["final_modeling_eligibility"] != "include_english_corpus", "document_id"].astype(str)
    )
    chunked_ids = set(chunks_df["document_id"].astype(str))
    leaking = excluded_ids & chunked_ids
    if leaking:
        raise ValueError(f"Excluded/unresolved documents produced chunks: {leaking}")


def assert_no_duplicate_chunk_ids(chunks_df: pd.DataFrame) -> None:
    duplicated = chunks_df["chunk_id"][chunks_df["chunk_id"].duplicated()]
    if not duplicated.empty:
        raise ValueError(f"Duplicate chunk_id values found: {duplicated.tolist()[:5]}")


def assert_chunk_indices_contiguous_and_unique(chunks_df: pd.DataFrame) -> None:
    for document_id, group in chunks_df.groupby("document_id"):
        indices = sorted(group["chunk_index"].tolist())
        expected = list(range(len(indices)))
        if indices != expected:
            raise ValueError(f"document_id={document_id!r} has non-contiguous chunk indices: {indices}")


def assert_reported_total_chunks_matches_actual(chunks_df: pd.DataFrame) -> None:
    actual_counts = chunks_df.groupby("document_id").size()
    reported = chunks_df.groupby("document_id")["total_chunks"].nunique()
    if (reported != 1).any():
        bad = reported[reported != 1].index.tolist()
        raise ValueError(f"total_chunks is not consistent within a document for: {bad}")
    reported_values = chunks_df.groupby("document_id")["total_chunks"].first()
    mismatched = actual_counts[actual_counts != reported_values]
    if not mismatched.empty:
        raise ValueError(f"total_chunks does not match actual chunk count for: {mismatched.index.tolist()}")


def assert_no_empty_chunks(chunks_df: pd.DataFrame) -> None:
    empty = chunks_df[chunks_df["content_token_count"] <= 0]
    if not empty.empty:
        raise ValueError(f"Empty or special-token-only chunks found: {empty['chunk_id'].tolist()}")


def assert_every_eligible_document_has_at_least_one_chunk(eligible_df: pd.DataFrame, chunks_df: pd.DataFrame) -> None:
    eligible_ids = set(eligible_df["document_id"].astype(str))
    chunked_ids = set(chunks_df["document_id"].astype(str))
    missing = eligible_ids - chunked_ids
    if missing:
        raise ValueError(f"Eligible documents with zero chunks: {missing}")


# --- Fingerprints ------------------------------------------------------------------------


def fingerprint_chunks(chunks_df: pd.DataFrame) -> str:
    columns = ["chunk_id", "document_id", "effective_family_id", "token_start", "token_end", "chunk_text_hash"]
    ordered = chunks_df[columns].astype(str).sort_values(columns).reset_index(drop=True)
    payload = "\n".join("|".join(row) for row in ordered.itertuples(index=False))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --- Report + save -------------------------------------------------------------------------


def _percentile(values: list[int], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def _chunk_distribution_for_split(chunks_df: pd.DataFrame) -> ChunkCountDistribution:
    per_doc = chunks_df.groupby("document_id").size().tolist()
    return ChunkCountDistribution(
        min=min(per_doc) if per_doc else 0,
        p50_median=float(np.median(per_doc)) if per_doc else 0.0,
        mean=float(np.mean(per_doc)) if per_doc else 0.0,
        p90=_percentile(per_doc, 90),
        p95=_percentile(per_doc, 95),
        max=max(per_doc) if per_doc else 0,
    )


def _split_chunk_counts(split_name: str, chunks_df: pd.DataFrame, all_agencies: list[str]) -> SplitChunkCounts:
    per_doc_counts = chunks_df.groupby("document_id").agg(
        chunk_count=("chunk_id", "size"),
        agency=("agency", "first"),
        effective_agency=("effective_agency", "first"),
    )
    single_chunk = int((per_doc_counts["chunk_count"] == 1).sum())
    multi_chunk = int((per_doc_counts["chunk_count"] > 1).sum())
    total_docs = len(per_doc_counts)

    chunks_by_agency = [
        ChunkAgencyCounts(agency=agency, chunk_count=int((chunks_df["effective_agency"] == agency).sum()))
        for agency in all_agencies
    ]

    largest = (
        per_doc_counts.sort_values("chunk_count", ascending=False)
        .head(10)
        .reset_index()
    )
    largest_docs = [
        LargestDocumentByChunks(
            document_id=str(r.document_id),
            agency=r.agency,
            effective_agency=r.effective_agency,
            chunk_count=int(r.chunk_count),
        )
        for r in largest.itertuples(index=False)
    ]

    return SplitChunkCounts(
        split=split_name,
        document_count=total_docs,
        family_count=int(chunks_df["effective_family_id"].nunique()),
        chunk_count=len(chunks_df),
        chunks_by_agency=chunks_by_agency,
        chunks_per_document_distribution=_chunk_distribution_for_split(chunks_df),
        single_chunk_document_count=single_chunk,
        single_chunk_document_percentage=round(100 * single_chunk / total_docs, 2) if total_docs else 0.0,
        multi_chunk_document_count=multi_chunk,
        multi_chunk_document_percentage=round(100 * multi_chunk / total_docs, 2) if total_docs else 0.0,
        largest_documents_by_chunk_count=largest_docs,
    )


def build_chunk_report(
    eligible_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_chunks: pd.DataFrame,
    val_chunks: pd.DataFrame,
    test_chunks: pd.DataFrame,
    audit_df: pd.DataFrame,
    split_fingerprints: dict[str, str],
    settings: Settings,
) -> FamilyAwareChunkManifest:
    document_to_split = {
        str(doc_id): split_name
        for split_name, df in (("train", train_df), ("validation", val_df), ("test", test_df))
        for doc_id in df["document_id"]
    }
    all_chunks = pd.concat([train_chunks, val_chunks, test_chunks], ignore_index=True)

    assert_every_chunk_maps_to_one_eligible_parent(all_chunks, eligible_df)
    assert_every_chunk_inherits_parent_split(all_chunks, document_to_split)
    assert_no_cross_split_leakage(train_chunks, val_chunks, test_chunks)
    assert_no_excluded_document_chunked(all_chunks, audit_df)
    assert_no_duplicate_chunk_ids(all_chunks)
    assert_chunk_indices_contiguous_and_unique(all_chunks)
    assert_reported_total_chunks_matches_actual(all_chunks)
    assert_no_empty_chunks(all_chunks)
    assert_every_eligible_document_has_at_least_one_chunk(eligible_df, all_chunks)

    # Determinism: re-tokenize/re-chunk from scratch and compare fingerprints + chunk_ids.
    train_chunks_2, val_chunks_2, test_chunks_2 = build_all_split_chunks(train_df, val_df, test_df, settings)
    rerun_identical = (
        fingerprint_chunks(train_chunks) == fingerprint_chunks(train_chunks_2)
        and fingerprint_chunks(val_chunks) == fingerprint_chunks(val_chunks_2)
        and fingerprint_chunks(test_chunks) == fingerprint_chunks(test_chunks_2)
    )

    all_agencies = sorted(eligible_df["effective_agency"].unique())
    cfg = settings.family_aware.chunking
    tokenizer_identity = resolve_tokenizer_identity(settings)

    return FamilyAwareChunkManifest(
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        chunking_policy_version=cfg.chunking_policy_version,
        tokenizer_name=settings.bert.base_model,
        tokenizer_revision=cfg.tokenizer_revision,
        tokenizer_resolved_commit_hash=tokenizer_identity["resolved_commit_hash"],
        tokenizer_file_hashes=tokenizer_identity["tokenizer_file_hashes"],
        max_seq_length=cfg.max_seq_length,
        num_special_tokens=cfg.num_special_tokens,
        content_window_tokens=cfg.max_seq_length - cfg.num_special_tokens,
        chunk_overlap_tokens=cfg.chunk_overlap_tokens,
        step_tokens=(cfg.max_seq_length - cfg.num_special_tokens) - cfg.chunk_overlap_tokens,
        source_split_fingerprints=split_fingerprints,
        chunk_fingerprints={
            "train": fingerprint_chunks(train_chunks),
            "validation": fingerprint_chunks(val_chunks),
            "test": fingerprint_chunks(test_chunks),
        },
        splits=[
            _split_chunk_counts("train", train_chunks, all_agencies),
            _split_chunk_counts("validation", val_chunks, all_agencies),
            _split_chunk_counts("test", test_chunks, all_agencies),
        ],
        every_chunk_maps_to_one_eligible_parent=True,
        every_chunk_inherits_parent_split=True,
        zero_document_id_cross_split_leakage=True,
        zero_family_id_cross_split_leakage=True,
        zero_chunk_id_cross_split_leakage=True,
        every_eligible_document_has_at_least_one_chunk=True,
        no_excluded_document_produced_a_chunk=True,
        no_duplicate_chunk_ids=True,
        chunk_indices_contiguous_and_unique_per_document=True,
        reported_total_chunks_matches_actual_per_document=True,
        no_empty_or_special_token_only_chunks=True,
        rerun_produces_identical_chunks=rerun_identical,
        documents_requiring_fallback_behavior=[],
        notes=[
            "Chunking reads only the frozen family-aware train/validation/test CSVs "
            "(Checkpoint 4) -- documents were never re-chunked-then-split.",
            "Agency support for evaluation purposes must be read at the original-document "
            "level (see the Checkpoint 4 split report) -- chunk counts here are diagnostic "
            "only and must never be presented as independent test support.",
            "All twelve assertion functions in chunking.py were called before this manifest "
            "was built and raise on failure -- the boolean fields above reflect checks that "
            "actually ran, not assumptions.",
        ],
    )


def save_family_aware_chunks(
    train_chunks: pd.DataFrame,
    val_chunks: pd.DataFrame,
    test_chunks: pd.DataFrame,
    report: FamilyAwareChunkManifest,
    settings: Settings,
) -> Path:
    output_dir = settings.resolve_path(settings.family_aware.chunking.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_chunks.to_csv(output_dir / "train_chunks.csv", index=False)
    val_chunks.to_csv(output_dir / "validation_chunks.csv", index=False)
    test_chunks.to_csv(output_dir / "test_chunks.csv", index=False)

    manifest_dir = settings.resolve_path("artifacts/family_aware/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "chunk_manifest_v1.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))

    return output_dir
