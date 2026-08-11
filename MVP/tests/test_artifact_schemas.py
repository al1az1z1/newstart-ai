"""For every artifact documented in docs/ARTIFACTS.md: the real file exists and loads, and
its row/document counts are internally consistent with its own manifest."""

from __future__ import annotations

from newstart_ai_mvp import artifact_report as ar


def test_language_audit_exists_and_loads(settings):
    report = ar.describe_language_audit(settings)
    assert report["document_count"] == 754


def test_family_audit_exists_and_loads(settings):
    report = ar.describe_family_audit(settings)
    assert report["document_count"] == 754
    assert sum(report["final_modeling_eligibility_counts"].values()) == 754


def test_split_exists_and_loads(settings):
    report = ar.describe_split(settings)
    total = sum(v["documents"] for v in report["counts"].values())
    assert total == 659  # 461 + 99 + 99, the eligible corpus


def test_chunks_exist_and_load(settings):
    report = ar.describe_chunks(settings)
    assert all(v["chunks"] > 0 for v in report["counts"].values())


def test_masked_exists_and_loads(settings):
    report = ar.describe_masked(settings)
    for split_counts in report["counts"].values():
        assert split_counts["documents_with_replacements"] <= split_counts["documents"]


def test_conditions_exist_and_load(settings):
    report = ar.describe_conditions(settings)
    assert len(report["test_conditions"]) == 10
    assert report["test_documents"] == 99


def test_bert_checkpoint_exists_and_is_ready(settings):
    report = ar.describe_bert_checkpoint(settings)
    assert report["status"] == "ready"
    assert report["best_epoch"] in report["history_epochs"]


def test_rag_index_exists_with_matching_masked_unmasked_counts(settings):
    report = ar.describe_rag_index(settings)
    assert report["masked"]["chunk_count"] == report["unmasked"]["chunk_count"]
