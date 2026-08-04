"""Metrics, paired comparison, robustness comparison, and statistical uncertainty for
Version 6 Checkpoint 10's head-to-head evaluation.

Every metric here operates on document-level predictions (one row per (document_id,
condition) per method) -- chunk counts are never treated as independent evaluation support,
and Checkpoint 9's retrieval hit-rate/MRR are retrieval diagnostics, never substituted for
these classification metrics.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from newstart_ai.schemas.checkpoint10 import (
    BootstrapResult,
    LatencyStats,
    McNemarResult,
    MethodConditionMetrics,
    PairedBootstrapResult,
    PerAgencyMetrics,
    PrimaryPairedComparison,
    RobustnessComparisonManifest,
    RobustnessConditionDelta,
    StatisticalUncertaintyManifest,
)

# A "failed"/"invalid" case has no predicted_label. The frozen failure-scoring rule treats
# it as an automatic wrong answer (never excluded from the denominator) -- scored against a
# sentinel label guaranteed not to match any true label, so it always counts as an error.
_FAILURE_SENTINEL_LABEL = "__NO_PREDICTION__"


def _scored_prediction(case) -> str:
    return case.predicted_label if case.predicted_label is not None else _FAILURE_SENTINEL_LABEL


def build_method_condition_metrics(cases: list, label_order: list[str]) -> MethodConditionMetrics:
    """`cases` must be every CaseResult for one (method, condition) pair -- exactly one per
    document, including failed/invalid cases (never excluded)."""
    method = cases[0].method
    condition = cases[0].condition
    n = len(cases)

    y_true = [c.true_label for c in cases]
    y_pred = [_scored_prediction(c) for c in cases]

    invalid_count = sum(1 for c in cases if c.status == "invalid")
    failed_count = sum(1 for c in cases if c.status == "failed")
    success_count = sum(1 for c in cases if c.status == "success")
    coverage_rate = round(100 * success_count / n, 2) if n else 0.0

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=label_order, average="macro", zero_division=0
    )
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n if n else 0.0

    precisions, recalls, f1s, supports = precision_recall_fscore_support(y_true, y_pred, labels=label_order, zero_division=0)
    per_agency = [
        PerAgencyMetrics(agency=label, precision=float(p), recall=float(r), f1=float(f), support=int(s))
        for label, p, r, f, s in zip(label_order, precisions, recalls, f1s, supports)
    ]

    cm = sk_confusion_matrix(y_true, y_pred, labels=label_order)
    confusion = {t: {p: int(cm[i, j]) for j, p in enumerate(label_order)} for i, t in enumerate(label_order)}

    errors = sum(1 for t, p in zip(y_true, y_pred) if t != p)

    latencies = [c.latency_ms for c in cases if c.latency_ms is not None]
    latency_stats = LatencyStats(
        mean_ms=float(np.mean(latencies)) if latencies else 0.0,
        median_ms=float(np.median(latencies)) if latencies else 0.0,
        p95_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
        total_ms=float(np.sum(latencies)) if latencies else 0.0,
    )

    token_totals = [c.total_tokens for c in cases if c.total_tokens is not None]
    cost_totals = [c.estimated_cost_usd for c in cases if c.estimated_cost_usd is not None]

    notes = ["Failed/invalid cases are scored as incorrect and included in every denominator above -- never excluded."]
    if method == "bert":
        notes.append("BERT case results are reused from the frozen Checkpoint 8 predictions where fingerprints match -- not rerun.")

    return MethodConditionMetrics(
        method=method, condition=condition, document_count=n, coverage_rate=coverage_rate,
        invalid_count=invalid_count, failed_count=failed_count,
        document_macro_f1=float(macro_f1), document_accuracy=float(accuracy),
        macro_precision=float(macro_precision), macro_recall=float(macro_recall),
        per_agency=per_agency, confusion_matrix=confusion, error_count=errors,
        error_rate=round(100 * errors / n, 2) if n else 0.0, latency=latency_stats,
        estimated_tokens_total=int(sum(token_totals)) if token_totals else None,
        estimated_cost_total_usd=round(sum(cost_totals), 6) if cost_totals else None,
        notes=notes,
    )


def build_primary_paired_comparison(
    bert_preds: dict[str, str], llm_preds: dict[str, str], rag_preds: dict[str, str], true_labels: dict[str, str], condition: str = "complete_unmasked"
) -> PrimaryPairedComparison:
    document_ids = sorted(true_labels.keys())

    all_correct, all_incorrect = [], []
    bert_only, llm_only, rag_only = [], [], []
    rag_corrects, rag_breaks, identical = [], [], []

    for doc_id in document_ids:
        true = true_labels[doc_id]
        b_ok = bert_preds[doc_id] == true
        l_ok = llm_preds[doc_id] == true
        r_ok = rag_preds[doc_id] == true

        if b_ok and l_ok and r_ok:
            all_correct.append(doc_id)
        if not b_ok and not l_ok and not r_ok:
            all_incorrect.append(doc_id)
        if not b_ok and l_ok and r_ok:
            bert_only.append(doc_id)
        if b_ok and not l_ok and r_ok:
            llm_only.append(doc_id)
        if b_ok and l_ok and not r_ok:
            rag_only.append(doc_id)
        if not l_ok and r_ok:
            rag_corrects.append(doc_id)
        if l_ok and not r_ok:
            rag_breaks.append(doc_id)
        if llm_preds[doc_id] == rag_preds[doc_id]:
            identical.append(doc_id)

    return PrimaryPairedComparison(
        version="v1", created_at=datetime.now(timezone.utc).isoformat(), condition=condition,
        document_count=len(document_ids),
        all_three_correct=all_correct, all_three_incorrect=all_incorrect,
        bert_only_errors=bert_only, plain_llm_only_errors=llm_only,
        rag_only_errors=rag_only,
        rag_corrects_plain_llm=rag_corrects, rag_breaks_plain_llm=rag_breaks,
        plain_llm_and_rag_identical_predictions=identical,
        plain_llm_and_rag_agreement_rate=round(100 * len(identical) / len(document_ids), 2) if document_ids else 0.0,
        notes=[
            "'X_only_errors' means every OTHER method got this document right and X alone got it wrong.",
            "Do not claim statistical superiority from these small raw counts alone -- see the statistical uncertainty manifest.",
        ],
    )


def build_robustness_comparison(all_metrics: list[MethodConditionMetrics]) -> RobustnessComparisonManifest:
    by_method: dict[str, dict[str, MethodConditionMetrics]] = {}
    for m in all_metrics:
        by_method.setdefault(m.method, {})[m.condition] = m

    deltas = []
    for method, by_condition in by_method.items():
        base = by_condition.get("complete_unmasked")
        if base is None:
            continue
        for condition, m in by_condition.items():
            deltas.append(
                RobustnessConditionDelta(
                    method=method, condition=condition,
                    macro_f1_delta_from_complete_unmasked=round(m.document_macro_f1 - base.document_macro_f1, 4),
                    accuracy_delta_from_complete_unmasked=round(m.document_accuracy - base.document_accuracy, 4),
                    error_count=m.error_count, error_rate=m.error_rate,
                )
            )

    masked_conditions = {"complete_masked", "beginning_only_masked", "middle_only_masked", "end_only_masked", "beginning_middle_end_masked"}
    masking_notes = []
    rag_help_notes = []
    for method, by_condition in by_method.items():
        base = by_condition.get("complete_unmasked")
        masked_base = by_condition.get("complete_masked")
        if base and masked_base:
            masking_notes.append(
                f"{method}: complete_unmasked macro F1 {base.document_macro_f1:.4f} vs. complete_masked {masked_base.document_macro_f1:.4f} "
                f"(delta {masked_base.document_macro_f1 - base.document_macro_f1:+.4f})."
            )
    if "llm" in by_method and "llm_rag" in by_method:
        for condition in masked_conditions:
            if condition in by_method["llm"] and condition in by_method["llm_rag"]:
                llm_f1 = by_method["llm"][condition].document_macro_f1
                rag_f1 = by_method["llm_rag"][condition].document_macro_f1
                rag_help_notes.append(f"{condition}: plain LLM {llm_f1:.4f} vs. LLM+RAG {rag_f1:.4f} (RAG {'helped' if rag_f1 > llm_f1 else 'did not help' if rag_f1 < llm_f1 else 'made no difference'}).")

    partial_notes = []
    for method, by_condition in by_method.items():
        for region in ("beginning_only_unmasked", "middle_only_unmasked", "end_only_unmasked", "beginning_middle_end_unmasked"):
            if region in by_condition:
                partial_notes.append(f"{method} {region}: macro F1 {by_condition[region].document_macro_f1:.4f}, error rate {by_condition[region].error_rate}%.")

    return RobustnessComparisonManifest(
        version="v1", created_at=datetime.now(timezone.utc).isoformat(), deltas=deltas,
        masking_effect_notes=masking_notes, partial_input_effect_notes=partial_notes,
        rag_help_notes=rag_help_notes,
        error_concentration_notes=["See the per-method error-analysis CSV for agency/family/chunk-count/truncation breakdowns."],
        irs_caution_note="IRS test support is only 4 documents -- any IRS-specific delta here can swing by 25 percentage points from a single case. Treat as descriptive, not conclusive.",
        disclaimer="All comparisons in this manifest are descriptive associations from one evaluation run, not causal effects.",
    )


def _bootstrap_metric(y_true: list[str], y_pred: list[str], label_order: list[str], metric: str, n_bootstrap: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true_arr, y_pred_arr = np.array(y_true), np.array(y_pred)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bt, bp = y_true_arr[idx], y_pred_arr[idx]
        if metric == "macro_f1":
            _, _, f1, _ = precision_recall_fscore_support(bt, bp, labels=label_order, average="macro", zero_division=0)
            values.append(f1)
        else:
            values.append(float(np.mean(bt == bp)))
    point = np.mean([1 if t == p else 0 for t, p in zip(y_true, y_pred)]) if metric == "accuracy" else precision_recall_fscore_support(y_true, y_pred, labels=label_order, average="macro", zero_division=0)[2]
    return float(point), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def build_statistical_uncertainty(
    cases_by_method: dict[str, list], label_order: list[str], condition: str = "complete_unmasked", n_bootstrap: int = 2000, seed: int = 42
) -> StatisticalUncertaintyManifest:
    from scipy import stats as scipy_stats

    bootstrap_results = []
    paired_results = []
    mcnemar_results = []

    per_method_preds: dict[str, dict[str, str]] = {}
    per_method_true: dict[str, dict[str, str]] = {}
    for method, cases in cases_by_method.items():
        per_method_preds[method] = {c.document_id: _scored_prediction(c) for c in cases}
        per_method_true[method] = {c.document_id: c.true_label for c in cases}
        y_true = [c.true_label for c in cases]
        y_pred = [_scored_prediction(c) for c in cases]
        for metric in ("macro_f1", "accuracy"):
            point, lo, hi = _bootstrap_metric(y_true, y_pred, label_order, metric, n_bootstrap, seed)
            bootstrap_results.append(BootstrapResult(metric=metric, method=method, point_estimate=point, ci_low=lo, ci_high=hi, n_bootstrap=n_bootstrap, seed=seed))

    methods = list(cases_by_method.keys())
    document_ids = sorted(per_method_true[methods[0]].keys()) if methods else []
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method_a, method_b = methods[i], methods[j]
            for metric in ("macro_f1", "accuracy"):
                rng = np.random.default_rng(seed)
                n = len(document_ids)
                diffs = []
                y_true_all = [per_method_true[method_a][d] for d in document_ids]
                y_pred_a_all = [per_method_preds[method_a][d] for d in document_ids]
                y_pred_b_all = [per_method_preds[method_b][d] for d in document_ids]
                for _ in range(n_bootstrap):
                    idx = rng.integers(0, n, size=n)
                    bt = [y_true_all[k] for k in idx]
                    ba = [y_pred_a_all[k] for k in idx]
                    bb = [y_pred_b_all[k] for k in idx]
                    if metric == "macro_f1":
                        _, _, f1a, _ = precision_recall_fscore_support(bt, ba, labels=label_order, average="macro", zero_division=0)
                        _, _, f1b, _ = precision_recall_fscore_support(bt, bb, labels=label_order, average="macro", zero_division=0)
                        diffs.append(f1a - f1b)
                    else:
                        diffs.append(float(np.mean(np.array(bt) == np.array(ba)) - np.mean(np.array(bt) == np.array(bb))))
                if metric == "macro_f1":
                    _, _, obs_a, _ = precision_recall_fscore_support(y_true_all, y_pred_a_all, labels=label_order, average="macro", zero_division=0)
                    _, _, obs_b, _ = precision_recall_fscore_support(y_true_all, y_pred_b_all, labels=label_order, average="macro", zero_division=0)
                    observed = float(obs_a - obs_b)
                else:
                    observed = float(np.mean(np.array(y_true_all) == np.array(y_pred_a_all)) - np.mean(np.array(y_true_all) == np.array(y_pred_b_all)))
                paired_results.append(
                    PairedBootstrapResult(
                        metric=metric, method_a=method_a, method_b=method_b, observed_difference=observed,
                        ci_low=float(np.percentile(diffs, 2.5)), ci_high=float(np.percentile(diffs, 97.5)),
                        n_bootstrap=n_bootstrap, seed=seed,
                    )
                )

            a_correct_b_incorrect = sum(
                1 for d in document_ids
                if per_method_preds[method_a][d] == per_method_true[method_a][d] and per_method_preds[method_b][d] != per_method_true[method_b][d]
            )
            b_correct_a_incorrect = sum(
                1 for d in document_ids
                if per_method_preds[method_b][d] == per_method_true[method_b][d] and per_method_preds[method_a][d] != per_method_true[method_a][d]
            )
            n_disagree = a_correct_b_incorrect + b_correct_a_incorrect
            if n_disagree >= 10:
                binom_result = scipy_stats.binomtest(a_correct_b_incorrect, n_disagree, 0.5)
                statistic, p_value = float(binom_result.statistic), float(binom_result.pvalue)
                note = "McNemar exact binomial test computed (sufficient disagreement count)."
            else:
                statistic, p_value = None, None
                note = f"Disagreement count ({n_disagree}) too small for a meaningful McNemar test -- not computed."
            mcnemar_results.append(
                McNemarResult(
                    method_a=method_a, method_b=method_b, a_correct_b_incorrect=a_correct_b_incorrect,
                    a_incorrect_b_correct=b_correct_a_incorrect, statistic=statistic, p_value=p_value, note=note,
                )
            )

    return StatisticalUncertaintyManifest(
        version="v1", created_at=datetime.now(timezone.utc).isoformat(), condition=condition, seed=seed,
        bootstrap_results=bootstrap_results, paired_bootstrap_results=paired_results, mcnemar_results=mcnemar_results,
        interpretation_caution=(
            "With only 99 test documents (and 4 IRS documents), confidence intervals are wide "
            "and any hypothesis test result should be read as suggestive at most, not "
            "definitive proof of superiority."
        ),
    )
