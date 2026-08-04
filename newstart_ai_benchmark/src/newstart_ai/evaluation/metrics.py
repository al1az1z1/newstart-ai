"""Shared metrics computation for every method (BERT, LLM, LLM+RAG).

Macro F1 is always the primary metric (docs/BLUEPRINT.md Section 6); per-class metrics and
the confusion matrix are always included so a small class (IRS) is never hidden behind an
aggregate number.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from newstart_ai.schemas.classification import Method
from newstart_ai.schemas.evaluation import MetricsReport, PerClassMetrics


def evaluate_predictions(
    true_labels: list[str],
    predicted_labels: list[str],
    label_order: list[str],
    method: Method,
    split: str,
    latencies_ms: list[float] | None = None,
    total_token_usage: int | None = None,
    total_estimated_cost: float | None = None,
    notes: list[str] | None = None,
) -> MetricsReport:
    """Computes accuracy, macro/weighted F1, per-class metrics, and the confusion matrix.

    `split` should be "validation" or "test" -- callers are responsible for never mixing the
    two (see docs/BLUEPRINT.md Section 4: the validation-only side evaluation never feeds the
    primary test-set comparison).
    """
    true_arr = np.array(true_labels)
    pred_arr = np.array(predicted_labels)

    accuracy = float((true_arr == pred_arr).mean())

    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=label_order, average=None, zero_division=0
    )
    per_class = [
        PerClassMetrics(label=label, precision=float(p), recall=float(r), f1=float(f), support=int(s))
        for label, p, r, f, s in zip(label_order, precision, recall, f1, support)
    ]

    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    _, _, weighted_f1_arr, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=label_order, average="weighted", zero_division=0
    )
    weighted_f1 = float(weighted_f1_arr)

    cm = confusion_matrix(true_labels, predicted_labels, labels=label_order).tolist()

    mean_latency_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    cost_per_document = (
        total_estimated_cost / len(true_labels)
        if total_estimated_cost is not None and len(true_labels) > 0
        else None
    )

    return MetricsReport(
        method=method,
        split=split,
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class=per_class,
        confusion_matrix=cm,
        confusion_matrix_labels=label_order,
        mean_latency_ms=mean_latency_ms,
        total_token_usage=total_token_usage,
        total_estimated_cost=total_estimated_cost,
        cost_per_document=cost_per_document,
        notes=notes or [],
    )
