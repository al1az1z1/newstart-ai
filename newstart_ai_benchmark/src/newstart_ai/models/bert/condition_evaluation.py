"""Evaluates the family-aware chunked BERT across all ten frozen robustness conditions
(Version 6, Checkpoint 7).

For every condition, the exact chunk text registered by Checkpoint 6's condition registry is
what gets encoded and scored -- "complete" conditions use every one of a document's frozen
chunks (unmasked or masked); partial conditions use the specific chunk(s) Checkpoint 6's
partial-input policy selected for that document. Document-level macro F1/accuracy (never
chunk counts) are the reported metrics. Robustness results here must never feed back into
re-tuning masking or partial-input policy -- they are measurements, not a tuning signal.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from newstart_ai.data.partial_input import CONDITIONS as PARTIAL_CONDITIONS
from newstart_ai.models.bert.aggregation import aggregate_document
from newstart_ai.schemas.checkpoint7 import ConditionEvaluationManifest, ConditionEvaluationResult

_CONDITION_SPECS: dict[str, tuple[str, bool]] = {
    "complete_unmasked": ("complete", False),
    "beginning_only_unmasked": ("beginning", False),
    "middle_only_unmasked": ("middle", False),
    "end_only_unmasked": ("end", False),
    "beginning_middle_end_unmasked": ("beginning_middle_end", False),
    "complete_masked": ("complete", True),
    "beginning_only_masked": ("beginning", True),
    "middle_only_masked": ("middle", True),
    "end_only_masked": ("end", True),
    "beginning_middle_end_masked": ("beginning_middle_end", True),
}

_PARTIAL_TO_SELECTION_CONDITION = {
    "beginning": "beginning_only",
    "middle": "middle_only",
    "end": "end_only",
    "beginning_middle_end": "beginning_middle_end",
}


def _encode_text(text: str, tokenizer, max_seq_length: int, num_special_tokens: int) -> tuple[list[int], list[int]]:
    window = max_seq_length - num_special_tokens
    token_ids = tokenizer.encode(str(text), add_special_tokens=False, truncation=True, max_length=window)
    ids = [tokenizer.cls_token_id, *token_ids, tokenizer.sep_token_id]
    attention_mask = [1] * len(ids)
    pad_length = max_seq_length - len(ids)
    if pad_length > 0:
        ids = ids + [tokenizer.pad_token_id] * pad_length
        attention_mask = attention_mask + [0] * pad_length
    return ids, attention_mask


def _run_model_on_texts(model, tokenizer, texts: list[str], max_seq_length: int, num_special_tokens: int, device, batch_size: int = 16):
    model.eval()
    all_probs = []
    all_logits = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = [_encode_text(t, tokenizer, max_seq_length, num_special_tokens) for t in batch_texts]
            input_ids = torch.tensor([e[0] for e in encoded], dtype=torch.long).to(device)
            attention_mask = torch.tensor([e[1] for e in encoded], dtype=torch.long).to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.extend(list(probs))
            all_logits.extend(list(logits.cpu().numpy()))
    return all_probs, all_logits


def _condition_chunk_texts_for_document(
    document_id: str,
    condition_name: str,
    unmasked_chunks_by_doc: dict,
    masked_chunks_by_doc: dict,
    selections_by_doc_condition: dict,
) -> list[str]:
    region, masked = _CONDITION_SPECS[condition_name]
    doc_chunks = masked_chunks_by_doc[document_id] if masked else unmasked_chunks_by_doc[document_id]
    text_col = "masked_chunk_text" if masked else "chunk_text"

    if region == "complete":
        ordered = doc_chunks.sort_values("chunk_index")
        return ordered[text_col].tolist()

    selection_condition = _PARTIAL_TO_SELECTION_CONDITION[region]
    selection = selections_by_doc_condition[(document_id, selection_condition)]
    by_index = doc_chunks.set_index("chunk_index")[text_col]
    return [str(by_index.loc[i]) for i in selection.selected_chunk_indices]


def evaluate_all_conditions(
    model,
    tokenizer,
    val_chunks_df: pd.DataFrame,
    val_masked_chunks_df: pd.DataFrame,
    val_selections_df: pd.DataFrame,
    true_labels_by_doc: dict[str, str],
    label_order: list[str],
    aggregation_method: str,
    max_seq_length: int,
    num_special_tokens: int,
    condition_registry_fingerprint: str,
    device,
) -> tuple[ConditionEvaluationManifest, dict[str, dict[str, str]]]:
    unmasked_chunks_by_doc = {doc_id: g for doc_id, g in val_chunks_df.groupby("document_id")}
    masked_chunks_by_doc = {doc_id: g for doc_id, g in val_masked_chunks_df.groupby("document_id")}
    selections_by_doc_condition = {(row.document_id, row.condition): row for row in val_selections_df.itertuples(index=False)}

    results = []
    raw_predictions_by_condition: dict[str, dict[str, str]] = {}
    for condition_name in _CONDITION_SPECS:
        region, masked = _CONDITION_SPECS[condition_name]
        document_ids = list(unmasked_chunks_by_doc.keys())

        flat_texts: list[str] = []
        owner_index: list[int] = []
        for i, doc_id in enumerate(document_ids):
            texts = _condition_chunk_texts_for_document(
                doc_id, condition_name, unmasked_chunks_by_doc, masked_chunks_by_doc, selections_by_doc_condition
            )
            flat_texts.extend(texts)
            owner_index.extend([i] * len(texts))

        all_probs, all_logits = _run_model_on_texts(model, tokenizer, flat_texts, max_seq_length, num_special_tokens, device)

        probs_by_doc: dict[str, list[np.ndarray]] = {doc_id: [] for doc_id in document_ids}
        logits_by_doc: dict[str, list[np.ndarray]] = {doc_id: [] for doc_id in document_ids}
        for owner, prob, logit in zip(owner_index, all_probs, all_logits):
            doc_id = document_ids[owner]
            probs_by_doc[doc_id].append(prob)
            logits_by_doc[doc_id].append(logit)

        predictions = {}
        for doc_id in document_ids:
            result = aggregate_document(
                aggregation_method,
                label_order,
                chunk_probs=probs_by_doc[doc_id],
                chunk_logits=logits_by_doc[doc_id] if aggregation_method == "mean_logits" else None,
            )
            predictions[doc_id] = result.predicted_label

        raw_predictions_by_condition[condition_name] = predictions

        y_true = [true_labels_by_doc[d] for d in document_ids]
        y_pred = [predictions[d] for d in document_ids]

        macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=label_order, zero_division=0))
        accuracy = float(accuracy_score(y_true, y_pred))
        per_agency_f1 = {
            label: float(f1)
            for label, f1 in zip(label_order, f1_score(y_true, y_pred, average=None, labels=label_order, zero_division=0))
        }
        per_agency_support = {label: int(y_true.count(label)) for label in label_order}

        results.append(
            ConditionEvaluationResult(
                condition=condition_name,
                masked=masked,
                region=region,
                document_count=len(document_ids),
                document_macro_f1=macro_f1,
                document_accuracy=accuracy,
                per_agency_f1=per_agency_f1,
                per_agency_support=per_agency_support,
            )
        )

    manifest = ConditionEvaluationManifest(
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        aggregation_method_used=aggregation_method,
        condition_registry_fingerprint=condition_registry_fingerprint,
        results=results,
        notes=[
            "These are robustness measurements only -- results here were never used to "
            "retrain the model or alter masking/partial-input policy.",
            "Document-level macro F1/accuracy only; chunk counts within a condition are "
            "never treated as independent evaluation support.",
        ],
    )
    return manifest, raw_predictions_by_condition
