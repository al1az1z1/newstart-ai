"""Training loop for the new family-aware chunked BERT classifier (Version 6, Checkpoint 7).

Trains only on the frozen family-aware TRAIN chunks (complete, unmasked); uses the frozen
family-aware VALIDATION chunks only for per-epoch checkpoint selection (document-level macro
F1 via the provisional `mean_probabilities` aggregation -- reconfirmed against all four
methods on the selected best checkpoint afterward, see `aggregation.py`). Never opens or
references the test split.

This module is entirely separate from `newstart_ai.models.bert.classifier.BERTClassifier`
(the historical MVP model) -- the historical model, its artifact, and its training code are
never modified or reused for this new model.
"""

from __future__ import annotations

import copy
import random
import time
import warnings

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from newstart_ai.models.bert.aggregation import aggregate_document
from newstart_ai.models.bert.family_aware_dataset import FamilyAwareChunkDataset
from newstart_ai.models.bert.weighted_loss import weighted_cross_entropy
from newstart_ai.schemas.checkpoint7 import TrainingHistoryEpoch


def set_determinism(seed: int) -> list[str]:
    """Seeds every RNG this training loop touches and requests deterministic CUDA algorithms
    where possible. Returns any warnings PyTorch raised about operations it cannot guarantee
    fully deterministic on the current CUDA build (reported, never silently ignored)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    captured: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:  # pragma: no cover -- defensive, PyTorch-version dependent
            captured.append(str(exc))
        for w in caught:
            captured.append(str(w.message))
    return captured


def generate_chunk_level_outputs(
    model,
    tokenizer,
    chunks_df,
    document_texts: dict[str, str],
    label_order: list[str],
    max_seq_length: int,
    device,
    batch_size: int = 16,
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
    """Runs the model once over every chunk in `chunks_df` (position-sliced from
    `document_texts`, exactly reproducing Checkpoint 5's chunk boundaries) and groups the
    resulting per-chunk probabilities/logits back by document_id, in chunk_index order."""
    label_to_index = {label: i for i, label in enumerate(label_order)}
    dataset = FamilyAwareChunkDataset(
        chunks_df, document_texts, tokenizer, max_seq_length, label_to_index,
        class_weight_by_label={label: 1.0 for label in label_order},
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    probs_by_doc: dict[str, list[np.ndarray]] = {}
    logits_by_doc: dict[str, list[np.ndarray]] = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            doc_ids = batch["document_id"]
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            logits_np = logits.cpu().numpy()
            for doc_id, prob_vec, logit_vec in zip(doc_ids, probs, logits_np):
                probs_by_doc.setdefault(doc_id, []).append(prob_vec)
                logits_by_doc.setdefault(doc_id, []).append(logit_vec)
    return probs_by_doc, logits_by_doc


def train_family_aware_bert(
    model,
    tokenizer,
    train_chunks_df,
    train_document_texts: dict[str, str],
    val_chunks_df,
    val_document_texts: dict[str, str],
    label_order: list[str],
    class_weight_by_label: dict[str, float],
    settings,
    progress_callback=None,
) -> dict:
    cfg = settings.family_aware.training
    max_seq_length = settings.family_aware.chunking.max_seq_length
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    label_to_index = {label: i for i, label in enumerate(label_order)}

    train_dataset = FamilyAwareChunkDataset(
        train_chunks_df, train_document_texts, tokenizer, max_seq_length, label_to_index, class_weight_by_label
    )
    val_dataset = FamilyAwareChunkDataset(
        val_chunks_df, val_document_texts, tokenizer, max_seq_length, label_to_index,
        class_weight_by_label={label: 1.0 for label in label_order},  # unweighted for diagnostic val loss
    )

    generator = torch.Generator()
    generator.manual_seed(cfg.random_seed)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    val_true_labels = val_chunks_df.groupby("document_id")["effective_agency"].first().to_dict()
    selection_method = cfg.checkpoint_selection_aggregation_method

    history: list[TrainingHistoryEpoch] = []
    best_state = None
    best_epoch = None
    best_val_macro_f1 = -1.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    non_deterministic_op_warnings: list[str] = []
    training_start = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        epoch_start = time.time()

        model.train()
        running_loss = 0.0
        total_weight = 0.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                weights = batch["weight"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = weighted_cross_entropy(outputs.logits, labels, weights)
                loss.backward()
                optimizer.step()

                batch_weight_sum = weights.sum().item()
                running_loss += loss.item() * batch_weight_sum
                total_weight += batch_weight_sum
            for w in caught:
                message = str(w.message)
                if message not in non_deterministic_op_warnings:
                    non_deterministic_op_warnings.append(message)
        train_loss = running_loss / total_weight

        model.eval()
        chunk_probs_by_doc: dict[str, list[np.ndarray]] = {}
        chunk_logits_by_doc: dict[str, list[np.ndarray]] = {}
        val_loss_running = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                doc_ids = batch["document_id"]

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                per_example_loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
                val_loss_running += per_example_loss.sum().item()
                val_count += len(labels)

                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                logits_np = logits.cpu().numpy()
                for doc_id, prob_vec, logit_vec in zip(doc_ids, probs, logits_np):
                    chunk_probs_by_doc.setdefault(doc_id, []).append(prob_vec)
                    chunk_logits_by_doc.setdefault(doc_id, []).append(logit_vec)
        val_loss = val_loss_running / val_count

        val_preds = {}
        for doc_id, probs in chunk_probs_by_doc.items():
            result = aggregate_document(
                selection_method,
                label_order,
                chunk_probs=probs,
                chunk_logits=chunk_logits_by_doc[doc_id] if selection_method == "mean_logits" else None,
            )
            val_preds[doc_id] = result.predicted_label

        ordered_doc_ids = list(val_preds.keys())
        y_true = [val_true_labels[d] for d in ordered_doc_ids]
        y_pred = [val_preds[d] for d in ordered_doc_ids]
        val_macro_f1 = f1_score(y_true, y_pred, average="macro", labels=label_order, zero_division=0)
        val_accuracy = accuracy_score(y_true, y_pred)

        epoch_duration = time.time() - epoch_start
        entry = TrainingHistoryEpoch(
            epoch=epoch,
            train_loss=train_loss,
            validation_loss=val_loss,
            validation_document_macro_f1=float(val_macro_f1),
            validation_document_accuracy=float(val_accuracy),
            learning_rate=cfg.learning_rate,
            epoch_duration_seconds=epoch_duration,
        )
        history.append(entry)
        if progress_callback:
            progress_callback(entry)

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    training_time = time.time() - training_start
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else None

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_document_macro_f1": float(best_val_macro_f1),
        "stopping_reason": (
            f"Completed all {cfg.max_epochs} configured epochs (no early-stopping trigger "
            f"configured); selected epoch {best_epoch} by highest validation document-level "
            f"macro F1 ({cfg.checkpoint_selection_metric})."
        ),
        "training_time_seconds": training_time,
        "peak_gpu_memory_mb": peak_memory_mb,
        "non_deterministic_op_warnings": non_deterministic_op_warnings,
    }
