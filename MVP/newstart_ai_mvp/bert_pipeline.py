"""Self-contained BERT training, aggregation, and test-evaluation pipeline (Checkpoints 6-8).

A copy of the original project's family-aware BERT logic, reorganized into one module.
Manifests are plain dicts instead of Pydantic schema objects; the actual methodology
(combined chunk-count x class weighting, the training loop, the four aggregation methods,
and the frozen-checkpoint test evaluation) is unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


# =========================================================================================
# Class weighting (Checkpoint 7)
# =========================================================================================


def compute_class_weights(label_counts: dict[str, int], label_order: list[str], threshold: float) -> dict[str, float] | None:
    """Inverse-frequency weights: weight[label] = total / (num_labels * count[label]).
    Returns None (no weighting) if the training set's imbalance ratio is below `threshold`."""
    counts = [label_counts.get(label, 0) for label in label_order]
    if min(c for c in counts if c > 0) == 0:
        return None
    ratio = max(counts) / min(counts)
    if ratio < threshold:
        return None
    total = sum(counts)
    return {label: total / (len(label_order) * count) for label, count in zip(label_order, counts)}


def build_agency_class_weight_manifest(train_split_df: pd.DataFrame, label_order: list[str], settings) -> dict:
    threshold = settings.family_aware.training.imbalance.weighted_loss_threshold
    counts = train_split_df["effective_agency"].value_counts().to_dict()
    counts = {label: int(counts.get(label, 0)) for label in label_order}
    raw = compute_class_weights(counts, label_order, threshold)
    raw_dict = raw if raw is not None else {label: 1.0 for label in label_order}
    mean_weight = sum(raw_dict.values()) / len(raw_dict)
    return {
        "version": "v1", "created_at": datetime.now(timezone.utc).isoformat(), "label_order": label_order,
        "training_document_counts": counts, "weighted_loss_threshold": threshold,
        "weighting_applied": raw is not None, "raw_weights": raw_dict,
        "normalized_weights": {k: v / mean_weight for k, v in raw_dict.items()},
        "notes": ["Computed from eligible training-document counts, one row per document, never chunk counts."],
    }


# =========================================================================================
# Document balancing -- inverse chunk-count weighting (Checkpoint 6)
# =========================================================================================


def compute_inverse_chunk_count_weights(train_chunks_df: pd.DataFrame) -> pd.Series:
    """1 / (number of chunks that document produced) -- so every document's total training
    weight sums to exactly 1.0 regardless of how many chunks it was split into."""
    chunk_counts = train_chunks_df.groupby("document_id")["chunk_id"].transform("size")
    return 1.0 / chunk_counts


def build_document_balancing_manifest(train_chunks_df: pd.DataFrame, settings) -> dict:
    weights = compute_inverse_chunk_count_weights(train_chunks_df)
    total_documents = train_chunks_df["document_id"].nunique()
    weight_sums = train_chunks_df.assign(_weight=weights).groupby("document_id")["_weight"].sum()
    return {
        "version": "v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "total_training_documents": int(total_documents), "total_training_chunks": len(train_chunks_df),
        "weight_sum_equals_document_count": bool((weight_sums.round(9) == 1.0).all()),
        "notes": ["Corrects unequal chunk MULTIPLICITY per document -- separate from agency class weighting."],
    }


# =========================================================================================
# Combined per-chunk loss weighting (class weight x inverse chunk count)
# =========================================================================================


def compute_combined_weights(effective_agencies: list[str], chunk_counts: list[int], class_weight_by_label: dict[str, float]) -> torch.Tensor:
    weights = [class_weight_by_label[agency] * (1.0 / count) for agency, count in zip(effective_agencies, chunk_counts)]
    return torch.tensor(weights, dtype=torch.float32)


def weighted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, combined_weights: torch.Tensor) -> torch.Tensor:
    """Per-example cross-entropy, weighted, normalized by the sum of weights in the batch
    (a weighted mean) -- so loss scale doesn't depend on batch composition."""
    per_example_loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    return (per_example_loss * combined_weights).sum() / combined_weights.sum()


# =========================================================================================
# Dataset (Checkpoint 7)
# =========================================================================================


class FamilyAwareChunkDataset(Dataset):
    """Built strictly from Checkpoint 5's frozen chunk provenance: each document's full text
    is tokenized once, every chunk slices that same token-id list at its recorded
    token_start:token_end -- reproducing the frozen chunk boundaries bit-for-bit."""

    def __init__(self, chunks_df, document_texts: dict[str, str], tokenizer, max_seq_length: int, label_to_index: dict[str, int], class_weight_by_label: dict[str, float]):
        self.tokenized_docs = {doc_id: tokenizer.encode(str(text), add_special_tokens=False) for doc_id, text in document_texts.items()}
        self.rows = chunks_df.reset_index(drop=True)
        self.max_seq_length = max_seq_length
        self.label_to_index = label_to_index
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id
        self.combined_weights = compute_combined_weights(self.rows["effective_agency"].tolist(), self.rows["total_chunks"].tolist(), class_weight_by_label)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows.iloc[idx]
        document_id = str(row["document_id"])
        token_ids = self.tokenized_docs[document_id][int(row["token_start"]): int(row["token_end"])]
        ids = [self.cls_id, *token_ids, self.sep_id][: self.max_seq_length]
        attention_mask = [1] * len(ids)
        pad_length = self.max_seq_length - len(ids)
        if pad_length > 0:
            ids = ids + [self.pad_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long), "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(self.label_to_index[row["effective_agency"]], dtype=torch.long),
            "weight": self.combined_weights[idx], "document_id": document_id, "chunk_id": str(row["chunk_id"]),
        }


# =========================================================================================
# Aggregation (Checkpoint 6): combine one document's chunk-level outputs into one prediction
# =========================================================================================

AGGREGATION_METHODS = ("mean_logits", "mean_probabilities", "majority_vote", "max_confidence")


class AggregationResult:
    __slots__ = ("predicted_label", "predicted_index", "scores")

    def __init__(self, predicted_label, predicted_index, scores):
        self.predicted_label = predicted_label
        self.predicted_index = predicted_index
        self.scores = scores


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def aggregate_document(method: str, label_order: list[str], chunk_probs: list[np.ndarray] | None = None, chunk_logits: list[np.ndarray] | None = None) -> AggregationResult:
    """Dispatches to the named aggregation method. mean_probabilities (the frozen default)
    averages bounded [0,1] per-chunk probabilities so no single chunk can dominate beyond
    its 1/N share."""
    if method == "mean_logits":
        mean_logits = np.mean(np.stack(chunk_logits), axis=0)
        scores = _softmax(mean_logits)
    elif method == "mean_probabilities":
        scores = np.mean(np.stack(chunk_probs), axis=0)
    elif method == "majority_vote":
        votes = np.zeros(len(label_order), dtype=int)
        for s in chunk_probs:
            votes[int(np.argmax(s))] += 1
        summed = np.sum(np.stack(chunk_probs), axis=0)
        max_votes = votes.max()
        tied = [i for i, v in enumerate(votes) if v == max_votes]
        if len(tied) > 1:
            best_summed = max(summed[i] for i in tied)
            tied = [i for i in tied if summed[i] == best_summed]
        predicted_index = min(tied)
        return AggregationResult(label_order[predicted_index], predicted_index, votes.astype(float) / len(chunk_probs))
    elif method == "max_confidence":
        top_confidences = [float(np.max(s)) for s in chunk_probs]
        best_chunk_index = top_confidences.index(max(top_confidences))
        scores = chunk_probs[best_chunk_index]
    else:
        raise ValueError(f"Unknown aggregation method: {method!r}")

    predicted_index = int(np.argmax(scores))
    return AggregationResult(label_order[predicted_index], predicted_index, scores)


# =========================================================================================
# Training loop (Checkpoint 7)
# =========================================================================================


def set_determinism(seed: int) -> list[str]:
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
        except Exception as exc:
            captured.append(str(exc))
        captured.extend(str(w.message) for w in caught)
    return captured


def generate_chunk_level_outputs(model, tokenizer, chunks_df, document_texts: dict[str, str], label_order: list[str], max_seq_length: int, device, batch_size: int = 16):
    """Runs the model once over every chunk (position-sliced from document_texts, exactly
    reproducing the frozen chunk boundaries) and groups per-chunk probabilities/logits back
    by document_id."""
    label_to_index = {label: i for i, label in enumerate(label_order)}
    dataset = FamilyAwareChunkDataset(chunks_df, document_texts, tokenizer, max_seq_length, label_to_index, class_weight_by_label={label: 1.0 for label in label_order})
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


def train_family_aware_bert(model, tokenizer, train_chunks_df, train_document_texts, val_chunks_df, val_document_texts, label_order, class_weight_by_label, settings, progress_callback=None) -> dict:
    """The real training loop: every chunk trains as its own weighted example (no
    document-level aggregation happens here -- that's only used to score validation).
    No test-split parameter exists at all, structurally."""
    cfg = settings.family_aware.training
    max_seq_length = settings.family_aware.chunking.max_seq_length
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    label_to_index = {label: i for i, label in enumerate(label_order)}

    train_dataset = FamilyAwareChunkDataset(train_chunks_df, train_document_texts, tokenizer, max_seq_length, label_to_index, class_weight_by_label)
    val_dataset = FamilyAwareChunkDataset(val_chunks_df, val_document_texts, tokenizer, max_seq_length, label_to_index, class_weight_by_label={l: 1.0 for l in label_order})

    generator = torch.Generator()
    generator.manual_seed(cfg.random_seed)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    val_true_labels = val_chunks_df.groupby("document_id")["effective_agency"].first().to_dict()
    selection_method = cfg.checkpoint_selection_aggregation_method

    history: list[dict] = []
    best_state, best_epoch, best_val_macro_f1 = None, None, -1.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    non_deterministic_warnings: list[str] = []
    training_start = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss, total_weight = 0.0, 0.0
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
                if str(w.message) not in non_deterministic_warnings:
                    non_deterministic_warnings.append(str(w.message))
        train_loss = running_loss / total_weight

        model.eval()
        chunk_probs_by_doc: dict[str, list[np.ndarray]] = {}
        val_loss_running, val_count = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                doc_ids = batch["document_id"]
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                per_example_loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
                val_loss_running += per_example_loss.sum().item()
                val_count += len(labels)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                for doc_id, prob_vec in zip(doc_ids, probs):
                    chunk_probs_by_doc.setdefault(doc_id, []).append(prob_vec)
        val_loss = val_loss_running / val_count

        val_preds = {doc_id: aggregate_document(selection_method, label_order, chunk_probs=probs).predicted_label for doc_id, probs in chunk_probs_by_doc.items()}
        ordered_doc_ids = list(val_preds.keys())
        y_true = [val_true_labels[d] for d in ordered_doc_ids]
        y_pred = [val_preds[d] for d in ordered_doc_ids]
        val_macro_f1 = f1_score(y_true, y_pred, average="macro", labels=label_order, zero_division=0)
        val_accuracy = accuracy_score(y_true, y_pred)

        entry = {
            "epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
            "validation_document_macro_f1": float(val_macro_f1), "validation_document_accuracy": float(val_accuracy),
            "learning_rate": cfg.learning_rate, "epoch_duration_seconds": time.time() - epoch_start,
        }
        history.append(entry)
        if progress_callback:
            progress_callback(entry)

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1, best_epoch = val_macro_f1, epoch
            best_state = copy.deepcopy(model.state_dict())

    training_time = time.time() - training_start
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else None
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "history": history, "best_epoch": best_epoch, "best_validation_document_macro_f1": float(best_val_macro_f1),
        "stopping_reason": f"Completed all {cfg.max_epochs} configured epochs; selected epoch {best_epoch} by highest validation document-level macro F1 (earliest-epoch tie rule).",
        "training_time_seconds": training_time, "peak_gpu_memory_mb": peak_memory_mb, "non_deterministic_op_warnings": non_deterministic_warnings,
    }


# =========================================================================================
# BERT checkpoint save/load
# =========================================================================================


def new_family_aware_artifact_id() -> str:
    return uuid.uuid4().hex


def family_aware_artifact_dir(settings, artifact_id: str) -> Path:
    return settings.resolve_path(settings.family_aware.training.output_dir) / artifact_id


def hash_artifact_checkpoint_files(out_dir: Path) -> dict[str, str]:
    checkpoint_dir = out_dir / "checkpoint"
    hashes = {}
    for f in sorted(checkpoint_dir.glob("*")):
        if f.is_file():
            with open(f, "rb") as fh:
                hashes[f.name] = hashlib.sha256(fh.read()).hexdigest()
    return hashes


def save_family_aware_artifact(model, tokenizer, metadata: dict, settings) -> Path:
    out_dir = family_aware_artifact_dir(settings, metadata["artifact_id"])
    checkpoint_dir = out_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    _write_json(out_dir / "metadata.json", metadata)
    return out_dir


def load_family_aware_artifact_metadata(settings, artifact_id: str) -> dict:
    out_dir = family_aware_artifact_dir(settings, artifact_id)
    with open(out_dir / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


def load_family_aware_artifact(settings, artifact_id: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    metadata = load_family_aware_artifact_metadata(settings, artifact_id)
    if metadata.get("status") != "ready":
        raise RuntimeError(f"Artifact {artifact_id} is not ready (status={metadata.get('status')})")
    checkpoint_dir = family_aware_artifact_dir(settings, artifact_id) / "checkpoint"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir, num_labels=len(metadata["label_order"]))
    return model, tokenizer, metadata


def latest_ready_family_aware_artifact_id(settings) -> str | None:
    models_dir = settings.resolve_path(settings.family_aware.training.output_dir)
    if not models_dir.exists():
        return None
    candidates = []
    for artifact_dir in models_dir.iterdir():
        metadata_path = artifact_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        if metadata.get("status") == "ready":
            candidates.append((metadata.get("ready_at", metadata.get("created_at", "")), artifact_dir.name))
    if not candidates:
        return None
    return max(candidates)[1]


# =========================================================================================
# Test evaluation (Checkpoint 8): the one-time frozen test-set run
# =========================================================================================


def build_pre_test_freeze_record(settings, checkpoint_artifact_id: str, checkpoint_file_hashes: dict) -> dict:
    """Reads only Checkpoint 4-7 manifests, never test.csv -- must be saved before the first
    test-set inference call."""
    return {
        "version": "v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_artifact_id": checkpoint_artifact_id, "checkpoint_file_hashes": checkpoint_file_hashes,
        "no_changes_confirmation": "No model, prompt, retrieval, parsing, or evaluation rule will be changed based on any result produced by this evaluation.",
        "frozen": True,
    }


def evaluate_primary_test_condition(model, tokenizer, test_chunks_df, test_document_texts, true_labels_by_doc, label_order, max_seq_length, device, aggregation_method: str = "mean_probabilities") -> dict:
    """The frozen primary condition: complete_unmasked, mean_probabilities aggregation."""
    probs_by_doc, _ = generate_chunk_level_outputs(model, tokenizer, test_chunks_df, test_document_texts, label_order, max_seq_length, device)
    predictions = {doc_id: aggregate_document(aggregation_method, label_order, chunk_probs=probs).predicted_label for doc_id, probs in probs_by_doc.items()}
    document_ids = list(predictions.keys())
    y_true = [true_labels_by_doc[d] for d in document_ids]
    y_pred = [predictions[d] for d in document_ids]
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=label_order, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    return {"condition": "complete_unmasked", "document_count": len(document_ids), "document_macro_f1": macro_f1, "document_accuracy": accuracy, "predictions": predictions}


def evaluate_all_conditions(model, tokenizer, chunks_by_doc: dict[str, pd.DataFrame], masked_chunks_by_doc: dict[str, pd.DataFrame], selections_by_doc_condition: dict, true_labels_by_doc: dict[str, str], label_order: list[str], aggregation_method: str, max_seq_length: int, num_special_tokens: int, device) -> dict:
    """The real 10-condition robustness sweep -- for 'complete', every one of a document's
    frozen chunks is classified separately and mean-aggregated; partial conditions use the
    single already-selected chunk(s). No test-split parameter beyond the data explicitly
    passed in -- this function is generic over whichever split's chunks it's given."""
    from newstart_ai_mvp.data_pipeline import _CONDITION_SPECS

    def encode_text(text: str, window: int) -> tuple[list[int], list[int]]:
        token_ids = tokenizer.encode(str(text), add_special_tokens=False, truncation=True, max_length=window)
        ids = [tokenizer.cls_token_id, *token_ids, tokenizer.sep_token_id]
        attention_mask = [1] * len(ids)
        pad_length = max_seq_length - len(ids)
        if pad_length > 0:
            ids = ids + [tokenizer.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length
        return ids, attention_mask

    window = max_seq_length - num_special_tokens
    model.eval()
    results = []
    for condition_name, (region, masked) in _CONDITION_SPECS.items():
        document_ids = list(chunks_by_doc.keys())
        flat_texts, owner_index = [], []
        for i, doc_id in enumerate(document_ids):
            doc_chunks = masked_chunks_by_doc[doc_id] if masked else chunks_by_doc[doc_id]
            text_col = "masked_chunk_text" if masked else "chunk_text"
            if region == "complete":
                texts = doc_chunks.sort_values("chunk_index")[text_col].tolist()
            else:
                selection = selections_by_doc_condition[(doc_id, {"beginning": "beginning_only", "middle": "middle_only", "end": "end_only", "beginning_middle_end": "beginning_middle_end"}[region])]
                by_index = doc_chunks.set_index("chunk_index")[text_col]
                texts = [str(by_index.loc[i]) for i in selection["selected_chunk_indices"]]
            flat_texts.extend(texts)
            owner_index.extend([i] * len(texts))

        all_probs = []
        with torch.no_grad():
            for start in range(0, len(flat_texts), 16):
                batch_texts = flat_texts[start:start + 16]
                encoded = [encode_text(t, window) for t in batch_texts]
                input_ids = torch.tensor([e[0] for e in encoded], dtype=torch.long)
                attention_mask = torch.tensor([e[1] for e in encoded], dtype=torch.long)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                all_probs.extend(list(torch.softmax(logits, dim=-1).cpu().numpy()))

        probs_by_doc: dict[str, list[np.ndarray]] = {d: [] for d in document_ids}
        for owner, prob in zip(owner_index, all_probs):
            probs_by_doc[document_ids[owner]].append(prob)

        predictions = {d: aggregate_document(aggregation_method, label_order, chunk_probs=probs_by_doc[d]).predicted_label for d in document_ids}
        y_true = [true_labels_by_doc[d] for d in document_ids]
        y_pred = [predictions[d] for d in document_ids]
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=label_order, zero_division=0))
        accuracy = float(accuracy_score(y_true, y_pred))
        results.append({"condition": condition_name, "masked": masked, "region": region, "document_count": len(document_ids), "document_macro_f1": macro_f1, "document_accuracy": accuracy, "predictions": predictions})

    return {"version": "v1", "created_at": datetime.now(timezone.utc).isoformat(), "aggregation_method_used": aggregation_method, "results": results}
