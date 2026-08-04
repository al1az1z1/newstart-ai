"""Fine-tuned BERT classifier for agency routing.

The base checkpoint always comes from configs/bert.yaml (settings.bert.base_model) -- this
module never hard-codes a model name, so swapping the checkpoint is a config change only.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from newstart_ai.config.settings import Settings
from newstart_ai.models.bert.dataset import ChunkedTextDataset, chunk_to_model_inputs, tokenize_raw
from newstart_ai.models.bert.imbalance import compute_class_weights
from newstart_ai.models.bert.long_document import LongDocumentStrategy, build_long_document_strategy


class BERTClassifier:
    def __init__(self, settings: Settings, long_document_strategy: LongDocumentStrategy | None = None):
        self.settings = settings
        self.label_order = list(settings.base.labels)
        self.max_tokens = settings.base.long_document_strategy.max_tokens
        self.long_document_strategy = long_document_strategy or build_long_document_strategy(settings)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model = settings.bert.base_model  # never hard-coded -- always read from config
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=len(self.label_order)
        ).to(self.device)

        self.class_weights: np.ndarray | None = None

    def fit(
        self,
        train_df,
        val_df,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> dict:
        """Trains on train_df, selecting the best epoch by validation macro F1 (val_df).

        Class weights are computed from train_df only. Returns per-epoch history plus the
        best validation macro F1 achieved -- used both for progress reporting and for
        comparing long-document strategies (docs/BLUEPRINT.md Section 6).
        """
        ds_cfg = self.settings.base.dataset
        bert_cfg = self.settings.bert
        text_col, label_col = ds_cfg.text_column, ds_cfg.label_column

        train_label_counts = train_df[label_col].value_counts().to_dict()
        self.class_weights = compute_class_weights(
            train_label_counts, self.label_order, bert_cfg.imbalance.weighted_loss_threshold
        )
        weight_tensor = (
            torch.tensor(self.class_weights, dtype=torch.float32).to(self.device)
            if self.class_weights is not None
            else None
        )

        train_dataset = ChunkedTextDataset(
            texts=train_df[text_col].tolist(),
            labels=train_df[label_col].tolist(),
            tokenizer=self.tokenizer,
            strategy=self.long_document_strategy,
            max_tokens=self.max_tokens,
            label_order=self.label_order,
        )
        train_loader = DataLoader(train_dataset, batch_size=bert_cfg.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=bert_cfg.learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)

        val_texts = val_df[text_col].tolist()
        val_labels = val_df[label_col].tolist()

        best_state = None
        best_val_macro_f1 = -1.0
        history: list[dict] = []

        for epoch in range(1, bert_cfg.max_epochs + 1):
            self.model.train()
            running_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(labels)
            train_loss = running_loss / len(train_dataset)

            val_probs = self.predict_proba(val_texts)
            val_preds = [self.label_order[int(np.argmax(p))] for p in val_probs]
            val_loss = _mean_negative_log_likelihood(val_labels, val_probs, self.label_order)
            val_accuracy = accuracy_score(val_labels, val_preds)
            val_macro_f1 = f1_score(
                val_labels, val_preds, average="macro", labels=self.label_order, zero_division=0
            )

            entry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_accuracy": val_accuracy,
                "validation_macro_f1": val_macro_f1,
                "learning_rate": bert_cfg.learning_rate,
            }
            history.append(entry)
            (progress_callback or _default_progress_printer)(entry)

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                best_state = copy.deepcopy(self.model.state_dict())

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {"history": history, "best_validation_macro_f1": best_val_macro_f1}

    def predict_proba(self, texts: list[str], batch_size: int = 16) -> list[np.ndarray]:
        """Returns one aggregated probability vector per document, indexed by label_order."""
        self.model.eval()

        # Flatten every document's chunks into one batchable list, tracking which document
        # each chunk belongs to so results can be grouped back and aggregated afterward.
        flat_chunks = []
        owner_doc_index = []
        for doc_index, text in enumerate(texts):
            token_ids = tokenize_raw(text, self.tokenizer)
            for chunk in self.long_document_strategy.chunk(token_ids):
                flat_chunks.append(chunk)
                owner_doc_index.append(doc_index)

        chunk_probs_by_doc: list[list[np.ndarray]] = [[] for _ in texts]

        with torch.no_grad():
            for start in range(0, len(flat_chunks), batch_size):
                batch_chunks = flat_chunks[start : start + batch_size]
                batch_owners = owner_doc_index[start : start + batch_size]

                encoded = [chunk_to_model_inputs(c, self.tokenizer, self.max_tokens) for c in batch_chunks]
                input_ids = torch.stack([e["input_ids"] for e in encoded]).to(self.device)
                attention_mask = torch.stack([e["attention_mask"] for e in encoded]).to(self.device)

                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

                for owner, prob in zip(batch_owners, probs):
                    chunk_probs_by_doc[owner].append(prob)

        return [self.long_document_strategy.aggregate(chunk_probs) for chunk_probs in chunk_probs_by_doc]

    def predict(self, texts: list[str]) -> list[str]:
        probs = self.predict_proba(texts)
        return [self.label_order[int(np.argmax(p))] for p in probs]


def _mean_negative_log_likelihood(
    true_labels: list[str], probs: list[np.ndarray], label_order: list[str], eps: float = 1e-9
) -> float:
    label_to_index = {label: i for i, label in enumerate(label_order)}
    losses = [-np.log(max(p[label_to_index[label]], eps)) for label, p in zip(true_labels, probs)]
    return float(np.mean(losses))


def _default_progress_printer(entry: dict) -> None:
    print(
        f"epoch {entry['epoch']}  train_loss={entry['train_loss']:.4f}  "
        f"val_loss={entry['validation_loss']:.4f}  "
        f"val_acc={entry['validation_accuracy']:.4f}  "
        f"val_macro_f1={entry['validation_macro_f1']:.4f}"
    )
