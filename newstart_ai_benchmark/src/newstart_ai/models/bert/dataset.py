"""Tokenization helpers shared by training and prediction.

Chunk-level tokenization is centralized here so training (ChunkedTextDataset) and prediction
(BERTClassifier.predict_proba) always encode chunks identically.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from newstart_ai.models.bert.long_document import LongDocumentStrategy, TokenChunk

# Raw tokens considered before chunking. Bounds tokenization cost for extreme outliers (the
# dataset's longest document is ~640k characters) without affecting which windows
# first_512/beginning_middle_end select, since both only ever look at the very beginning,
# middle, and/or end of the token sequence -- well within this cap for every document here.
MAX_RAW_TOKENS_CONSIDERED = 20000


def tokenize_raw(text: str, tokenizer) -> list[int]:
    """Tokenizes a full document to raw token ids, without special tokens or padding."""
    return tokenizer.encode(
        text, add_special_tokens=False, truncation=True, max_length=MAX_RAW_TOKENS_CONSIDERED
    )


def chunk_to_model_inputs(chunk: TokenChunk, tokenizer, max_tokens: int) -> dict[str, torch.Tensor]:
    """Wraps one token window with [CLS]/[SEP] and pads it to max_tokens.

    Built directly from cls_token_id/sep_token_id/pad_token_id rather than a higher-level
    "prepare for model" helper -- those attributes are stable across tokenizer library
    versions, while helper method availability has not been.
    """
    ids = [tokenizer.cls_token_id, *chunk.token_ids, tokenizer.sep_token_id][:max_tokens]
    attention_mask = [1] * len(ids)

    pad_length = max_tokens - len(ids)
    if pad_length > 0:
        ids = ids + [tokenizer.pad_token_id] * pad_length
        attention_mask = attention_mask + [0] * pad_length

    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


class ChunkedTextDataset(Dataset):
    """Expands each document into 1+ chunks (per the configured long-document strategy),
    each chunk inheriting its document's label. Training-only -- prediction groups chunks
    back per document instead (see BERTClassifier.predict_proba)."""

    def __init__(
        self,
        texts: list[str],
        labels: list[str],
        tokenizer,
        strategy: LongDocumentStrategy,
        max_tokens: int,
        label_order: list[str],
    ):
        label_to_index = {label: i for i, label in enumerate(label_order)}
        self.examples: list[dict] = []
        for text, label in zip(texts, labels):
            token_ids = tokenize_raw(text, tokenizer)
            for chunk in strategy.chunk(token_ids):
                model_inputs = chunk_to_model_inputs(chunk, tokenizer, max_tokens)
                self.examples.append(
                    {
                        "input_ids": model_inputs["input_ids"],
                        "attention_mask": model_inputs["attention_mask"],
                        "label": torch.tensor(label_to_index[label], dtype=torch.long),
                    }
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]
