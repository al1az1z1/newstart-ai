"""Dataset for the family-aware chunked BERT, built strictly from Checkpoint 5's frozen
chunk provenance (Version 6, Checkpoint 7).

Each document's full text is tokenized exactly once; every chunk then slices that same
token-id list at its recorded `token_start:token_end` -- reproducing Checkpoint 5's chunks
bit-for-bit rather than re-deriving them from decoded `chunk_text` (a decode/re-encode round
trip is not guaranteed lossless for WordPiece).
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from newstart_ai.models.bert.weighted_loss import compute_combined_weights


class FamilyAwareChunkDataset(Dataset):
    def __init__(
        self,
        chunks_df,
        document_texts: dict[str, str],
        tokenizer,
        max_seq_length: int,
        label_to_index: dict[str, int],
        class_weight_by_label: dict[str, float],
    ):
        self.tokenized_docs = {
            doc_id: tokenizer.encode(str(text), add_special_tokens=False) for doc_id, text in document_texts.items()
        }
        self.rows = chunks_df.reset_index(drop=True)
        self.max_seq_length = max_seq_length
        self.label_to_index = label_to_index
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id

        self.combined_weights = compute_combined_weights(
            self.rows["effective_agency"].tolist(),
            self.rows["total_chunks"].tolist(),
            class_weight_by_label,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows.iloc[idx]
        document_id = str(row["document_id"])
        token_ids = self.tokenized_docs[document_id][int(row["token_start"]) : int(row["token_end"])]

        ids = [self.cls_id, *token_ids, self.sep_id][: self.max_seq_length]
        attention_mask = [1] * len(ids)
        pad_length = self.max_seq_length - len(ids)
        if pad_length > 0:
            ids = ids + [self.pad_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(self.label_to_index[row["effective_agency"]], dtype=torch.long),
            "weight": self.combined_weights[idx],
            "document_id": document_id,
            "chunk_id": str(row["chunk_id"]),
        }
