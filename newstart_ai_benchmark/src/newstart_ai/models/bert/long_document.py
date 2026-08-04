"""Configurable long-document strategies for BERT.

BERT accepts at most a small fixed number of tokens (base_model max position, typically 512),
while many documents in this dataset are far longer (up to ~640k characters). Naive
first-512-token truncation silently discards most of a long document; this module implements
both required MVP strategies (docs/BLUEPRINT.md Section 6) so the choice is made explicitly
and compared on validation macro F1, never assumed.

Every document produces at least one, and at most `max_chunks`, deterministic token windows.
Chunk-level predictions are aggregated (mean probability) into one document-level prediction
by the classifier -- this module only decides which token windows to use.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from newstart_ai.config.settings import Settings

# Reserve room for [CLS]/[SEP] special tokens added when a chunk is encoded for the model.
NUM_SPECIAL_TOKENS = 2


class TokenChunk:
    """A single deterministic token window plus its position, for traceability."""

    __slots__ = ("token_ids",)

    def __init__(self, token_ids: list[int]):
        self.token_ids = token_ids

    def __len__(self) -> int:
        return len(self.token_ids)


class LongDocumentStrategy(Protocol):
    def chunk(self, token_ids: list[int]) -> list[TokenChunk]:
        """Splits one document's raw token ids into 1+ deterministic windows."""
        ...

    def aggregate(self, chunk_probabilities: list[np.ndarray]) -> np.ndarray:
        """Combines one probability vector per chunk into a single document-level vector."""
        ...


class FirstNTokensStrategy:
    """Baseline: use only the first `max_tokens` tokens of the document."""

    name = "first_512"

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.window = max_tokens - NUM_SPECIAL_TOKENS

    def chunk(self, token_ids: list[int]) -> list[TokenChunk]:
        return [TokenChunk(token_ids[: self.window])]

    def aggregate(self, chunk_probabilities: list[np.ndarray]) -> np.ndarray:
        # Only ever one chunk, so aggregation is the identity.
        return chunk_probabilities[0]


class BeginningMiddleEndStrategy:
    """Deterministically samples up to `max_chunks` fixed windows: beginning, middle, end.

    The middle window is placed to overlap the beginning window by exactly
    `chunk_overlap_tokens` (when the document is long enough), giving the model some shared
    context between windows rather than three disjoint, context-free fragments. This keeps
    the number of training/inference inputs per document bounded and constant, regardless of
    how long the source document is.
    """

    name = "beginning_middle_end"

    def __init__(self, max_tokens: int, max_chunks: int, chunk_overlap_tokens: int):
        self.max_tokens = max_tokens
        self.max_chunks = max_chunks
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.window = max_tokens - NUM_SPECIAL_TOKENS

    def chunk(self, token_ids: list[int]) -> list[TokenChunk]:
        total = len(token_ids)
        window = self.window

        if total <= window:
            return [TokenChunk(token_ids)]

        begin_range = (0, window)
        end_start = max(0, total - window)
        end_range = (end_start, total)

        ranges = [begin_range]
        if self.max_chunks >= 3:
            mid_start = max(0, begin_range[1] - self.chunk_overlap_tokens)
            mid_start = min(mid_start, end_start)
            mid_range = (mid_start, min(mid_start + window, total))
            if mid_range not in ranges:
                ranges.append(mid_range)
        if end_range not in ranges:
            ranges.append(end_range)

        ranges = ranges[: self.max_chunks]
        return [TokenChunk(token_ids[start:end]) for start, end in ranges]

    def aggregate(self, chunk_probabilities: list[np.ndarray]) -> np.ndarray:
        # Documented aggregation rule: mean probability across chunks.
        return np.mean(np.stack(chunk_probabilities), axis=0)


def build_long_document_strategy(settings: Settings, override: str | None = None) -> LongDocumentStrategy:
    """Builds the configured strategy from configs/base.yaml -- never hard-coded in callers.

    `override` lets 05_bert_evaluation compare both strategies without editing config files.
    """
    cfg = settings.base.long_document_strategy
    name = override or cfg.default

    if name == "first_512":
        return FirstNTokensStrategy(max_tokens=cfg.max_tokens)
    if name == "beginning_middle_end":
        return BeginningMiddleEndStrategy(
            max_tokens=cfg.max_tokens,
            max_chunks=cfg.beginning_middle_end.max_chunks,
            chunk_overlap_tokens=cfg.beginning_middle_end.chunk_overlap_tokens,
        )
    raise ValueError(f"Unknown long-document strategy: {name!r}")
