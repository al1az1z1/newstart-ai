"""Stable content fingerprint for the dataset.

Recorded in the split manifest and every experiment's reproducibility record so a later run
can detect whether final_dataset.csv has actually changed since the frozen split was created.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from newstart_ai.config.settings import Settings


def dataset_fingerprint(df: pd.DataFrame, settings: Settings) -> str:
    """Hashes (id, label, text) for every row, sorted by id -- same content, same fingerprint
    regardless of row order in the source file."""
    ds_cfg = settings.base.dataset
    id_col, text_col, label_col = ds_cfg.id_column, ds_cfg.text_column, ds_cfg.label_column

    ordered = df[[id_col, label_col, text_col]].astype(str).sort_values(id_col)
    hasher = hashlib.sha256()
    for _, row in ordered.iterrows():
        hasher.update(row[id_col].encode("utf-8"))
        hasher.update(b"|")
        hasher.update(row[label_col].encode("utf-8"))
        hasher.update(b"|")
        hasher.update(row[text_col].encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()
