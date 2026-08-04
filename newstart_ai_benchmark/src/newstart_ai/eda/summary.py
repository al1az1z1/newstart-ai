"""Descriptive EDA over the validated dataset.

Purely descriptive: this module may inform preprocessing/training choices (long-document
strategy, class weighting) but must never be used to peek at test-set performance -- it only
ever runs on the full dataset before splitting (see docs/BLUEPRINT.md Section 4).
"""

from __future__ import annotations

import pandas as pd

from newstart_ai.config.settings import Settings


def class_distribution(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Returns label, count, and percentage, sorted from most to least common."""
    label_col = settings.base.dataset.label_column
    counts = df[label_col].value_counts()
    return pd.DataFrame(
        {
            "label": counts.index,
            "count": counts.values,
            "percentage": (100 * counts.values / len(df)).round(2),
        }
    )


def missing_value_summary(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Per-column missing-value counts and percentages, for every column in the dataset."""
    missing = df.isna().sum()
    return pd.DataFrame(
        {
            "column": missing.index,
            "missing_count": missing.values,
            "missing_percentage": (100 * missing.values / len(df)).round(2),
        }
    ).sort_values("missing_count", ascending=False, ignore_index=True)


def text_length_series(df: pd.DataFrame, settings: Settings) -> pd.Series:
    """Character length of the configured text column, one value per row."""
    text_col = settings.base.dataset.text_column
    return df[text_col].fillna("").astype(str).str.len()


def text_length_summary(df: pd.DataFrame, settings: Settings) -> dict:
    """Mean/median/max/95th-percentile text length -- the numbers EDA is required to show."""
    lengths = text_length_series(df, settings)
    return {
        "mean": float(lengths.mean()),
        "median": float(lengths.median()),
        "minimum": int(lengths.min()),
        "maximum": int(lengths.max()),
        "p95": float(lengths.quantile(0.95)),
    }


def longest_documents(df: pd.DataFrame, settings: Settings, top_n: int = 10) -> pd.DataFrame:
    """The longest documents by character length, for manually judging legitimate-vs-extraction-artifact length.

    Per docs/BLUEPRINT.md Section 6, an extreme length must be inspected here, not silently
    truncated -- if it's an extraction problem (merged pages, OCR noise), it gets fixed
    upstream in 00_data_acquisition as a new dataset version.
    """
    ds_cfg = settings.base.dataset
    lengths = text_length_series(df, settings)
    out = df[[ds_cfg.id_column, ds_cfg.label_column]].copy()
    out["text_length"] = lengths
    return out.sort_values("text_length", ascending=False).head(top_n).reset_index(drop=True)
