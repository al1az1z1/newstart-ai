"""Plotting helpers for 02_exploratory_data_analysis.ipynb.

Each function returns the Matplotlib Axes it drew on so notebooks can display or further
annotate the figure -- no plot is saved to disk here.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from newstart_ai.config.settings import Settings
from newstart_ai.eda.summary import text_length_series


def plot_class_distribution(df: pd.DataFrame, settings: Settings, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    label_col = settings.base.dataset.label_column
    df[label_col].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Class distribution")
    ax.set_xlabel("Agency")
    ax.set_ylabel("Document count")
    return ax


def plot_length_histogram(df: pd.DataFrame, settings: Settings, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    lengths = text_length_series(df, settings)
    ax.hist(lengths, bins=40)
    ax.set_title("Text length distribution")
    ax.set_xlabel("Character length")
    ax.set_ylabel("Document count")
    return ax


def plot_length_boxplot(df: pd.DataFrame, settings: Settings, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    label_col = settings.base.dataset.label_column
    lengths = text_length_series(df, settings)
    grouped = [lengths[df[label_col] == label] for label in sorted(df[label_col].unique())]
    ax.boxplot(grouped, tick_labels=sorted(df[label_col].unique()))
    ax.set_title("Text length by agency")
    ax.set_ylabel("Character length")
    return ax
