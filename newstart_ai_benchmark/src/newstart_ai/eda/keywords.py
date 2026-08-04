"""Representative TF-IDF terms per agency, for descriptive EDA only.

Fit on the full dataset before splitting -- this is purely descriptive (helps a reader sanity
-check that classes are textually distinguishable) and is never used as a model feature or
to tune anything against test performance.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from newstart_ai.config.settings import Settings


def top_tfidf_terms_per_class(
    df: pd.DataFrame,
    settings: Settings,
    top_n: int = 15,
    max_features: int = 5000,
) -> dict[str, list[str]]:
    """Returns the top_n highest-mean-TF-IDF terms for each agency label."""
    ds_cfg = settings.base.dataset
    text_col, label_col = ds_cfg.text_column, ds_cfg.label_column

    texts = df[text_col].fillna("").astype(str)
    vectorizer = TfidfVectorizer(
        max_features=max_features, stop_words="english", lowercase=True, ngram_range=(1, 2)
    )
    matrix = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()

    result: dict[str, list[str]] = {}
    for label in sorted(df[label_col].unique()):
        mask = (df[label_col] == label).to_numpy()
        mean_scores = matrix[mask].mean(axis=0)
        mean_scores = mean_scores.A1  # flatten sparse matrix mean to a 1-D array
        top_indices = mean_scores.argsort()[::-1][:top_n]
        result[label] = [terms[i] for i in top_indices]
    return result
