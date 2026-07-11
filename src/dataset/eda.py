"""
Task 5 -- exploratory data analysis over documents.csv.

Why this is a plain module and not just notebook cells: keeping the actual
computations here (class balance, text-length stats, word frequencies)
means they're reusable, testable, and reproducible on demand from
data/processed/documents.csv -- notebooks/03_eda.ipynb imports these
functions rather than re-implementing them, so the two can never drift
apart.
"""

import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CSV_PATH = DATA_DIR / "processed" / "documents.csv"
FIG_DIR = DATA_DIR / "processed" / "eda_figures"

# A minimal stopword list, not a full NLP dependency -- Task 5 asks for
# "most common words by category", i.e. a sanity check, not
# publication-grade topic modeling.
STOPWORDS = {
    "the", "of", "to", "and", "a", "in", "for", "is", "on", "you", "your",
    "or", "this", "be", "if", "must", "with", "an", "as", "are", "by",
}


def load(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def class_counts(df: pd.DataFrame) -> pd.Series:
    """Documents per class -- surfaces class imbalance at a glance."""
    return df["label"].value_counts()


def agency_counts(df: pd.DataFrame) -> pd.Series:
    return df["source_agency"].value_counts()


def text_length_stats(df: pd.DataFrame) -> pd.Series:
    """Word-count distribution -- flags outliers before they skew training."""
    lengths = df["text"].str.split().str.len()
    return lengths.describe()


def duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exact-text duplicates, most often caused by an agency linking the same
    PDF from two different index pages during Task 2's crawl.
    """
    return df[df.duplicated(subset="text", keep=False)]


def top_words_by_category(df: pd.DataFrame, n: int = 15) -> dict:
    result = {}
    for label, group in df.groupby("label"):
        words = re.findall(r"[a-z]+", " ".join(group["text"]).lower())
        words = [w for w in words if w not in STOPWORDS and len(w) > 2]
        result[label] = Counter(words).most_common(n)
    return result


def save_bar_chart(series: pd.Series, title: str, filename: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ax = series.plot(kind="bar", title=title)
    out_path = FIG_DIR / filename
    ax.figure.savefig(out_path, bbox_inches="tight")
    plt.close(ax.figure)
    return out_path


def run() -> None:
    df = load()
    print("=== Documents per class ===")
    print(class_counts(df))
    print("\n=== Documents per agency ===")
    print(agency_counts(df))
    print("\n=== Text length (words) ===")
    print(text_length_stats(df))
    dupes = duplicate_rows(df)
    print(f"\n=== Duplicate text rows: {len(dupes)} ===")
    print("\n=== Top words by category ===")
    for label, words in top_words_by_category(df).items():
        print(f"{label}: {words}")

    save_bar_chart(class_counts(df), "Documents per class", "class_counts.png")
    save_bar_chart(agency_counts(df), "Documents per agency", "agency_counts.png")


if __name__ == "__main__":
    run()
