"""Non-destructive dataset validation.

Per docs/BLUEPRINT.md and NewStart_AI_MVP.md Section 3: this module only reports problems.
It never rewrites, drops rows from, or otherwise cleans final_dataset.csv. If a real fix is
needed, it happens upstream in notebooks/00_data_acquisition as a new documented dataset
version -- not silently here.
"""

from __future__ import annotations

import pandas as pd

from newstart_ai.config.settings import Settings
from newstart_ai.schemas.validation import ClassCount, LengthStats, ValidationReport

# Smallest per-class row count for which a 64/16/20 stratified split still leaves at least
# one example of that class in every split. Below this, stratification is not reliable.
MIN_ROWS_PER_CLASS_FOR_SPLIT = 5

# Majority:minority training-set ratio at or above which the dataset is flagged as
# meaningfully imbalanced (matches configs/bert.yaml: imbalance.weighted_loss_threshold).
IMBALANCE_WARNING_RATIO = 4.0


def load_dataset(settings: Settings) -> pd.DataFrame:
    """Loads the fixed dataset exactly as configured in configs/base.yaml -- no cleaning."""
    path = settings.resolve_path(settings.base.dataset.path)
    return pd.read_csv(path, encoding_errors="replace")


def validate_dataset(df: pd.DataFrame, settings: Settings) -> ValidationReport:
    """Runs every non-destructive check from NewStart_AI_MVP.md Section 3 and returns a report.

    Callers (e.g. 01_dataset_validation.ipynb) are responsible for deciding whether to stop
    based on `report.has_critical_errors` -- this function only observes and describes.
    """
    ds_cfg = settings.base.dataset
    id_col, text_col, label_col = ds_cfg.id_column, ds_cfg.text_column, ds_cfg.label_column
    allowed_labels = set(settings.base.labels)

    required_columns = [id_col, text_col, label_col]
    missing_columns = [c for c in required_columns if c not in df.columns]

    if missing_columns:
        return ValidationReport(
            row_count=len(df),
            required_columns_present=False,
            missing_columns=missing_columns,
            document_id_column_unique=False,
            duplicate_document_id_count=0,
            empty_text_count=0,
            duplicate_text_count=0,
            valid_labels=False,
            invalid_label_values=[],
            class_counts=[],
            minimum_class_count=0,
            imbalance_ratio=0.0,
            stratified_split_feasible=False,
            stratified_split_blockers=["Required columns are missing; cannot check further."],
            text_length=LengthStats(mean=0, median=0, minimum=0, maximum=0, p95=0),
            warnings=[f"Missing required columns: {missing_columns}"],
            recommendations=[
                "Add the missing columns upstream (00_data_acquisition) before re-validating."
            ],
        )

    row_count = len(df)
    warnings: list[str] = []
    recommendations: list[str] = []

    id_series = df[id_col].astype(str)
    duplicate_document_id_count = int(id_series.duplicated().sum())
    document_id_column_unique = duplicate_document_id_count == 0

    text_series = df[text_col].fillna("").astype(str)
    empty_text_count = int((text_series.str.strip() == "").sum())
    duplicate_text_count = int(text_series.duplicated().sum())

    label_series = df[label_col].astype(str)
    invalid_mask = ~label_series.isin(allowed_labels)
    invalid_label_values = sorted(label_series[invalid_mask].unique().tolist())
    valid_labels = len(invalid_label_values) == 0

    counts = label_series.value_counts()
    class_counts = [
        ClassCount(label=str(label), count=int(count), percentage=round(100 * count / row_count, 2))
        for label, count in counts.items()
    ]
    minimum_class_count = int(counts.min()) if len(counts) else 0
    imbalance_ratio = float(counts.max() / counts.min()) if len(counts) and counts.min() > 0 else 0.0

    stratified_split_blockers: list[str] = []
    for label, count in counts.items():
        if count < MIN_ROWS_PER_CLASS_FOR_SPLIT:
            stratified_split_blockers.append(
                f"Class '{label}' has only {count} rows (< {MIN_ROWS_PER_CLASS_FOR_SPLIT}); "
                "its test slice will be extremely small and noisy."
            )
    stratified_split_feasible = minimum_class_count >= 2  # absolute floor: >=1 per split

    lengths = text_series.str.len()
    text_length = LengthStats(
        mean=float(lengths.mean()),
        median=float(lengths.median()),
        minimum=int(lengths.min()),
        maximum=int(lengths.max()),
        p95=float(lengths.quantile(0.95)),
    )

    if duplicate_document_id_count:
        warnings.append(f"{duplicate_document_id_count} duplicate '{id_col}' values found.")
    if empty_text_count:
        warnings.append(f"{empty_text_count} rows have empty or whitespace-only text.")
    if duplicate_text_count:
        warnings.append(f"{duplicate_text_count} rows have exactly duplicated text.")
    if not valid_labels:
        warnings.append(
            f"Found label values outside the configured set {sorted(allowed_labels)}: "
            f"{invalid_label_values}"
        )
    if stratified_split_blockers:
        warnings.extend(stratified_split_blockers)
    if imbalance_ratio >= IMBALANCE_WARNING_RATIO:
        warnings.append(
            f"Class imbalance ratio is {imbalance_ratio:.1f}x (majority/minority) -- "
            "class-weighted loss will apply during BERT training."
        )
        recommendations.append(
            "Report macro F1 (not accuracy) as the primary metric, and flag the smallest "
            "class's per-class metrics as statistically uncertain given its small test slice."
        )
    if text_length.maximum > 20 * text_length.median:
        recommendations.append(
            "Inspect the longest records in 02_exploratory_data_analysis before choosing "
            "the long-document strategy -- extreme lengths may be legitimate long documents "
            "or upstream extraction artifacts (merged pages, OCR noise)."
        )

    return ValidationReport(
        row_count=row_count,
        required_columns_present=True,
        missing_columns=[],
        document_id_column_unique=document_id_column_unique,
        duplicate_document_id_count=duplicate_document_id_count,
        empty_text_count=empty_text_count,
        duplicate_text_count=duplicate_text_count,
        valid_labels=valid_labels,
        invalid_label_values=invalid_label_values,
        class_counts=class_counts,
        minimum_class_count=minimum_class_count,
        imbalance_ratio=imbalance_ratio,
        stratified_split_feasible=stratified_split_feasible,
        stratified_split_blockers=stratified_split_blockers,
        text_length=text_length,
        warnings=warnings,
        recommendations=recommendations,
    )
