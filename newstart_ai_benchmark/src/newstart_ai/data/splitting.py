"""The one frozen, reproducible stratified train/validation/test split.

Created exactly once (03_reproducible_splitting.ipynb) and then treated as immutable by
every other notebook -- see docs/BLUEPRINT.md Section 4 for the leakage rules this protects.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from newstart_ai.config.settings import Settings
from newstart_ai.data.fingerprinting import dataset_fingerprint
from newstart_ai.schemas.splitting import SplitClassDistribution, SplitManifest


def create_stratified_split(
    df: pd.DataFrame, settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitManifest]:
    """Splits df into (train, validation, test) using the ratios and seed in configs/base.yaml.

    Returns the three DataFrames plus a SplitManifest recording exactly which document_ids
    went where, so leakage can be checked and the split reproduced later.
    """
    ds_cfg = settings.base.dataset
    split_cfg = settings.base.split
    id_col, label_col = ds_cfg.id_column, ds_cfg.label_column
    seed = split_cfg.random_seed

    train_val_df, test_df = train_test_split(
        df,
        test_size=split_cfg.test,
        stratify=df[label_col],
        random_state=seed,
    )
    # validation's share of the remaining (train+validation) portion
    relative_val_fraction = split_cfg.validation / (split_cfg.train + split_cfg.validation)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_fraction,
        stratify=train_val_df[label_col],
        random_state=seed,
    )

    fingerprint = dataset_fingerprint(df, settings)

    class_distribution: list[SplitClassDistribution] = []
    for split_name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        for label, count in split_df[label_col].value_counts().items():
            class_distribution.append(
                SplitClassDistribution(split=split_name, label=str(label), count=int(count))
            )

    manifest = SplitManifest(
        random_seed=seed,
        dataset_fingerprint=fingerprint,
        created_at=datetime.now(timezone.utc).isoformat(),
        train_row_count=len(train_df),
        validation_row_count=len(val_df),
        test_row_count=len(test_df),
        train_document_ids=train_df[id_col].astype(str).tolist(),
        validation_document_ids=val_df[id_col].astype(str).tolist(),
        test_document_ids=test_df[id_col].astype(str).tolist(),
        class_distribution=class_distribution,
    )
    manifest.assert_no_overlap()

    return train_df, val_df, test_df, manifest


def save_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    manifest: SplitManifest,
    settings: Settings,
) -> Path:
    """Writes train.csv, validation.csv, test.csv, and split_manifest.json to data/splits/."""
    output_dir = settings.resolve_path(settings.base.split.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    with open(output_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))

    return output_dir


def load_split(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitManifest]:
    """Loads a previously saved split -- used by every notebook after 03 so the split is
    created exactly once and only ever read afterward."""
    output_dir = settings.resolve_path(settings.base.split.output_dir)

    train_df = pd.read_csv(output_dir / "train.csv", encoding_errors="replace")
    val_df = pd.read_csv(output_dir / "validation.csv", encoding_errors="replace")
    test_df = pd.read_csv(output_dir / "test.csv", encoding_errors="replace")

    with open(output_dir / "split_manifest.json", "r", encoding="utf-8") as f:
        manifest = SplitManifest.model_validate_json(f.read())

    return train_df, val_df, test_df, manifest
