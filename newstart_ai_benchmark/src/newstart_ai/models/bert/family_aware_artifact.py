"""Versioned save/load for the new family-aware chunked BERT artifact (Version 6,
Checkpoint 7).

Written under `artifacts/family_aware/models/<artifact_id>/` -- entirely separate from
`artifacts/models/` (the historical bert-mvp artifact), which this module never reads,
writes, or overwrites.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from newstart_ai.schemas.checkpoint7 import FamilyAwareModelMetadata


def new_family_aware_artifact_id() -> str:
    return uuid.uuid4().hex


def family_aware_artifact_dir(settings, artifact_id: str) -> Path:
    models_dir = settings.resolve_path(settings.family_aware.training.output_dir)
    return models_dir / artifact_id


def save_family_aware_artifact(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    metadata: FamilyAwareModelMetadata,
    settings,
) -> Path:
    out_dir = family_aware_artifact_dir(settings, metadata.artifact_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(out_dir / "checkpoint")
    tokenizer.save_pretrained(out_dir / "checkpoint")

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        f.write(metadata.model_dump_json(indent=2))

    return out_dir


def hash_artifact_checkpoint_files(out_dir: Path) -> dict[str, str]:
    checkpoint_dir = out_dir / "checkpoint"
    hashes = {}
    for path in sorted(checkpoint_dir.iterdir()):
        if path.is_file():
            with open(path, "rb") as f:
                hashes[path.name] = hashlib.sha256(f.read()).hexdigest()
    return hashes


def load_family_aware_artifact_metadata(settings, artifact_id: str) -> FamilyAwareModelMetadata:
    out_dir = family_aware_artifact_dir(settings, artifact_id)
    with open(out_dir / "metadata.json", "r", encoding="utf-8") as f:
        return FamilyAwareModelMetadata.model_validate_json(f.read())


def load_family_aware_artifact(settings, artifact_id: str):
    metadata = load_family_aware_artifact_metadata(settings, artifact_id)
    if metadata.status != "ready":
        raise RuntimeError(f"Family-aware artifact {artifact_id} has status={metadata.status!r}, not 'ready'.")
    out_dir = family_aware_artifact_dir(settings, artifact_id)
    model = AutoModelForSequenceClassification.from_pretrained(out_dir / "checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(out_dir / "checkpoint")
    return model, tokenizer, metadata


def latest_ready_family_aware_artifact_id(settings) -> str | None:
    models_dir = settings.resolve_path(settings.family_aware.training.output_dir)
    if not models_dir.exists():
        return None
    candidates: list[tuple[str, str]] = []
    for entry in models_dir.iterdir():
        metadata_path = entry / "metadata.json"
        if not metadata_path.exists():
            continue
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = FamilyAwareModelMetadata.model_validate_json(f.read())
        if metadata.status == "ready":
            candidates.append((metadata.created_at, metadata.artifact_id))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    return candidates[-1][1]
