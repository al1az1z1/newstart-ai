"""Versioned BERT artifact save/load.

An artifact bundles everything needed to reproduce or reuse a trained model: the checkpoint,
tokenizer, label mapping, training configuration, dataset fingerprint, and metrics. The
`artifact_id` (not the display name) is the only trusted key/path component -- matching the
naming rule from the original larger design, still good practice even without multi-user
artifact selection in this MVP.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from newstart_ai.config.settings import Settings


class BertArtifactMetadata(BaseModel):
    artifact_id: str
    display_name: str = "bert-mvp"
    base_model: str
    version: int = 1
    label_order: list[str]
    dataset_fingerprint: str
    long_document_strategy: str
    training_config: dict = Field(default_factory=dict)
    validation_metrics: dict = Field(default_factory=dict)
    test_metrics: dict = Field(default_factory=dict)
    status: Literal["training", "ready", "failed"] = "training"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ready_at: str | None = None


def new_artifact_id() -> str:
    return uuid.uuid4().hex


def artifact_dir(settings: Settings, artifact_id: str) -> Path:
    models_dir = settings.resolve_path("artifacts/models")
    return models_dir / artifact_id


def save_artifact(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    metadata: BertArtifactMetadata,
    settings: Settings,
) -> Path:
    """Saves the model, tokenizer, and metadata under artifacts/models/<artifact_id>/.

    Only call this once training and (if applicable) validation-set evaluation have
    completed successfully -- set metadata.status = "ready" before saving so downstream code
    never picks up a partially-trained artifact.
    """
    out_dir = artifact_dir(settings, metadata.artifact_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(out_dir / "checkpoint")
    tokenizer.save_pretrained(out_dir / "checkpoint")

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        f.write(metadata.model_dump_json(indent=2))

    return out_dir


def load_artifact_metadata(settings: Settings, artifact_id: str) -> BertArtifactMetadata:
    out_dir = artifact_dir(settings, artifact_id)
    with open(out_dir / "metadata.json", "r", encoding="utf-8") as f:
        return BertArtifactMetadata.model_validate_json(f.read())


def load_artifact(
    settings: Settings, artifact_id: str
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer, BertArtifactMetadata]:
    """Loads a previously saved artifact. Raises if it isn't marked READY."""
    metadata = load_artifact_metadata(settings, artifact_id)
    if metadata.status != "ready":
        raise RuntimeError(
            f"Artifact {artifact_id} has status={metadata.status!r}, not 'ready'. "
            "Failed or incomplete artifacts must not be loaded for evaluation or inference."
        )
    out_dir = artifact_dir(settings, artifact_id)
    model = AutoModelForSequenceClassification.from_pretrained(out_dir / "checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(out_dir / "checkpoint")
    return model, tokenizer, metadata


def latest_ready_artifact_id(settings: Settings) -> str | None:
    """Finds the most recently created READY artifact -- used by the demo app and
    05_bert_evaluation so callers never need to hard-code an artifact_id."""
    models_dir = settings.resolve_path("artifacts/models")
    if not models_dir.exists():
        return None

    candidates: list[tuple[str, str]] = []  # (created_at, artifact_id)
    for entry in models_dir.iterdir():
        metadata_path = entry / "metadata.json"
        if not metadata_path.exists():
            continue
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = BertArtifactMetadata.model_validate_json(f.read())
        if metadata.status == "ready":
            candidates.append((metadata.created_at, metadata.artifact_id))

    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    return candidates[-1][1]
