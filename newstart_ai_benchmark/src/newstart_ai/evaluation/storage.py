"""Persists row-level predictions and metrics reports to artifacts/, so summary notebooks
(09, 10) and the demo app read stored results rather than recomputing anything.
"""

from __future__ import annotations

import json
from pathlib import Path

from newstart_ai.config.settings import Settings
from newstart_ai.schemas.classification import ClassificationResult
from newstart_ai.schemas.evaluation import MetricsReport


def _predictions_path(settings: Settings, method: str, split: str) -> Path:
    return settings.resolve_path("artifacts/predictions") / f"{method}_{split}.json"


def _metrics_path(settings: Settings, method: str, split: str) -> Path:
    return settings.resolve_path("artifacts/reports") / f"{method}_{split}_metrics.json"


def save_predictions(
    results: list[ClassificationResult], method: str, split: str, settings: Settings
) -> Path:
    path = _predictions_path(settings, method, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)
    return path


def load_predictions(method: str, split: str, settings: Settings) -> list[ClassificationResult]:
    path = _predictions_path(settings, method, split)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [ClassificationResult.model_validate(r) for r in raw]


def save_metrics_report(report: MetricsReport, settings: Settings) -> Path:
    path = _metrics_path(settings, report.method, report.split)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))
    return path


def load_metrics_report(method: str, split: str, settings: Settings) -> MetricsReport:
    path = _metrics_path(settings, method, split)
    with open(path, "r", encoding="utf-8") as f:
        return MetricsReport.model_validate_json(f.read())
