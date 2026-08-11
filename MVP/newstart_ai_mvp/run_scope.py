"""Redirects frozen-artifact writes to a fresh MVP/runs/<run_id>/ directory for the duration
of one CLI invocation, so an explicit --run/--run-training/--run-api/--rebuild-* stage can
never overwrite the submitted experiment's frozen artifacts.

Every writer in newstart_ai -- config-driven or hardcoded-literal -- routes through
Settings.resolve_path(relative_path). Patching resolve_path itself (rather than overriding
individual Settings fields) is the only redirection that also covers the functions that
hardcode a frozen-artifact path string directly (e.g. save_family_audit, which writes to a
literal "artifacts/family_aware/reports"/"manifests" regardless of any per-field override,
and raises FileExistsError if its versioned overrides file already exists) and the Chroma
index builder (build_family_aware_corpus_index), which calls client.delete_collection() on
the persisted collection before rebuilding it -- pointed at the real persist_dir, that call
would delete the frozen vector store outright.
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from newstart_ai_mvp.config import Settings

# Every relative path a real save_*/build_*_index function might ask Settings.resolve_path
# to resolve, that must never be allowed to land in the real frozen locations.
FROZEN_OUTPUT_PREFIXES = (
    "artifacts/family_aware/manifests",
    "artifacts/family_aware/reports",
    "artifacts/family_aware/models",
    "artifacts/family_aware/embedding_cache",
    "artifacts/family_aware/llm_eval_cache",
    "artifacts/family_aware/vector_stores",
    "data/family_aware_splits",
    "data/family_aware_chunks",
    "data/family_aware_masked",
    "data/family_aware_conditions",
)


def new_run_id() -> str:
    """A sortable, filesystem-safe run identifier, e.g. '20260811T193000Z'."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_root(run_id: str) -> Path:
    """MVP/runs/<run_id> -- resolved from this file's own location, never from the caller's
    current working directory, so it's stable regardless of where a CLI command is invoked
    from."""
    mvp_root = Path(__file__).resolve().parents[1]
    return mvp_root / "runs" / run_id


def _is_frozen_output(relative_path: str) -> bool:
    normalized = relative_path.replace("\\", "/")
    return any(
        normalized == prefix or normalized.startswith(prefix + "/") for prefix in FROZEN_OUTPUT_PREFIXES
    )


@contextlib.contextmanager
def redirect_frozen_outputs(run_id: str):
    """While active, any Settings.resolve_path(rel) whose `rel` starts with a known
    frozen-output prefix resolves under MVP/runs/<run_id>/<rel> instead of the real project
    root. Every other path (dataset input, prompts, configs, tokenizer cache) resolves
    exactly as it would without this context manager -- only writes are ever redirected,
    reads of upstream frozen inputs still work normally unless the caller also passes
    --input-run-id to read a previous run's own output instead.

    Yields the resolved run root (MVP/runs/<run_id>) for convenience."""
    real_resolve_path = Settings.resolve_path
    destination_root = run_root(run_id)

    def scoped_resolve_path(self: Settings, relative_path: str) -> Path:
        if _is_frozen_output(relative_path):
            return destination_root / relative_path
        return real_resolve_path(self, relative_path)

    with patch.object(Settings, "resolve_path", scoped_resolve_path):
        yield destination_root
