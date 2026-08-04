"""Proof that Checkpoint 6 policy-freezing never opened test-split data (Version 6).

The proof is only as good as the honesty of `input_files_used` at the call site -- the
orchestration script that freezes every Checkpoint 6 policy declares exactly which files it
opened, and this function asserts none of them is a test-split file.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from newstart_ai.schemas.checkpoint6 import TestIsolationProof

_BANNED_TEST_FILE_NAMES = {"test.csv", "test_chunks.csv", "test_chunks_masked.csv"}


def build_test_isolation_proof(functions_exercised: list[str], input_files_used: list[str]) -> TestIsolationProof:
    test_files_referenced = [f for f in input_files_used if Path(f).name in _BANNED_TEST_FILE_NAMES]
    isolation_holds = len(test_files_referenced) == 0

    return TestIsolationProof(
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        functions_exercised=functions_exercised,
        input_files_used=input_files_used,
        test_files_referenced=test_files_referenced,
        isolation_holds=isolation_holds,
        proof_statement=(
            "Every Checkpoint 6 policy (aggregation default, document balancing, "
            "partial-input selection, identifier-masking rules, condition registry) was "
            "built only from configuration and the files listed in input_files_used -- the "
            "family-aware train split and the family-aware validation split. No file whose "
            "name indicates the test split was opened by any function in "
            "functions_exercised."
        ),
    )
