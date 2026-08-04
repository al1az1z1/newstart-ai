"""Result schema for the one frozen reproducible train/validation/test split.

See src/newstart_ai/data/splitting.py and fingerprinting.py, and
docs/BLUEPRINT.md Section 4 for the leakage-protection rules this manifest supports.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SplitClassDistribution(BaseModel):
    split: str  # "train" | "validation" | "test"
    label: str
    count: int


class SplitManifest(BaseModel):
    random_seed: int
    dataset_fingerprint: str
    created_at: str  # ISO 8601 timestamp

    train_row_count: int
    validation_row_count: int
    test_row_count: int

    train_document_ids: list[str]
    validation_document_ids: list[str]
    test_document_ids: list[str]

    class_distribution: list[SplitClassDistribution] = Field(default_factory=list)

    def assert_no_overlap(self) -> None:
        """Raises if any document_id appears in more than one split.

        This is the concrete leakage check every downstream notebook relies on --
        BERT training, RAG index creation, and prompt development must never see a
        test document.
        """
        train = set(self.train_document_ids)
        validation = set(self.validation_document_ids)
        test = set(self.test_document_ids)
        overlaps = {
            "train/validation": train & validation,
            "train/test": train & test,
            "validation/test": validation & test,
        }
        leaking = {name: ids for name, ids in overlaps.items() if ids}
        if leaking:
            raise ValueError(f"Data leakage detected between splits: {leaking}")
