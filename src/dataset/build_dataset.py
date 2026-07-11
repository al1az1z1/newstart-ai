"""
Task 4 -- build the final labeled dataset (documents.csv).

Why `label` is derived from source_agency via a fixed mapping instead of
being assigned per file by hand: for this MVP, each of Task 1's four
agencies maps to exactly one routing category (see the project intro's
service domains). Hand-labeling would just be re-typing what the folder
structure already tells us, and would risk inconsistency between reviewers.
If a single agency ever needs to span multiple categories, that's the
moment to switch to genuine per-file labeling.

Why `document_type` is a small closed set: the BERT and RAG teams need a
stable label space to train and evaluate against. An open-ended free-text
field would let this column grow into an unreviewable pile of near-duplicate
categories over time.
"""

import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
EXTRACTED_DIR = DATA_DIR / "extracted_text"
OUT_PATH = DATA_DIR / "processed" / "documents.csv"

# For this MVP's four agencies only -- see module docstring for why this is
# a fixed mapping, not a per-file decision.
AGENCY_TO_LABEL = {
    "uscis": "immigration",
    "dmv": "motor_vehicle",
    "ssa": "social_security",
    "irs": "tax",
}

# Closed set on purpose -- see module docstring. Extend this deliberately,
# together with docs/dataset_summary.md, not by adding a new string
# wherever convenient.
DOCUMENT_TYPES = {"form", "notice", "instructions", "other"}

FIELDNAMES = ["file_path", "text", "label", "source_agency", "document_type"]


def infer_document_type(filename: str) -> str:
    """
    Best-effort guess from filename conventions. This seeds the column so
    Task 5's EDA/review pass has something to correct by hand -- it is not
    meant to be trusted as ground truth on its own.
    """
    name = filename.lower()
    if "instr" in name:
        return "instructions"
    if "notice" in name:
        return "notice"
    return "form"


def build_dataset() -> list[dict]:
    """Walks every agency's extracted text and assembles dataset rows."""
    rows = []
    for agency, label in AGENCY_TO_LABEL.items():
        agency_dir = EXTRACTED_DIR / agency
        if not agency_dir.exists():
            continue
        for txt_path in sorted(agency_dir.glob("*.txt")):
            text = txt_path.read_text(encoding="utf-8").strip()
            if not text:
                # Already flagged by Task 3's extraction step; skip rather
                # than add a useless empty row here.
                continue
            rows.append(
                {
                    "file_path": str(txt_path.relative_to(DATA_DIR.parent)),
                    "text": text,
                    "label": label,
                    "source_agency": agency,
                    "document_type": infer_document_type(txt_path.name),
                }
            )
    return rows


def run() -> None:
    rows = build_dataset()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    run()
