# Dataset Summary (Module 3)

> Fill this in after running the full pipeline (crawl -> extract -> build
> dataset -> EDA) on the real, collected documents. This file ships to the
> BERT and RAG teams as the single source of truth for "what is in
> `documents.csv` and what its limits are" -- it should answer the
> questions those teams would otherwise have to reverse-engineer from the
> pipeline code.

## Source

- Agencies: USCIS, DMV (California), SSA, IRS
- Collection method: automated crawler (`src/crawler/`) + manual spot-checks
- Collection date range: _fill in_
- Per-file provenance: `data/raw/<agency>/manifest.csv` (source URL, sha256, download timestamp)

## Schema (`data/processed/documents.csv`)

| column | meaning |
|---|---|
| `file_path` | path to the extracted `.txt` this row was built from |
| `text` | full extracted document text |
| `label` | routing category: `immigration` / `motor_vehicle` / `social_security` / `tax` |
| `source_agency` | `uscis` / `dmv` / `ssa` / `irs` |
| `document_type` | `form` / `notice` / `instructions` / `other` |

## Dataset statistics

_Fill in from `src/dataset/eda.py` / `notebooks/03_eda.ipynb` output:_

- Documents per class:
- Documents per agency:
- Text length distribution (min / median / max words):
- Duplicate rows found:
- Class imbalance notes:

## Known limitations

- Scanned/image-only PDFs are excluded -- Task 3 only extracts text already
  embedded in the PDF; OCR for scanned documents is future work (see the
  project intro's OCR step for *user-uploaded* documents, which is separate).
- `document_type` starts from a filename heuristic (`infer_document_type` in
  `src/dataset/build_dataset.py`) and needs manual review before it's
  treated as ground truth.
- Only 4 of the platform's eventual service categories are represented here.
  Healthcare, education, and employment documents are future data-collection
  work, not present in this MVP dataset.
- Labels are assigned per-agency, not per-document (see the "why" in
  `src/dataset/build_dataset.py`'s module docstring) -- this is a
  simplification appropriate for the MVP's 1:1 agency-to-category mapping.

## Handoff

- **BERT training** consumes `data/processed/documents.csv` directly
  (`text` + `label` columns).
- **RAG knowledge base** consumes `data/extracted_text/<agency>/*.txt`
  (full document text, pre-chunking) plus `manifest.csv` for source
  attribution.
