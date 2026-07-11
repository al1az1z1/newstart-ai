"""
Task 3 -- extract machine-readable text from digitally-generated PDFs.

Why PyMuPDF ("fitz") over alternatives: it's fast enough to run over the
whole corpus in seconds rather than minutes, and its page-by-page API
mirrors the structure OCR will eventually need for scanned documents -- so
when a future task adds OCR for image-based notices, it can slot in as a
per-page fallback instead of requiring a rewrite of this module.

Why raw/ and extracted_text/ are mirrored folder trees (same agency
subfolders, matching filenames): it makes the link between a text file and
its source PDF obvious from the path alone, no lookup table required.

Explicitly out of scope here: scanned / image-only PDFs. The intro's OCR
step (for uploaded user documents) is a separate, later concern -- this
task only extracts text that's already embedded in the PDF.
"""

from pathlib import Path

import fitz  # PyMuPDF

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
EXTRACTED_DIR = DATA_DIR / "extracted_text"

AGENCIES = ["uscis", "dmv", "ssa", "irs"]

# Below this character count we treat extraction as having failed rather
# than silently writing a near-empty text file into the dataset. Scanned/
# image-only PDFs land here -- they need OCR, which is out of scope for
# this task (see module docstring) -- so we log them instead of guessing.
MIN_CHARS_FOR_SUCCESS = 20


def extract_pdf_text(pdf_path: Path) -> str:
    """Returns a PDF's text, page by page, in reading order."""
    with fitz.open(pdf_path) as doc:
        return "\n".join(page.get_text() for page in doc)


def extract_agency(agency: str) -> list[str]:
    """
    Extracts text for every PDF under data/raw/<agency>/, writing one .txt
    per PDF into data/extracted_text/<agency>/.

    Returns the filenames that failed the MIN_CHARS_FOR_SUCCESS check, so
    callers can report them instead of the failure being silently hidden
    inside an empty-ish text file.
    """
    src_dir = RAW_DIR / agency
    dst_dir = EXTRACTED_DIR / agency
    dst_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for pdf_path in sorted(src_dir.glob("*.pdf")):
        text = extract_pdf_text(pdf_path)
        if len(text.strip()) < MIN_CHARS_FOR_SUCCESS:
            failures.append(pdf_path.name)
            continue
        out_path = dst_dir / (pdf_path.stem + ".txt")
        out_path.write_text(text, encoding="utf-8")
    return failures


def run() -> dict[str, list[str]]:
    """Extracts every agency and returns {agency: [failed filenames]}."""
    all_failures = {}
    for agency in AGENCIES:
        failures = extract_agency(agency)
        all_failures[agency] = failures
        if failures:
            print(
                f"[{agency}] {len(failures)} file(s) produced little/no text "
                f"(likely scanned images -- needs OCR, out of scope for "
                f"Task 3): {failures}"
            )
    return all_failures


if __name__ == "__main__":
    run()
