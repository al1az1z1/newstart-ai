"""
USCIS document crawler.

Research finding (validated manually against the live site in
notebooks/01_crawler_research.ipynb): the "All Forms" index page
(https://www.uscis.gov/forms/all-forms) returns HTTP 200 with hundreds of
links but zero direct .pdf links. It links individual form-detail pages
instead (e.g. /i-485, /n-400, /i-485supa), and each *detail* page is where
the actual PDF(s) live (e.g. .../i-485.pdf, .../i-485instr.pdf). So this
crawler is two-stage: index page -> form-detail pages -> PDF links ->
download. The original single-stage version was based on an incorrect
assumption and never actually found any PDFs.

Why this file still holds no download/retry/hashing/manifest/domain-check
logic: that stays in base_crawler.py (see its module docstring). This file
only knows USCIS's specific page structure -- the index page, what a
form-detail URL looks like, and where PDFs sit on a detail page.

Filenames of discovered PDFs (e.g. i-485.pdf vs i-485instr.pdf vs
i-485supa.pdf) are preserved as-is unless a genuine collision occurs (see
ManifestWriter.resolve_filename in base_crawler.py). That's deliberate:
Task 4's dataset build already distinguishes forms/instructions/supplements
from these filename conventions, so nothing here should rename or discard
that signal.
"""

import logging
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from .base_crawler import (
    ManifestWriter,
    dedupe_urls,
    download_pdf,
    normalize_url,
    polite_get,
)

logger = logging.getLogger(__name__)

AGENCY = "uscis"
BASE_URL = "https://www.uscis.gov"
ALLOWED_DOMAINS = ["uscis.gov"]
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / AGENCY

INDEX_URLS: list[str] = [
    "https://www.uscis.gov/forms/all-forms",
]

# A USCIS form-detail path is a short slug at the site root: 1-2 letters, a
# dash, digits, and an optional lowercase/alphanumeric suffix for
# supplements or versions -- e.g. "/i-485", "/n-400", "/i-485supa",
# "/i-765v". This deliberately excludes general navigation pages such as
# "/forms/all-forms" or "/forms/filing-guidance", which live under a
# "/forms/" prefix and don't match this letter-dash-digits shape at all.
FORM_DETAIL_PATH_RE = re.compile(r"^/[a-z]{1,2}-\d+[a-z0-9]*$", re.IGNORECASE)


def is_form_detail_path(path: str) -> bool:
    """True for a USCIS form-detail path like /i-485; False for nav pages."""
    return bool(FORM_DETAIL_PATH_RE.match(path))


def find_form_detail_links(index_url: str) -> list[str]:
    """
    Stage 1: parses the "All Forms" index page and returns deduped,
    absolute URLs of individual form-detail pages (e.g.
    https://www.uscis.gov/i-485). Does not look for PDFs here -- the index
    page doesn't have any (see module docstring).
    """
    soup = BeautifulSoup(polite_get(index_url).text, "lxml")
    candidates = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        path = urlparse(urljoin(BASE_URL, href)).path
        if is_form_detail_path(path):
            candidates.append(normalize_url(href, BASE_URL))
    return dedupe_urls(candidates)


def find_pdf_links(detail_url: str) -> list[str]:
    """
    Stage 2: parses one form-detail page and returns deduped, absolute PDF
    URLs found on it (typically the form itself and its instructions).
    """
    soup = BeautifulSoup(polite_get(detail_url).text, "lxml")
    links = [
        normalize_url(a["href"], detail_url)
        for a in soup.find_all("a", href=True)
        if a["href"].lower().split("?")[0].endswith(".pdf")
    ]
    return dedupe_urls(links)


def discover_pdf_links(index_urls: list[str] | None = None) -> list[str]:
    """
    Runs the full discovery pipeline (index -> detail pages -> PDFs)
    without downloading anything. Safe to call freely -- from a notebook or
    otherwise -- to see what a crawl *would* fetch before committing to
    `run()`.

    A failure on one detail page is logged and skipped rather than aborting
    the whole discovery: one broken or moved form shouldn't block every
    other form from being found. A failure on the index page itself does
    propagate, since without it there are no detail pages to visit at all.
    """
    index_urls = index_urls if index_urls is not None else INDEX_URLS
    all_pdf_links: list[str] = []

    for index_url in index_urls:
        detail_urls = find_form_detail_links(index_url)
        logger.info("index %s: %d form-detail page(s) discovered", index_url, len(detail_urls))

        processed, failed = 0, 0
        for detail_url in detail_urls:
            try:
                pdf_links = find_pdf_links(detail_url)
            except Exception as exc:
                failed += 1
                logger.warning("skipping detail page %s after error: %s", detail_url, exc)
                continue
            processed += 1
            all_pdf_links.extend(pdf_links)

        logger.info(
            "index %s: %d detail page(s) processed, %d failed", index_url, processed, failed
        )

    deduped = dedupe_urls(all_pdf_links)
    logger.info("discovery complete: %d unique PDF link(s) found", len(deduped))
    return deduped


def run(limit: int | None = None) -> list[Path]:
    """
    Discovers PDFs (see discover_pdf_links) and downloads them, in
    first-seen order, stopping once `limit` *newly downloaded* files have
    been saved. Files already recorded in manifest.csv from a prior run
    don't count against `limit`, aren't re-downloaded, and aren't
    duplicated in the manifest (see base_crawler.download_pdf).
    """
    manifest = ManifestWriter(RAW_DIR)
    pdf_links = discover_pdf_links()

    downloaded: list[Path] = []
    failures: list[str] = []
    for pdf_url in pdf_links:
        if limit is not None and len(downloaded) >= limit:
            break
        filename = pdf_url.rsplit("/", 1)[-1]
        try:
            result = download_pdf(
                pdf_url, RAW_DIR, filename, manifest, allowed_domains=ALLOWED_DOMAINS
            )
        except Exception as exc:
            failures.append(pdf_url)
            logger.warning("failed to download %s: %s", pdf_url, exc)
            continue
        if result is not None:
            downloaded.append(result)

    skipped = len(pdf_links) - len(downloaded) - len(failures)
    logger.info(
        "download complete: %d new file(s), %d already had, %d failure(s)",
        len(downloaded), skipped, len(failures),
    )
    return downloaded


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Deliberately capped -- see the crawling-etiquette notes in
    # notebooks/01_crawler_research.ipynb. An unlimited run is a decision to
    # make explicitly, not something that happens by running this file out
    # of habit.
    run(limit=5)
