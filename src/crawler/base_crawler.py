"""
Shared crawling utilities for NewStart AI's document collection pipeline.

Why a shared base instead of copy-pasting per agency: every agency site is
different, but the *safety rules* around scraping a government website
should not depend on which agency we're touching. If the download / rate
limit / manifest logic were duplicated into every uscis_crawler.py,
dmv_crawler.py, etc., a fix to one of them (say, a courtesy delay that turns
out to be too aggressive) would silently not apply to the others.
Centralizing it here means every crawler is respectful and auditable by
construction, not by copy-paste discipline.
"""

import csv
import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import requests

# We identify ourselves honestly. Government sites are public infrastructure,
# not ours -- an honest User-Agent lets a site operator see who is hitting
# them and reach out if we're causing trouble. Pretending to be a browser
# would be a small dishonesty for no real benefit here.
USER_AGENT = (
    "NewStartAI-Capstone-Crawler/0.1 "
    "(USD MS-AAI capstone project; non-commercial, educational use)"
)

# Government sites aren't built for a research project's traffic patterns.
# A fixed delay between requests is the simplest way to guarantee we never
# look like -- or become -- a denial-of-service concern, even if a future
# bug causes a retry loop.
REQUEST_DELAY_SECONDS = 2.0

REQUEST_TIMEOUT_SECONDS = 30

PDF_MAGIC_BYTES = b"%PDF"


class InvalidPDFError(Exception):
    """Raised when a downloaded response doesn't actually look like a PDF."""


@dataclass
class DownloadRecord:
    """One row of manifest.csv -- the paper trail for a downloaded file."""

    filename: str
    source_url: str
    sha256: str
    downloaded_at: str


class ManifestWriter:
    """
    Tracks and appends provenance rows to data/raw/<agency>/manifest.csv.

    Why this exists: Task 1's deliverable is "the original PDFs you
    downloaded," but a PDF sitting in a folder doesn't say *where* it came
    from or *when*. Task 6 asks for dataset documentation, and a future
    re-crawl (to check for updated form revisions) needs to trace every
    file back to its official source URL.

    Why it loads the existing CSV at construction time: re-running a crawl
    (e.g. after fixing a bug or adding new detail pages) shouldn't
    re-download files we already have, or grow the manifest with duplicate
    rows for the same source URL. Loading prior state up front makes both
    checks a plain dict lookup instead of re-reading the file per row.
    """

    FIELDNAMES = ["filename", "source_url", "sha256", "downloaded_at"]

    def __init__(self, agency_dir: Path):
        agency_dir.mkdir(parents=True, exist_ok=True)
        self.path = agency_dir / "manifest.csv"
        self._sha256_by_source_url: dict[str, str] = {}
        self._sha256_by_filename: dict[str, str] = {}

        if self.path.exists():
            with self.path.open("r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self._sha256_by_source_url[row["source_url"]] = row["sha256"]
                    self._sha256_by_filename[row["filename"]] = row["sha256"]
        else:
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.FIELDNAMES)

    def already_downloaded(self, source_url: str) -> str | None:
        """Returns the recorded sha256 for this URL, or None if never seen."""
        return self._sha256_by_source_url.get(source_url)

    def resolve_filename(self, preferred_filename: str, sha256: str) -> str:
        """
        Picks the filename to save under, avoiding a silent overwrite when
        two different source documents would otherwise land on the same name.

        Why this can happen: form-detail pages occasionally link two
        distinct PDFs that share a basename (e.g. served from different
        folders). If the name is already taken by *different* content, we
        suffix it with a short hash instead of overwriting -- deterministic,
        so re-running the crawler doesn't keep minting new names for the
        same file, but still never clobbers a different one.
        """
        existing_sha = self._sha256_by_filename.get(preferred_filename)
        if existing_sha is None or existing_sha == sha256:
            return preferred_filename

        stem, dot, ext = preferred_filename.rpartition(".")
        unique_name = f"{stem}-{sha256[:8]}.{ext}" if dot else f"{preferred_filename}-{sha256[:8]}"
        print(
            f"[manifest] filename collision: {preferred_filename!r} is already used by "
            f"different content; saving this file as {unique_name!r} instead"
        )
        return unique_name

    def append(self, record: DownloadRecord) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [record.filename, record.source_url, record.sha256, record.downloaded_at]
            )
        self._sha256_by_source_url[record.source_url] = record.sha256
        self._sha256_by_filename[record.filename] = record.sha256


def polite_get(url: str) -> requests.Response:
    """
    Single point of control for every HTTP request the crawlers make.

    Why wrap requests.get() instead of calling it directly in each crawler:
    if we ever need to change the shared timeout, retry policy, or delay,
    there should be exactly one place to edit -- not several agency files
    that have quietly drifted apart.
    """
    time.sleep(REQUEST_DELAY_SECONDS)
    response = requests.get(
        url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response


def normalize_url(href: str, base_url: str) -> str:
    """
    Turns a link found in HTML (often relative, sometimes with a stray
    query string or fragment) into one canonical absolute URL.

    Why this matters for a crawler and not just a nicety: the same PDF is
    frequently linked from an agency site as both "/forms/i-485.pdf" and
    "https://www.uscis.gov/forms/i-485.pdf?ver=2". Without normalizing, our
    dedupe step (below) would treat those as two different documents and we'd
    download -- and count -- the same form twice.
    """
    absolute = urljoin(base_url, href)
    parsed = urlparse(absolute)
    # Drop query string and fragment: they're typically cache-busting or
    # tracking params on government sites, not different documents.
    return urlunparse(parsed._replace(query="", fragment=""))


def dedupe_urls(urls: list[str]) -> list[str]:
    """
    Removes duplicate links while preserving first-seen order.

    Why order matters: manifests and logs read most naturally when files
    appear in the order they were discovered on the page, which helps a
    human skim "did this crawl find what I expected" during review.
    """
    seen = set()
    deduped = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


def is_on_allowed_domain(url: str, allowed_domains: list[str]) -> bool:
    """
    Confirms a URL's host matches (or is a subdomain of) one of the
    caller's approved domains.

    Why this is shared rather than agency-specific: a redirect or a
    malformed relative link could otherwise send the crawler off the
    agency's own site entirely. Every crawler should refuse to download
    from a domain it wasn't told to trust, regardless of which agency it's
    written for.
    """
    host = urlparse(url).netloc.lower()
    return any(host == domain or host.endswith(f".{domain}") for domain in allowed_domains)


def _looks_like_pdf(response: requests.Response) -> bool:
    """
    Government sites occasionally serve an HTML error/redirect/login page
    at a URL that ends in .pdf (a broken link, a moved document, a 404
    handled with a 200 status). Trusting the URL or the Content-Type header
    alone would let us silently save that HTML as if it were the form. The
    PDF file signature -- its first four bytes -- is the one thing that's
    hard to get wrong by accident, so it's treated as the deciding check;
    Content-Type is only a secondary, more lenient signal (some servers
    misreport it even for genuine PDFs).
    """
    content_type = response.headers.get("Content-Type", "").lower()
    starts_with_pdf_signature = response.content[:4] == PDF_MAGIC_BYTES
    return starts_with_pdf_signature or "pdf" in content_type


def download_pdf(
    url: str,
    dest_dir: Path,
    filename: str,
    manifest: ManifestWriter,
    allowed_domains: list[str] | None = None,
) -> Path | None:
    """
    Downloads one PDF and records it in the manifest.

    Returns None -- without making a network request -- if this exact
    source URL is already in the manifest from a prior run: re-running a
    crawl shouldn't re-download files we already have or duplicate their
    manifest rows.

    `allowed_domains`, if given, is checked both against `url` (before any
    request) and against the final `response.url` (after any redirect) --
    a form-detail page redirecting somewhere unexpected shouldn't result in
    downloading and trusting whatever is on the other end.

    Why we hash the content: agencies periodically reissue a form under the
    same filename but with revised content (a new edition date, a changed
    field). The sha256 lets Task 6's documentation -- or a future crawl --
    detect "this is actually a different file" even when the name matches.
    """
    if manifest.already_downloaded(url) is not None:
        return None

    if allowed_domains is not None and not is_on_allowed_domain(url, allowed_domains):
        raise InvalidPDFError(f"Refusing to download from untrusted domain: {url}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    response = polite_get(url)

    if allowed_domains is not None and not is_on_allowed_domain(response.url, allowed_domains):
        raise InvalidPDFError(
            f"Refusing to save {url}: redirected off the approved domain(s) to {response.url}"
        )

    if not _looks_like_pdf(response):
        raise InvalidPDFError(
            f"{url} does not look like a PDF "
            f"(Content-Type={response.headers.get('Content-Type')!r}, "
            f"first bytes={response.content[:8]!r})"
        )

    content = response.content
    sha256 = hashlib.sha256(content).hexdigest()
    safe_filename = manifest.resolve_filename(filename, sha256)

    dest_path = dest_dir / safe_filename
    dest_path.write_bytes(content)

    manifest.append(
        DownloadRecord(
            filename=safe_filename,
            source_url=url,
            sha256=sha256,
            downloaded_at=datetime.now(timezone.utc).isoformat(),
        )
    )
    return dest_path
