"""
California DMV document crawler.

Same shape as uscis_crawler.py -- see that file for why the shared logic
lives in base_crawler.py instead of here. This module only knows where DMV
publishes its forms/notices.

TODO (needs research before the first real run):
- Confirm index page(s), e.g. https://www.dmv.ca.gov/portal/forms/
- The README scopes this to *California* DMV specifically (other states
  have their own sites/forms); keep that assumption explicit here so it
  isn't silently generalized later.
"""

from pathlib import Path

from bs4 import BeautifulSoup

from .base_crawler import ManifestWriter, dedupe_urls, download_pdf, normalize_url, polite_get

AGENCY = "dmv"
BASE_URL = "https://www.dmv.ca.gov"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / AGENCY

INDEX_URLS: list[str] = [
    # "https://www.dmv.ca.gov/portal/forms/",
]


def find_pdf_links(index_url: str) -> list[str]:
    soup = BeautifulSoup(polite_get(index_url).text, "lxml")
    links = [
        normalize_url(a["href"], BASE_URL)
        for a in soup.find_all("a", href=True)
        if a["href"].lower().split("?")[0].endswith(".pdf")
    ]
    return dedupe_urls(links)


def run(limit: int | None = None) -> list[Path]:
    manifest = ManifestWriter(RAW_DIR)
    downloaded = []
    for index_url in INDEX_URLS:
        for pdf_url in find_pdf_links(index_url):
            if limit is not None and len(downloaded) >= limit:
                return downloaded
            filename = pdf_url.rsplit("/", 1)[-1]
            downloaded.append(download_pdf(pdf_url, RAW_DIR, filename, manifest))
    return downloaded


if __name__ == "__main__":
    run()
