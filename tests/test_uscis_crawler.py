"""
Unit tests for src/crawler/uscis_crawler.py's two-stage discovery logic.

All HTML fixtures below are inline strings standing in for real USCIS
pages -- no live requests are made. See notebooks/01_crawler_research.ipynb
for the manually-validated research against the real site that motivated
this two-stage design.
"""

from src.crawler import uscis_crawler


class FakeHTMLResponse:
    def __init__(self, text):
        self.text = text


INDEX_PAGE_HTML = """
<html><body>
  <a href="/i-485">Form I-485</a>
  <a href="/i-765">Form I-765</a>
  <a href="/n-400">Form N-400</a>
  <a href="/i-485supa">Form I-485 Supplement A</a>
  <a href="/i-765v">Form I-765V</a>
  <a href="/forms/all-forms">All Forms</a>
  <a href="/forms/forms">Forms</a>
  <a href="/forms/filing-guidance">Filing Guidance</a>
  <a href="/forms/filing-fees">Filing Fees</a>
  <a href="/forms/explore-my-options">Explore My Options</a>
  <a href="/about-us">About USCIS</a>
</body></html>
"""

DETAIL_PAGE_HTML = """
<html><body>
  <a href="/sites/default/files/document/forms/i-485.pdf">Form I-485 (PDF)</a>
  <a href="/sites/default/files/document/forms/i-485instr.pdf">Instructions (PDF)</a>
  <a href="/i-485">Back to form page</a>
</body></html>
"""


# --- is_form_detail_path ---------------------------------------------------


def test_is_form_detail_path_matches_known_forms():
    for path in [
        "/i-485", "/i-765", "/i-90", "/n-400", "/i-130",
        "/i-485supa", "/i-485supj", "/i-765v", "/i-905", "/i-907",
    ]:
        assert uscis_crawler.is_form_detail_path(path), path


def test_is_form_detail_path_rejects_navigation_pages():
    for path in [
        "/forms/forms", "/forms/all-forms", "/forms/explore-my-options",
        "/forms/filing-guidance", "/forms/filing-fees", "/about-us",
    ]:
        assert not uscis_crawler.is_form_detail_path(path), path


def test_is_form_detail_path_is_case_insensitive():
    assert uscis_crawler.is_form_detail_path("/I-485")


# --- find_form_detail_links -------------------------------------------------


def test_find_form_detail_links_accepts_forms_and_rejects_nav_pages(monkeypatch):
    monkeypatch.setattr(
        uscis_crawler, "polite_get", lambda url: FakeHTMLResponse(INDEX_PAGE_HTML)
    )

    links = uscis_crawler.find_form_detail_links("https://www.uscis.gov/forms/all-forms")

    for expected in [
        "https://www.uscis.gov/i-485",
        "https://www.uscis.gov/i-765",
        "https://www.uscis.gov/n-400",
        "https://www.uscis.gov/i-485supa",
        "https://www.uscis.gov/i-765v",
    ]:
        assert expected in links

    for nav_fragment in ["all-forms", "forms/forms", "filing-guidance", "filing-fees",
                          "explore-my-options", "about-us"]:
        assert not any(nav_fragment in url for url in links), nav_fragment


def test_find_form_detail_links_deduplicates(monkeypatch):
    html = """
    <html><body>
      <a href="/i-90">Form I-90</a>
      <a href="/i-90?ref=list">Form I-90 again</a>
    </body></html>
    """
    monkeypatch.setattr(uscis_crawler, "polite_get", lambda url: FakeHTMLResponse(html))

    links = uscis_crawler.find_form_detail_links("https://www.uscis.gov/forms/all-forms")
    assert links.count("https://www.uscis.gov/i-90") == 1


# --- find_pdf_links ----------------------------------------------------------


def test_find_pdf_links_extracts_only_pdf_links(monkeypatch):
    monkeypatch.setattr(
        uscis_crawler, "polite_get", lambda url: FakeHTMLResponse(DETAIL_PAGE_HTML)
    )

    links = uscis_crawler.find_pdf_links("https://www.uscis.gov/i-485")

    assert links == [
        "https://www.uscis.gov/sites/default/files/document/forms/i-485.pdf",
        "https://www.uscis.gov/sites/default/files/document/forms/i-485instr.pdf",
    ]


# --- discover_pdf_links (two-stage composition + failure handling) ----------


def test_discover_pdf_links_skips_failing_detail_page_but_keeps_going(monkeypatch):
    def fake_polite_get(url):
        if url == "https://www.uscis.gov/forms/all-forms":
            return FakeHTMLResponse(INDEX_PAGE_HTML)
        if url == "https://www.uscis.gov/i-485":
            return FakeHTMLResponse(DETAIL_PAGE_HTML)
        raise RuntimeError(f"simulated failure fetching {url}")

    monkeypatch.setattr(uscis_crawler, "polite_get", fake_polite_get)

    pdf_links = uscis_crawler.discover_pdf_links(["https://www.uscis.gov/forms/all-forms"])

    # i-485's PDFs were found even though every other detail page "failed".
    assert "https://www.uscis.gov/sites/default/files/document/forms/i-485.pdf" in pdf_links
    assert "https://www.uscis.gov/sites/default/files/document/forms/i-485instr.pdf" in pdf_links


def test_discover_pdf_links_propagates_index_page_failure(monkeypatch):
    def fake_polite_get(url):
        raise RuntimeError("index page unreachable")

    monkeypatch.setattr(uscis_crawler, "polite_get", fake_polite_get)

    try:
        uscis_crawler.discover_pdf_links(["https://www.uscis.gov/forms/all-forms"])
        assert False, "expected the index-page failure to propagate"
    except RuntimeError:
        pass


# --- run(limit=...) ----------------------------------------------------------


def test_run_respects_limit_and_stops_downloading(tmp_path, monkeypatch):
    monkeypatch.setattr(uscis_crawler, "RAW_DIR", tmp_path)
    monkeypatch.setattr(
        uscis_crawler,
        "discover_pdf_links",
        lambda index_urls=None: [
            "https://www.uscis.gov/a.pdf",
            "https://www.uscis.gov/b.pdf",
            "https://www.uscis.gov/c.pdf",
        ],
    )

    calls = []

    def fake_download_pdf(url, dest_dir, filename, manifest, allowed_domains=None):
        calls.append(url)
        path = dest_dir / filename
        path.write_bytes(b"%PDF-1.4")
        return path

    monkeypatch.setattr(uscis_crawler, "download_pdf", fake_download_pdf)

    result = uscis_crawler.run(limit=2)

    assert len(result) == 2
    assert calls == ["https://www.uscis.gov/a.pdf", "https://www.uscis.gov/b.pdf"]


def test_run_continues_after_a_download_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(uscis_crawler, "RAW_DIR", tmp_path)
    monkeypatch.setattr(
        uscis_crawler,
        "discover_pdf_links",
        lambda index_urls=None: [
            "https://www.uscis.gov/a.pdf",
            "https://www.uscis.gov/broken.pdf",
            "https://www.uscis.gov/c.pdf",
        ],
    )

    def fake_download_pdf(url, dest_dir, filename, manifest, allowed_domains=None):
        if "broken" in url:
            raise Exception("simulated download failure")
        path = dest_dir / filename
        path.write_bytes(b"%PDF-1.4")
        return path

    monkeypatch.setattr(uscis_crawler, "download_pdf", fake_download_pdf)

    result = uscis_crawler.run(limit=None)

    assert len(result) == 2
    assert all("broken" not in str(p) for p in result)
