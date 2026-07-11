"""
Unit tests for the shared crawler utilities in src/crawler/base_crawler.py.

No live HTTP requests are made: `polite_get` is monkeypatched with a
FakeResponse everywhere a network call would otherwise happen, per the
project's crawling-etiquette rule that only real, explicitly-capped crawl
runs talk to actual government servers.
"""

import pytest

from src.crawler import base_crawler


class FakeResponse:
    def __init__(self, content=b"", url="", headers=None, status_code=200):
        self.content = content
        self.text = content.decode("utf-8", errors="ignore")
        self.url = url
        self.headers = headers or {}
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


# --- normalize_url / dedupe_urls -----------------------------------------


def test_normalize_url_strips_query_and_resolves_relative():
    base = "https://www.uscis.gov"
    relative = base_crawler.normalize_url("/forms/i-485.pdf?ver=2", base)
    absolute = base_crawler.normalize_url("https://www.uscis.gov/forms/i-485.pdf", base)
    assert relative == absolute == "https://www.uscis.gov/forms/i-485.pdf"


def test_dedupe_urls_preserves_first_seen_order():
    urls = ["a", "b", "a", "c", "b"]
    assert base_crawler.dedupe_urls(urls) == ["a", "b", "c"]


# --- is_on_allowed_domain -------------------------------------------------


def test_is_on_allowed_domain_accepts_domain_and_subdomains():
    assert base_crawler.is_on_allowed_domain("https://www.uscis.gov/i-485", ["uscis.gov"])
    assert base_crawler.is_on_allowed_domain("https://uscis.gov/i-485", ["uscis.gov"])


def test_is_on_allowed_domain_rejects_other_domains():
    assert not base_crawler.is_on_allowed_domain(
        "https://evil.example.com/i-485.pdf", ["uscis.gov"]
    )


# --- download_pdf validation ----------------------------------------------


def test_download_pdf_rejects_html_masquerading_as_pdf(tmp_path, monkeypatch):
    fake = FakeResponse(content=b"<html>not a pdf</html>", url="https://www.uscis.gov/i-485.pdf")
    monkeypatch.setattr(base_crawler, "polite_get", lambda url: fake)

    manifest = base_crawler.ManifestWriter(tmp_path)
    with pytest.raises(base_crawler.InvalidPDFError):
        base_crawler.download_pdf(
            "https://www.uscis.gov/i-485.pdf",
            tmp_path,
            "i-485.pdf",
            manifest,
            allowed_domains=["uscis.gov"],
        )
    assert not (tmp_path / "i-485.pdf").exists()


def test_download_pdf_rejects_redirect_off_domain(tmp_path, monkeypatch):
    fake = FakeResponse(content=b"%PDF-1.4 ...", url="https://evil.example.com/i-485.pdf")
    monkeypatch.setattr(base_crawler, "polite_get", lambda url: fake)

    manifest = base_crawler.ManifestWriter(tmp_path)
    with pytest.raises(base_crawler.InvalidPDFError):
        base_crawler.download_pdf(
            "https://www.uscis.gov/i-485.pdf",
            tmp_path,
            "i-485.pdf",
            manifest,
            allowed_domains=["uscis.gov"],
        )


def test_download_pdf_accepts_valid_pdf_and_records_manifest(tmp_path, monkeypatch):
    fake = FakeResponse(content=b"%PDF-1.4 fake pdf bytes", url="https://www.uscis.gov/i-485.pdf")
    monkeypatch.setattr(base_crawler, "polite_get", lambda url: fake)

    manifest = base_crawler.ManifestWriter(tmp_path)
    result = base_crawler.download_pdf(
        "https://www.uscis.gov/i-485.pdf",
        tmp_path,
        "i-485.pdf",
        manifest,
        allowed_domains=["uscis.gov"],
    )
    assert result == tmp_path / "i-485.pdf"
    assert result.read_bytes() == fake.content

    rows = (tmp_path / "manifest.csv").read_text().splitlines()
    assert len(rows) == 2  # header + one row


def test_download_pdf_skips_url_already_in_manifest(tmp_path, monkeypatch):
    calls = []

    def fake_polite_get(url):
        calls.append(url)
        return FakeResponse(content=b"%PDF-1.4 abc", url=url)

    monkeypatch.setattr(base_crawler, "polite_get", fake_polite_get)

    manifest = base_crawler.ManifestWriter(tmp_path)
    url = "https://www.uscis.gov/i-485.pdf"

    first = base_crawler.download_pdf(url, tmp_path, "i-485.pdf", manifest, allowed_domains=["uscis.gov"])
    second = base_crawler.download_pdf(url, tmp_path, "i-485.pdf", manifest, allowed_domains=["uscis.gov"])

    assert first is not None
    assert second is None  # skipped: already recorded, no re-download
    assert len(calls) == 1  # no repeat network call


def test_manifest_reloads_prior_state_across_instances(tmp_path, monkeypatch):
    fake = FakeResponse(content=b"%PDF-1.4 abc", url="https://www.uscis.gov/i-485.pdf")
    monkeypatch.setattr(base_crawler, "polite_get", lambda url: fake)

    first_manifest = base_crawler.ManifestWriter(tmp_path)
    base_crawler.download_pdf(
        "https://www.uscis.gov/i-485.pdf", tmp_path, "i-485.pdf", first_manifest,
        allowed_domains=["uscis.gov"],
    )

    # Simulate a fresh process / notebook re-run reading the same folder.
    second_manifest = base_crawler.ManifestWriter(tmp_path)
    assert second_manifest.already_downloaded("https://www.uscis.gov/i-485.pdf") is not None


# --- filename collision handling ------------------------------------------


def test_manifest_filename_collision_gets_deterministic_suffix(tmp_path):
    manifest = base_crawler.ManifestWriter(tmp_path)

    name_a = manifest.resolve_filename("i-90.pdf", sha256="a" * 64)
    manifest.append(
        base_crawler.DownloadRecord(
            filename=name_a, source_url="https://www.uscis.gov/i-90.pdf",
            sha256="a" * 64, downloaded_at="2026-01-01T00:00:00+00:00",
        )
    )

    # Different content, same preferred filename -> must not silently collide.
    name_b = manifest.resolve_filename("i-90.pdf", sha256="b" * 64)
    assert name_a == "i-90.pdf"
    assert name_b != "i-90.pdf"
    assert name_b.endswith(".pdf")

    # Same content, same filename -> idempotent, reuse the same name.
    assert manifest.resolve_filename("i-90.pdf", sha256="a" * 64) == name_a
