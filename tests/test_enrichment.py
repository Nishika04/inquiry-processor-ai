"""
tests/test_enrichment.py

Unit tests for utils/enrichment.py — pure Python, no LLM calls required.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from utils.enrichment import (
    extract_domain,
    is_personal_email,
    infer_company_name,
    infer_industry,
    infer_company_size,
    generate_linkedin_url,
    estimate_confidence,
    enrich_contact,
)


# ── extract_domain ─────────────────────────────────────────────────────────────

class TestExtractDomain:
    def test_simple(self):
        assert extract_domain("alice@example.com") == "example.com"

    def test_subdomain(self):
        assert extract_domain("bob@mail.acme.co.uk") == "mail.acme.co.uk"

    def test_uppercase_normalised(self):
        assert extract_domain("ALICE@EXAMPLE.COM") == "example.com"

    def test_whitespace_stripped(self):
        assert extract_domain("  carol@test.org  ") == "test.org"


# ── is_personal_email ──────────────────────────────────────────────────────────

class TestIsPersonalEmail:
    @pytest.mark.parametrize("domain", [
        "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
        "icloud.com", "protonmail.com",
    ])
    def test_personal_domains(self, domain):
        assert is_personal_email(domain) is True

    def test_corporate_domain(self):
        assert is_personal_email("stripe.com") is False

    def test_unknown_domain(self):
        assert is_personal_email("unknownstartup.io") is False


# ── infer_company_name ────────────────────────────────────────────────────────

class TestInferCompanyName:
    def test_known_domain(self):
        assert infer_company_name("google.com") == "Google"

    def test_known_domain_stripe(self):
        assert infer_company_name("stripe.com") == "Stripe"

    def test_personal_email(self):
        result = infer_company_name("gmail.com")
        assert "personal" in result.lower() or "n/a" in result.lower()

    def test_hyphenated_domain(self):
        result = infer_company_name("acme-corp.com")
        assert result == "Acme Corp"

    def test_single_word_domain(self):
        result = infer_company_name("widgets.io")
        # Should capitalise the slug
        assert result[0].isupper()


# ── infer_industry ────────────────────────────────────────────────────────────

class TestInferIndustry:
    def test_known_domain(self):
        assert infer_industry("stripe.com") == "FinTech"

    def test_tech_hint(self):
        result = infer_industry("cloudsoftware.com")
        assert result == "Technology"

    def test_health_hint(self):
        result = infer_industry("healthapp.io")
        assert result == "Healthcare / Life Sciences"

    def test_unknown(self):
        result = infer_industry("zzz12345.com")
        assert result == "Unknown"


# ── infer_company_size ────────────────────────────────────────────────────────

class TestInferCompanySize:
    def test_known_large_company(self):
        assert infer_company_size("google.com") == "10,000+"

    def test_personal_email(self):
        assert infer_company_size("gmail.com") == "N/A"

    def test_short_slug_heuristic(self):
        # "acme.com" — slug length 4 → 500+
        result = infer_company_size("acme.com")
        assert "500" in result or "+" in result


# ── generate_linkedin_url ─────────────────────────────────────────────────────

class TestGenerateLinkedInUrl:
    def test_basic(self):
        url = generate_linkedin_url("Alice", "Smith", "acme.com")
        assert url == "https://www.linkedin.com/in/alice-smith"

    def test_special_chars_stripped(self):
        url = generate_linkedin_url("Jean-Luc", "O'Brien", "example.com")
        # hyphens and apostrophes should be stripped
        assert "https://www.linkedin.com/in/" in url
        assert "'" not in url

    def test_case_normalised(self):
        url = generate_linkedin_url("BOB", "JONES", "example.com")
        assert url == "https://www.linkedin.com/in/bob-jones"


# ── estimate_confidence ───────────────────────────────────────────────────────

class TestEstimateConfidence:
    def test_known_domain_high(self):
        assert estimate_confidence("salesforce.com") == "high"

    def test_personal_low(self):
        assert estimate_confidence("gmail.com") == "low"

    def test_unknown_medium(self):
        assert estimate_confidence("mystartup.io") == "medium"


# ── enrich_contact (integration) ──────────────────────────────────────────────

class TestEnrichContact:
    def test_known_company(self):
        result = enrich_contact("Sarah", "Chen", "sarah.chen@stripe.com")
        assert result["company_name"] == "Stripe"
        assert result["industry"] == "FinTech"
        assert result["confidence"] == "high"
        assert result["linkedin_profile"] == "https://www.linkedin.com/in/sarah-chen"
        assert result["domain"] == "stripe.com"
        assert result["is_personal_email"] is False

    def test_personal_email(self):
        result = enrich_contact("John", "Doe", "john.doe@gmail.com")
        assert result["is_personal_email"] is True
        assert result["confidence"] == "low"

    def test_unknown_domain(self):
        result = enrich_contact("Alex", "Park", "alex@mystartup.io")
        assert result["confidence"] == "medium"
        assert result["domain"] == "mystartup.io"
        assert "mystartup" in result["company_name"].lower()
