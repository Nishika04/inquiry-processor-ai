"""
utils/enrichment.py

Lightweight utilities for enriching a contact record from an email address.

NOTE: This module performs *realistic simulation* — no live scraping is done.
In production you would swap these functions for calls to Clearbit, Hunter.io,
LinkedIn API, or similar data-enrichment services.
"""

import re
from typing import Tuple

# ---------------------------------------------------------------------------
# Known-domain registry (extend as needed)
# Maps domain → (company_name, industry, company_size_bucket)
# ---------------------------------------------------------------------------
KNOWN_DOMAINS: dict[str, Tuple[str, str, str]] = {
    "google.com":        ("Google",          "Technology",           "10,000+"),
    "microsoft.com":     ("Microsoft",        "Technology",           "10,000+"),
    "amazon.com":        ("Amazon",           "E-Commerce / Cloud",   "10,000+"),
    "apple.com":         ("Apple",            "Technology",           "10,000+"),
    "meta.com":          ("Meta",             "Social Media / Tech",  "10,000+"),
    "salesforce.com":    ("Salesforce",       "CRM / SaaS",           "10,000+"),
    "hubspot.com":       ("HubSpot",          "Marketing SaaS",       "5,000-10,000"),
    "stripe.com":        ("Stripe",           "FinTech",              "1,000-5,000"),
    "shopify.com":       ("Shopify",          "E-Commerce SaaS",      "5,000-10,000"),
    "slack.com":         ("Slack",            "Productivity SaaS",    "1,000-5,000"),
    "notion.so":         ("Notion",           "Productivity SaaS",    "200-1,000"),
    "openai.com":        ("OpenAI",           "Artificial Intelligence","200-1,000"),
    "anthropic.com":     ("Anthropic",        "Artificial Intelligence","200-1,000"),
    "twilio.com":        ("Twilio",           "Communications SaaS",  "1,000-5,000"),
    "datadog.com":       ("Datadog",          "Observability SaaS",   "1,000-5,000"),
    "github.com":        ("GitHub",           "Developer Tools",      "1,000-5,000"),
    "atlassian.com":     ("Atlassian",        "Developer SaaS",       "5,000-10,000"),
    "zoom.us":           ("Zoom",             "Communications SaaS",  "5,000-10,000"),
    "netflix.com":       ("Netflix",          "Entertainment / Tech", "10,000+"),
    "spotify.com":       ("Spotify",          "Music / Tech",         "5,000-10,000"),
    "airbnb.com":        ("Airbnb",           "Travel / Marketplace", "5,000-10,000"),
    "uber.com":          ("Uber",             "Transportation Tech",  "10,000+"),
    "lyft.com":          ("Lyft",             "Transportation Tech",  "1,000-5,000"),
    "twitter.com":       ("X (Twitter)",      "Social Media",         "1,000-5,000"),
    "x.com":             ("X (Twitter)",      "Social Media",         "1,000-5,000"),
    "linkedin.com":      ("LinkedIn",         "Professional Network", "10,000+"),
}

# Free / consumer email providers — company data is not inferrable
PERSONAL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "icloud.com", "protonmail.com", "aol.com", "live.com",
    "me.com", "mac.com",
}

# Industry keywords to help guess an unknown domain's sector
_INDUSTRY_HINTS: list[Tuple[str, str]] = [
    (r"bank|finance|capital|invest|credit|fund|wealth",  "Financial Services"),
    (r"health|medical|pharma|clinic|hospital|med",       "Healthcare / Life Sciences"),
    (r"school|edu|university|college|academy|learn",     "Education"),
    (r"law|legal|attorney|counsel",                      "Legal Services"),
    (r"shop|store|retail|market|commerce",               "Retail / E-Commerce"),
    (r"tech|software|digital|data|cloud|ai|ml|dev",      "Technology"),
    (r"media|news|publish|press|journal",                "Media / Publishing"),
    (r"consult|advisory|strategy|partner",               "Consulting / Professional Services"),
    (r"realty|real.?estate|property|homes",              "Real Estate"),
    (r"logistics|shipping|supply|freight|cargo",         "Logistics / Supply Chain"),
    (r"energy|solar|wind|power|utility",                 "Energy / Utilities"),
    (r"food|restaurant|cafe|kitchen|catering",           "Food & Beverage"),
    (r"travel|hotel|hospitality|resort|airline",         "Travel & Hospitality"),
    (r"gov|government|municipal|state\.",                "Government / Public Sector"),
    (r"non.?profit|charity|foundation|ngo",              "Non-Profit"),
    (r"agency|creative|design|brand|marketing",          "Marketing / Creative Agency"),
    (r"manufacture|industrial|factory|production",       "Manufacturing"),
    (r"telecom|wireless|mobile|network|internet",        "Telecommunications"),
]


def extract_domain(email: str) -> str:
    """Return the domain portion of an email address, lowercased."""
    return email.strip().lower().split("@")[-1]


def is_personal_email(domain: str) -> bool:
    """Return True if the domain is a known free/consumer provider."""
    return domain in PERSONAL_DOMAINS


def _slug_to_name(slug: str) -> str:
    """
    Convert a domain slug (the part before the TLD) into a human-readable
    company name by capitalising each hyphen- or digit-separated word.

    Examples:
        "acme-corp"    → "Acme Corp"
        "mybigcompany" → "Mybigcompany"   (no separators → leave as-is)
        "x2-solutions" → "X2 Solutions"
    """
    # Remove numeric-only segments that look like port numbers
    slug = re.sub(r"^\d+$", "", slug)
    words = re.split(r"[-_]", slug)
    return " ".join(w.capitalize() for w in words if w)


def infer_company_name(domain: str) -> str:
    """
    Return a best-guess company name for the given domain.

    Order of precedence:
    1. Known-domain registry
    2. Strip common TLDs and capitalise the slug
    3. Fall back to returning the raw domain
    """
    if domain in KNOWN_DOMAINS:
        return KNOWN_DOMAINS[domain][0]

    if is_personal_email(domain):
        return "N/A (personal email)"

    # Strip the TLD(s) — e.g. "acme.co.uk" → "acme", "widgets.io" → "widgets"
    parts = domain.split(".")
    slug = parts[0] if len(parts) >= 2 else domain
    return _slug_to_name(slug)


def infer_industry(domain: str) -> str:
    """
    Return a best-guess industry label.

    1. Known-domain registry
    2. Regex hints against the full domain string
    3. Default to "Unknown"
    """
    if domain in KNOWN_DOMAINS:
        return KNOWN_DOMAINS[domain][1]

    domain_lower = domain.lower()
    for pattern, industry in _INDUSTRY_HINTS:
        if re.search(pattern, domain_lower, re.IGNORECASE):
            return industry

    return "Unknown"


def infer_company_size(domain: str) -> str:
    """
    Return a best-guess company size bucket.

    1. Known-domain registry
    2. Educated guess from domain length / type
    3. Default to "Unknown"
    """
    if domain in KNOWN_DOMAINS:
        return KNOWN_DOMAINS[domain][2]

    if is_personal_email(domain):
        return "N/A"

    # Heuristic: shorter, single-word .com domains tend to be larger companies
    parts = domain.split(".")
    slug_length = len(parts[0])
    if slug_length <= 5:
        return "500+"       # short slugs are often established brands
    if slug_length <= 10:
        return "10-500"
    return "1-50"


def generate_linkedin_url(
    first_name: str,
    last_name: str,
    domain: str,
) -> str:
    """
    Generate a *probable* LinkedIn profile URL using standard naming patterns.

    LinkedIn profile slugs follow the convention:
        /in/firstname-lastname[-N]
    where N is a disambiguation number LinkedIn appends for common names.
    We omit the number since it is not deterministically known without a lookup.
    """
    fn = re.sub(r"[^a-z0-9]", "", first_name.lower())
    ln = re.sub(r"[^a-z0-9]", "", last_name.lower())
    return f"https://www.linkedin.com/in/{fn}-{ln}"


def estimate_confidence(domain: str) -> str:
    """
    Estimate how confident we are in the enrichment data.

    high   → domain is in our known-domain registry
    medium → domain looks corporate (not personal, has recognisable structure)
    low    → personal email or very ambiguous domain
    """
    if domain in KNOWN_DOMAINS:
        return "high"
    if is_personal_email(domain):
        return "low"
    return "medium"


def enrich_contact(first_name: str, last_name: str, email: str) -> dict:
    """
    Top-level helper that returns a fully enriched contact dict.
    This is called by the Research Agent tool.
    """
    domain = extract_domain(email)
    return {
        "domain":          domain,
        "company_name":    infer_company_name(domain),
        "industry":        infer_industry(domain),
        "company_size":    infer_company_size(domain),
        "linkedin_profile": generate_linkedin_url(first_name, last_name, domain),
        "confidence":      estimate_confidence(domain),
        "is_personal_email": is_personal_email(domain),
    }
