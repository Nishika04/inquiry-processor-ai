"""
Pydantic models for request/response schemas throughout the inquiry processor.
All agent inputs and outputs are typed and validated here.
"""

from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional, Literal, List


# ── Input ──────────────────────────────────────────────────────────────────────

class InquiryFormInput(BaseModel):
    """Raw inquiry form submitted by a prospective contact."""

    first_name: str
    last_name: str
    email: str
    phone: Optional[str] = None
    inquiry_type: Optional[str] = None          # hint from the form (may be empty)
    subject: str
    message: str
    preferred_contact_method: Optional[str] = "email"

    @field_validator("email")
    @classmethod
    def email_must_contain_at(cls, v: str) -> str:
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("Invalid email address")
        return v.lower().strip()

    @field_validator("first_name", "last_name", "subject", "message")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be empty")
        return v.strip()


# ── Intermediate agent outputs ─────────────────────────────────────────────────

class ResearchOutput(BaseModel):
    """Enriched contact / company data produced by the Research Agent."""

    company_name: str
    industry: str
    company_size: str                           # e.g. "1-10", "51-200", "1000+"
    linkedin_profile: str                       # probable LinkedIn URL
    domain: str                                 # raw email domain
    confidence: str                             # "high" | "medium" | "low"

    # Web-researched fields (populated when not a personal email)
    overview: Optional[str] = None              # 2-3 sentence company description
    revenue: Optional[str] = None              # e.g. "$1.2B" or "Unknown"
    founded: Optional[str] = None              # founding year e.g. "2010"
    headquarters: Optional[str] = None         # e.g. "San Francisco, USA"
    website: Optional[str] = None             # company website URL
    recent_news: Optional[List[str]] = None    # up to 5 recent headlines


class ClassificationOutput(BaseModel):
    """Inquiry category produced by the Classification Agent."""

    inquiry_type: Literal["Sales", "Support", "Partnership", "General"]
    confidence: str                     # "high" | "medium" | "low"
    reasoning: str                      # one-line LLM explanation


class SummarizationOutput(BaseModel):
    """Human-readable, structured summary produced by the Summarization Agent."""

    who: str            # who is contacting (name, company, inferred role)
    what: str           # what they are asking for
    context: str        # relevant background from form + research data
    priority: Literal["High", "Medium", "Low"]
    full_summary: str   # a concise paragraph combining the above


class RoutingOutput(BaseModel):
    """Final routing decision produced by the Routing Agent."""

    department: str
    route_to: str       # destination email address
    escalate: bool      # true if the inquiry warrants immediate attention
    notes: str          # any routing caveats or special handling notes


# ── Final API response ─────────────────────────────────────────────────────────

class ProcessedInquiry(BaseModel):
    """Complete structured output returned by the API after crew execution."""

    contact: dict                   # name, email, company, linkedin
    inquiry_type: str
    summary: str
    route_to: str
    priority: str
    department: str
    escalate: bool
    routing_notes: str
    research: ResearchOutput
    classification: ClassificationOutput
