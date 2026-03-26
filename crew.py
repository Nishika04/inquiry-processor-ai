"""
crew.py

Assembles the four-agent inquiry-processing pipeline.

Pipeline (sequential):
  ① Research Agent      → enriches contact from email domain
  ② Classification Agent → classifies intent
  ③ Summarization Agent  → produces structured brief
  ④ Routing Agent        → emits final routing decision

Usage:
    from crew import run_inquiry_crew
    result = run_inquiry_crew(form_data)   # form_data: InquiryFormInput
"""

from __future__ import annotations

import json
import logging
import re

from agents import (
    Agent, Task, Crew,
    create_research_agent,
    create_classification_agent,
    create_summarization_agent,
    create_routing_agent,
)
from models import (
    InquiryFormInput,
    ProcessedInquiry,
    ResearchOutput,
    ClassificationOutput,
)
from utils.enrichment import enrich_contact
from utils.web_research import research_company, format_web_data_for_prompt

logger = logging.getLogger(__name__)


# ── Task builders ──────────────────────────────────────────────────────────────

def _research_task(agent: Agent, form: InquiryFormInput, enriched: dict, web_data: dict) -> Task:
    """
    Research Task

    Combines deterministic domain enrichment with live web-scraped data so
    the LLM can produce a fully populated company intelligence record.
    """
    web_section = format_web_data_for_prompt(web_data)
    web_block = (
        f"\nLive web research data:\n{web_section}\n"
        if web_section
        else "\n(No live web data available — use world knowledge only.)\n"
    )

    return Task(
        description=f"""
You have received an inquiry from the following person:

  Name  : {form.first_name} {form.last_name}
  Email : {form.email}
  Phone : {form.phone or 'not provided'}

Pre-computed enrichment data (from domain analysis):
{json.dumps(enriched, indent=2)}
{web_block}
Your job:
1. Review ALL data sources above and cross-reference them.
2. Correct any obvious errors in the pre-computed data.
3. Extract and populate all fields below using the web research data where available.
4. For fields not found in any source, use your world knowledge or set to "Unknown".
5. Return ONLY a valid JSON object matching this schema exactly:

{{
  "company_name": "<string>",
  "industry": "<string>",
  "company_size": "<string — employee range e.g. '1,000-5,000'>",
  "linkedin_profile": "<string: full LinkedIn URL>",
  "domain": "<string>",
  "confidence": "<high|medium|low>",
  "overview": "<2-3 sentence description of what the company does>",
  "revenue": "<annual revenue e.g. '$1.2B', '$500M', or 'Unknown'>",
  "founded": "<founding year e.g. '2010', or 'Unknown'>",
  "headquarters": "<city and country e.g. 'San Francisco, USA', or 'Unknown'>",
  "website": "<https://... company homepage URL>",
  "recent_news": ["<headline 1>", "<headline 2>", "<headline 3>"]
}}

For recent_news include up to 5 items; use an empty list [] if none found.
""",
        expected_output=(
            "JSON: company_name, industry, company_size, linkedin_profile, domain, "
            "confidence, overview, revenue, founded, headquarters, website, recent_news"
        ),
        agent=agent,
    )


def _classification_task(agent: Agent, form: InquiryFormInput) -> Task:
    """
    Classification Task

    The agent receives the full inquiry text and must reason about intent
    to produce one of four labels.
    """
    return Task(
        description=f"""
Classify the following customer inquiry using semantic reasoning.
Do NOT rely on simple keyword matching.

Inquiry details:
  Subject        : {form.subject}
  Message        : {form.message}
  Form hint type : {form.inquiry_type or 'not provided'}
  Contact method : {form.preferred_contact_method or 'email'}

Valid categories (pick exactly one):
  • Sales        — prospect interested in buying, pricing, demo, trial, features
  • Support      — existing customer with a problem, bug, question, or SLA issue
  • Partnership  — organisation wanting to integrate, resell, co-market, or collaborate
  • General      — anything that doesn't fit the above (press, careers, misc.)

Return ONLY a valid JSON object:

{{
  "inquiry_type": "<Sales|Support|Partnership|General>",
  "confidence": "<high|medium|low>",
  "reasoning": "<one sentence explaining the classification>"
}}
""",
        expected_output="JSON: inquiry_type, confidence, reasoning",
        agent=agent,
    )


def _summarization_task(
    agent: Agent,
    form: InquiryFormInput,
    research_task: Task,
    classification_task: Task,
) -> Task:
    """
    Summarization Task

    Receives context from both upstream tasks via the `context` field and
    synthesises a structured brief.
    """
    return Task(
        description=f"""
Create a structured intelligence brief for the following inquiry.
Use the research and classification results provided in the context section.

Original form data:
  Name    : {form.first_name} {form.last_name}
  Email   : {form.email}
  Phone   : {form.phone or 'not provided'}
  Subject : {form.subject}
  Message : {form.message}
  Contact : {form.preferred_contact_method or 'email'}

Return ONLY a valid JSON object:

{{
  "who": "<name, company, inferred role/seniority — 1-2 sentences>",
  "what": "<clear statement of what they are requesting — 1-2 sentences>",
  "context": "<relevant background: industry, company size, any signals — 1-2 sentences>",
  "priority": "<High|Medium|Low>",
  "full_summary": "<a single concise paragraph (3-5 sentences) combining who/what/context and priority>"
}}

Priority guidance:
  High   — enterprise prospect, production issue, strategic partner, urgent language
  Medium — qualified SMB prospect, clear use-case, standard support question
  Low    — general inquiry, unclear need, personal email, no urgency signals
""",
        expected_output="JSON: who, what, context, priority, full_summary",
        agent=agent,
        context=[research_task, classification_task],   # receives upstream output
    )


def _routing_task(
    agent: Agent,
    form: InquiryFormInput,
    classification_task: Task,
    summarization_task: Task,
) -> Task:
    """
    Routing Task — final task that emits the complete routing decision.
    """
    return Task(
        description=f"""
Determine the correct routing destination for the following inquiry.

Original contact:
  Name  : {form.first_name} {form.last_name}
  Email : {form.email}

Use the classification and summary from the context section.

Routing table:
  Sales        → sales@company.com
  Support      → support@company.com
  Partnership  → partnerships@company.com
  General      → info@company.com

Escalation rules — set escalate=true when ANY of the following is true:
  • Priority is "High"
  • Message contains urgent / outage / down / SLA / critical / emergency
  • Company size is 1,000+ and inquiry_type is Sales or Partnership
  • Inquiry mentions a named executive (CEO, CTO, VP, Director, etc.)

Return ONLY a valid JSON object:

{{
  "department": "<Sales|Support|Partnerships|General>",
  "route_to": "<email address>",
  "escalate": <true|false>,
  "notes": "<special handling instructions — 1 sentence, or empty string>"
}}
""",
        expected_output="JSON: department, route_to, escalate, notes",
        agent=agent,
        context=[classification_task, summarization_task],
    )


# ── Output parsing ─────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> dict:
    """
    Robustly extract a JSON object from an agent's raw string output.

    Handles:
      - Clean JSON strings
      - JSON wrapped in markdown code fences (```json … ```)
      - JSON preceded or followed by prose
    """
    if not raw:
        raise ValueError("Agent returned an empty response")

    # Strip markdown fences if present
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    # Find the outermost { … } block
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group(0))

    raise ValueError(f"No JSON object found in agent output:\n{raw[:500]}")


# ── Public entry point ─────────────────────────────────────────────────────────

def run_inquiry_crew(form: InquiryFormInput) -> ProcessedInquiry:
    """
    Execute the full inquiry-processing pipeline for a single form submission.

    Steps:
      1. Pre-compute domain enrichment (fast, deterministic Python)
      2. Build agents and tasks with context chaining
      3. Run the sequential Crew
      4. Parse + validate each agent's JSON output
      5. Return a fully typed ProcessedInquiry

    Raises:
      ValueError      — agent returned unparseable output
      EnvironmentError — missing LLM API keys
      RuntimeError    — unexpected execution failure
    """
    logger.info(
        "Starting inquiry crew for %s %s <%s>",
        form.first_name, form.last_name, form.email,
    )

    # ── 1. Deterministic pre-enrichment ───────────────────────────────────────
    enriched_raw = enrich_contact(form.first_name, form.last_name, form.email)
    logger.debug("Pre-enrichment result: %s", enriched_raw)

    # ── 1b. Live web research (skipped for personal emails) ───────────────────
    web_data: dict = {}
    if not enriched_raw.get("is_personal_email"):
        company_name = enriched_raw.get("company_name", "")
        domain       = enriched_raw.get("domain", "")
        logger.info("Running web research for company: %r", company_name)
        try:
            web_data = research_company(company_name, domain)
        except Exception as exc:
            logger.warning("Web research failed, continuing without it: %s", exc)

    # ── 2. Build agents ───────────────────────────────────────────────────────
    research_agent       = create_research_agent()
    classification_agent = create_classification_agent()
    summarization_agent  = create_summarization_agent()
    routing_agent        = create_routing_agent()

    # ── 3. Build tasks (order matters for context chaining) ───────────────────
    t_research       = _research_task(research_agent, form, enriched_raw, web_data)
    t_classification = _classification_task(classification_agent, form)
    t_summarization  = _summarization_task(
        summarization_agent, form, t_research, t_classification
    )
    t_routing = _routing_task(
        routing_agent, form, t_classification, t_summarization
    )

    # ── 4. Run the crew ───────────────────────────────────────────────────────
    crew = Crew(
        agents=[research_agent, classification_agent, summarization_agent, routing_agent],
        tasks=[t_research, t_classification, t_summarization, t_routing],
        verbose=True,
    )
    crew.kickoff()
    logger.info("Crew execution complete.")

    # ── 5. Parse each task's output ───────────────────────────────────────────
    try:
        research_data       = _extract_json(t_research.output)
        classification_data = _extract_json(t_classification.output)
        summary_data        = _extract_json(t_summarization.output)
        routing_data        = _extract_json(t_routing.output)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.error("Agent output parsing failed: %s", exc)
        raise ValueError(f"Agent output parsing failed: {exc}") from exc

    # ── 6. Validate with Pydantic and assemble final response ─────────────────
    research_obj       = ResearchOutput(**research_data)
    classification_obj = ClassificationOutput(**classification_data)

    # Deterministic routing fallback
    ROUTING_TABLE = {
      "Sales": "sales@company.com",
      "Support": "support@company.com",
      "Partnership": "partnerships@company.com",
      "General": "info@company.com",
    }

    # Prefer the routing agent's decision when it returns a sensible department
    raw_dept = routing_data.get("department")
    default_dept = "General"

    def _normalize_dept(d: str) -> str:
      if not d:
        return ""
      d = d.strip()
      # Accept minor variations
      if d.lower().startswith("sales"):
        return "Sales"
      if d.lower().startswith("support"):
        return "Support"
      if d.lower().startswith("part"):
        return "Partnership"
      if d.lower().startswith("gen") or d.lower().startswith("info"):
        return "General"
      return d

    routed_dept = _normalize_dept(raw_dept) or _normalize_dept(classification_obj.inquiry_type) or default_dept

    # If the routing agent returned an obviously wrong department (e.g. generic or
    # mismatched against classification) prefer the classification result.
    if routed_dept == "Sales" and classification_obj.inquiry_type != "Sales":
      routed_dept = _normalize_dept(classification_obj.inquiry_type) or routed_dept

    route_to_final = routing_data.get("route_to") or ROUTING_TABLE.get(routed_dept, "info@company.com")

    # Escalation: true if either agent flagged it or summary priority is High
    priority_final = summary_data.get("priority", "Medium")
    escalate_final = bool(routing_data.get("escalate", False)) or (priority_final == "High")

    return ProcessedInquiry(
      contact={
        "name":     f"{form.first_name} {form.last_name}",
        "email":    form.email,
        "company":  research_obj.company_name,
        "linkedin": research_obj.linkedin_profile,
        "phone":    form.phone or "",
      },
      inquiry_type  = classification_obj.inquiry_type,
      summary       = summary_data.get("full_summary", ""),
      route_to      = route_to_final,
      priority      = priority_final,
      department    = routed_dept,
      escalate      = escalate_final,
      routing_notes = routing_data.get("notes", ""),
      research      = research_obj,
      classification= classification_obj,
    )
