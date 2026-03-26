"""
agents.py

Lightweight agent framework that replicates the CrewAI Agent/Task/Crew pattern
without the crewai package — fully compatible with Python 3.14+.

Architecture:
  Agent   — wraps an LLM with a persona (role + goal + backstory)
  Task    — a prompt sent to a specific agent, optionally inheriting
            the raw output of upstream tasks as context
  Crew    — runs a list of tasks sequentially and collects outputs
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── LLM client factory ────────────────────────────────────────────────────────

def _make_client():
    """
    Return an OpenAI client. Anthropic support has been removed — the
    project now requires `OPENAI_API_KEY` to be set.
    """
    if os.getenv("OPENAI_API_KEY"):
        import openai
        return openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    raise EnvironmentError(
        "No LLM API key found. Set OPENAI_API_KEY."
    )


def _call_llm(system: str, user: str) -> str:
    """
    Send a system + user message to the configured LLM and return the text reply.

    Handles both OpenAI chat completions and Anthropic messages APIs.
    """
    client = _make_client()
    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    # Some OpenAI preview/mini models only accept the default temperature (1).
    # Use 1 to avoid 'unsupported value' errors when a stricter model is set.
    response = client.chat.completions.create(
        model=model,
        temperature=1,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return response.choices[0].message.content


# ── Core primitives ───────────────────────────────────────────────────────────

@dataclass
class Agent:
    """
    An LLM persona defined by three fields (mirrors CrewAI's Agent API):

      role      — job title / function label
      goal      — what this agent is trying to accomplish
      backstory — rich context that shapes tone and expertise
    """
    role:      str
    goal:      str
    backstory: str

    def _system_prompt(self) -> str:
        return (
            f"You are a {self.role}.\n\n"
            f"Your goal: {self.goal}\n\n"
            f"Background: {self.backstory}\n\n"
            "Always follow the exact output format specified in each task. "
            "Return ONLY valid JSON when asked — no markdown fences, no prose."
        )

    def run(self, prompt: str) -> str:
        """Send `prompt` to the LLM as the user turn and return the raw reply."""
        logger.debug("[%s] prompt length=%d chars", self.role, len(prompt))
        reply = _call_llm(system=self._system_prompt(), user=prompt)
        logger.debug("[%s] reply length=%d chars", self.role, len(reply))
        return reply


@dataclass
class Task:
    """
    A unit of work assigned to a specific Agent.

    Fields:
      description     — the full prompt sent to the agent
      expected_output — human-readable description of the desired output
                        (used for logging / docs only)
      agent           — the Agent that will execute this task
      context         — list of upstream Tasks whose output will be appended
                        to the prompt before execution
      output          — populated by Crew after execution
    """
    description:     str
    expected_output: str
    agent:           Agent
    context:         list[Task] = field(default_factory=list)
    output:          Optional[str] = field(default=None, init=False)

    def build_prompt(self) -> str:
        """
        Assemble the final prompt.

        If upstream tasks are listed in `context`, their raw outputs are
        appended so the agent can reference prior work — identical to
        CrewAI's task context mechanism.
        """
        parts = [self.description]

        if self.context:
            parts.append("\n\n---\nContext from previous agents:\n")
            for i, t in enumerate(self.context, 1):
                if t.output:
                    parts.append(f"\n[Agent {i} — {t.agent.role}]:\n{t.output}")

        return "\n".join(parts)


@dataclass
class Crew:
    """
    Orchestrates a list of Tasks in sequential order.

    After kickoff(), each Task's `output` attribute is populated with the
    raw string returned by its agent.
    """
    agents: list[Agent]
    tasks:  list[Task]
    verbose: bool = True

    def kickoff(self) -> str:
        """
        Execute all tasks in order and return the final task's output.

        Each task receives the outputs of any tasks listed in its `context`
        field, enabling a data-passing chain without shared state.
        """
        last_output = ""
        for task in self.tasks:
            logger.info("Running task → agent: %s", task.agent.role)
            if self.verbose:
                print(f"\n▶ [{task.agent.role}] {task.expected_output[:80]}…")

            prompt = task.build_prompt()
            task.output = task.agent.run(prompt)
            last_output = task.output

            if self.verbose:
                # Show a trimmed preview so progress is visible in the console
                preview = task.output[:200].replace("\n", " ")
                print(f"  ✓ output: {preview}…" if len(task.output) > 200 else f"  ✓ {task.output}")

        return last_output


# ── Agent factory functions ───────────────────────────────────────────────────
# Each function is a clean constructor so agents can be unit-tested in isolation.

def create_research_agent() -> Agent:
    """
    Research Agent — enriches a contact from their email domain.

    Uses world-knowledge (no live scraping) to infer company, industry, size,
    and a probable LinkedIn URL. Signals confidence level.
    """
    return Agent(
        role="B2B Contact Research Specialist",
        goal=(
            "Extract the email domain, infer the company name, industry, and "
            "company size, and generate a probable LinkedIn profile URL for the "
            "given contact. Return a well-structured JSON object."
        ),
        backstory=(
            "You are a seasoned B2B sales-intelligence analyst who has spent "
            "years enriching CRM records using email metadata, domain patterns, "
            "and publicly available data. You know how to infer company details "
            "from a domain name even without live web access, and you always "
            "signal your confidence level clearly."
        ),
    )


def create_classification_agent() -> Agent:
    """
    Classification Agent — classifies inquiry intent with semantic LLM reasoning.

    Labels: Sales | Support | Partnership | General
    """
    return Agent(
        role="Inquiry Classification Specialist",
        goal=(
            "Analyse the inquiry subject and message using semantic reasoning "
            "and classify it into exactly one category: Sales, Support, "
            "Partnership, or General. Provide a confidence level and a concise "
            "one-sentence rationale."
        ),
        backstory=(
            "You are an expert in customer intent analysis with deep experience "
            "in B2B SaaS go-to-market teams. You understand the nuanced "
            "differences between a prospect evaluating a product (Sales), an "
            "existing customer with a problem (Support), an organisation wanting "
            "to collaborate (Partnership), and someone with a generic enquiry "
            "(General). You never resort to simple keyword matching — you reason "
            "about intent and context."
        ),
    )


def create_summarization_agent() -> Agent:
    """
    Summarization Agent — synthesises a structured intelligence brief.

    Combines form data + research + classification into a scannable brief
    with a priority estimate.
    """
    return Agent(
        role="Senior Account Intelligence Analyst",
        goal=(
            "Produce a structured, concise summary of the inquiry that gives a "
            "sales or support rep everything they need to respond intelligently "
            "within 30 seconds of reading. Include who, what, context, and a "
            "priority assessment."
        ),
        backstory=(
            "You are a principal analyst at a revenue-operations team, "
            "responsible for writing the intel briefs that sales reps read "
            "before picking up the phone. Your briefs are famous for being "
            "dense with insight yet short enough to read in under a minute. "
            "You synthesise enriched contact data, inquiry content, and "
            "classification signals into a single structured brief every time."
        ),
    )


def create_routing_agent() -> Agent:
    """
    Routing Agent — maps inquiry to the correct team + email.

    Applies escalation rules and emits final routing decision.
    """
    return Agent(
        role="Intelligent Triage & Routing Coordinator",
        goal=(
            "Given the inquiry classification, summary, and enriched contact "
            "data, determine the correct destination email address and "
            "department, decide whether to escalate the ticket, and add any "
            "special handling notes. Return a fully structured JSON routing "
            "decision."
        ),
        backstory=(
            "You are the triage lead at a fast-growing SaaS company, "
            "responsible for ensuring every inbound inquiry lands with the right "
            "team within minutes. You have deep knowledge of the company's "
            "internal routing rules and you apply good judgement when signals "
            "suggest a high-value or time-sensitive situation."
        ),
    )
