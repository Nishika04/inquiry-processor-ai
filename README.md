# Inquiry Processor — CrewAI + FastAPI

An AI-powered pipeline that transforms raw contact-form submissions into enriched, classified, summarised, and routed inquiry records.

---

## Architecture

```
POST /process-inquiry
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                        CrewAI Crew                      │
│  (sequential process — each agent feeds the next)       │
│                                                         │
│  ① Research Agent                                       │
│     • Extract email domain                              │
│     • Infer company name, industry, size                │
│     • Generate probable LinkedIn URL                    │
│                                                         │
│  ② Classification Agent                                 │
│     • Semantic LLM reasoning (not keyword matching)     │
│     • Sales | Support | Partnership | General           │
│                                                         │
│  ③ Summarization Agent                                  │
│     • Who is contacting + what they want                │
│     • Context (company, inferred role)                  │
│     • Priority: High | Medium | Low                     │
│                                                         │
│  ④ Routing Agent                                        │
│     • Map type → department email                       │
│     • Escalation decision                               │
│     • Handling notes                                    │
└─────────────────────────────────────────────────────────┘
        │
        ▼
   ProcessedInquiry (JSON)
```

---

## Project Structure

```
inquiry_processor/
├── main.py              FastAPI app, endpoints, middleware
├── agents.py            CrewAI agent factory functions
├── crew.py              Task definitions + crew orchestration
├── models.py            Pydantic schemas (input / output / intermediate)
├── utils/
│   ├── __init__.py
│   └── enrichment.py   Domain-based company enrichment utilities
├── tests/
│   └── test_enrichment.py
├── .env.example         Environment variable template
├── requirements.txt
└── README.md
```

---

## Compatibility

| Python | Status |
|--------|--------|
| 3.10 – 3.14+ | ✅ Supported |

The pipeline is built on `openai` / `anthropic` SDKs directly — no `crewai`
dependency, so there are no Python version constraints.

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url>
cd inquiry_processor

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY or ANTHROPIC_API_KEY
```

### 3. Run the API

```bash
uvicorn main:app --reload --port 8000
```

The API is now live at **http://localhost:8000**

- Interactive docs: http://localhost:8000/docs
- ReDoc:           http://localhost:8000/redoc
- Health check:    http://localhost:8000/health

---

## API Reference

### `POST /process-inquiry`

**Request body**

```json
{
  "first_name": "Sarah",
  "last_name": "Chen",
  "email": "sarah.chen@stripe.com",
  "phone": "+1-415-555-0192",
  "inquiry_type": "",
  "subject": "Interested in enterprise plan — need custom SLA",
  "message": "Hi, I'm the Head of Engineering at Stripe and we're evaluating your platform for our internal tooling. We have ~4,000 engineers and need a custom SLA with 99.99% uptime guarantee. Can we schedule a call with your enterprise team?",
  "preferred_contact_method": "phone"
}
```

**Response (200 OK)**

```json
{
  "contact": {
    "name": "Sarah Chen",
    "email": "sarah.chen@stripe.com",
    "company": "Stripe",
    "linkedin": "https://www.linkedin.com/in/sarahchen",
    "phone": "+1-415-415-555-0192"
  },
  "inquiry_type": "Sales",
  "summary": "Sarah Chen, Head of Engineering at Stripe (FinTech, 1,000–5,000 employees), is evaluating the platform for internal developer tooling at scale. She represents ~4,000 engineers and is requesting an enterprise plan with a custom 99.99% SLA. This is a high-value enterprise Sales opportunity requiring urgent follow-up.",
  "route_to": "sales@company.com",
  "priority": "High",
  "department": "Sales",
  "escalate": true,
  "routing_notes": "Enterprise prospect from Stripe. Requesting custom SLA — loop in enterprise AE and solutions engineer.",
  "research": {
    "company_name": "Stripe",
    "industry": "FinTech",
    "company_size": "1,000-5,000",
    "linkedin_profile": "https://www.linkedin.com/in/sarahchen",
    "domain": "stripe.com",
    "confidence": "high"
  },
  "classification": {
    "inquiry_type": "Sales",
    "confidence": "high",
    "reasoning": "The contact is evaluating the product for enterprise purchase and requests a custom SLA, which is a clear enterprise sales signal."
  }
}
```

---

## cURL Example

```bash
curl -X POST http://localhost:8000/process-inquiry \
  -H "Content-Type: application/json" \
  -d '{
    "first_name": "Sarah",
    "last_name": "Chen",
    "email": "sarah.chen@stripe.com",
    "phone": "+1-415-555-0192",
    "inquiry_type": "",
    "subject": "Interested in enterprise plan — need custom SLA",
    "message": "Hi, I am the Head of Engineering at Stripe and we are evaluating your platform for internal tooling. We have ~4,000 engineers and need a custom SLA with 99.99% uptime guarantee.",
    "preferred_contact_method": "phone"
  }'
```

---

## Python Client Example

```python
import httpx

payload = {
    "first_name": "Marcus",
    "last_name": "Webb",
    "email": "m.webb@gmail.com",
    "phone": None,
    "inquiry_type": "",
    "subject": "Your app keeps crashing on iOS 17",
    "message": "Hey, ever since I updated to iOS 17 the app crashes every time I try to export a report. This is blocking my whole workflow. Please help ASAP.",
    "preferred_contact_method": "email"
}

response = httpx.post("http://localhost:8000/process-inquiry", json=payload)
result = response.json()

print(f"Type     : {result['inquiry_type']}")
print(f"Priority : {result['priority']}")
print(f"Route to : {result['route_to']}")
print(f"Escalate : {result['escalate']}")
print(f"Summary  : {result['summary']}")
```

---

## LLM Configuration

| Variable          | Default                     | Description                         |
|-------------------|-----------------------------|-------------------------------------|
| `OPENAI_API_KEY`  | —                           | OpenAI key (use this **or** Claude) |
| `OPENAI_MODEL`    | `gpt-4o-mini`               | Any OpenAI chat model               |
| `ANTHROPIC_API_KEY` | —                         | Anthropic key                       |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Any Claude model                    |

If both keys are set, **Anthropic takes priority**.

---

## Routing Table

| Inquiry Type | Department   | Destination Email        |
|--------------|-------------|--------------------------|
| Sales        | Sales        | sales@company.com        |
| Support      | Support      | support@company.com      |
| Partnership  | Partnerships | partnerships@company.com |
| General      | General      | info@company.com         |

---

## Escalation Rules

A ticket is auto-escalated (`escalate: true`) when **any** of the following apply:

- Priority assessed as **High**
- Message contains: `urgent`, `outage`, `down`, `SLA`, `critical`, `emergency`
- Company size ≥ 1,000 employees **and** type is Sales or Partnership
- Contact mentions a C-level / VP / Director title

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Extending the System

**Add a new classification category**
- Update the `Literal` in `models.py` → `ClassificationOutput.inquiry_type`
- Update the routing table in `crew.py` → `_routing_task`
- Update the routing agent's description in `agents.py`

**Swap in a real enrichment provider**
- Replace the functions in `utils/enrichment.py` with Clearbit / Hunter.io API calls
- The agent prompts are provider-agnostic — no changes needed there

**Persist results**
- Add a database session dependency to `main.py`
- Call `db.save(result)` after `run_inquiry_crew()` returns
