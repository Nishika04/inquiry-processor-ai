"""
main.py

FastAPI application exposing the inquiry-processing endpoint.

Endpoints:
  POST /process-inquiry   — run the full CrewAI pipeline on a form submission
  GET  /health            — lightweight health check
  GET  /                  — API metadata

Run locally:
  uvicorn main:app --reload --port 8000
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Load .env before any module that reads env vars
load_dotenv()

from models import InquiryFormInput, ProcessedInquiry   # noqa: E402  (after load_dotenv)
from crew import run_inquiry_crew                        # noqa: E402

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate required env vars at startup; nothing to clean up on shutdown."""
    missing = []
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("OPENAI_API_KEY or ANTHROPIC_API_KEY")

    if missing:
        logger.error("Missing required environment variables: %s", missing)
        raise RuntimeError(
            f"Set these environment variables before starting: {missing}"
        )

    logger.info("Inquiry Processor API started successfully.")
    yield
    logger.info("Inquiry Processor API shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Inquiry Processor API",
    description=(
        "AI-powered inquiry processing pipeline. "
        "Enriches contacts, classifies intent, summarises requests, "
        "and routes to the correct team — all in one API call."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins in development (tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing middleware ─────────────────────────────────────────────────

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.3f}s"
    return response


# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Meta"])
async def root():
    """Return API metadata."""
    return {
        "name":    "Inquiry Processor API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/health", tags=["Meta"])
async def health_check():
    """Lightweight liveness probe."""
    llm_provider = (
        "anthropic" if os.getenv("ANTHROPIC_API_KEY")
        else "openai"  if os.getenv("OPENAI_API_KEY")
        else "unconfigured"
    )
    return {
        "status":       "ok",
        "llm_provider": llm_provider,
    }


@app.post(
    "/process-inquiry",
    response_model=ProcessedInquiry,
    status_code=status.HTTP_200_OK,
    tags=["Inquiry"],
    summary="Process a contact inquiry form submission",
    response_description="Enriched, classified, summarised, and routed inquiry",
)
async def process_inquiry(form: InquiryFormInput) -> ProcessedInquiry:
    """
    Run the full CrewAI pipeline on a single inquiry form submission.

    Pipeline stages (sequential):
      1. **Research**       — enrich contact from email domain
      2. **Classification** — classify intent (Sales / Support / Partnership / General)
      3. **Summarization**  — produce a structured intelligence brief
      4. **Routing**        — determine destination team and email

    Returns a `ProcessedInquiry` object containing all enrichment data,
    the classification, the summary, and the routing decision.
    """
    logger.info(
        "Received inquiry from %s %s <%s> | subject: %r",
        form.first_name, form.last_name, form.email, form.subject,
    )

    try:
        result = run_inquiry_crew(form)
    except ValueError as exc:
        # Agent output parsing failure — 422 (unprocessable)
        logger.warning("Inquiry processing returned invalid output: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except EnvironmentError as exc:
        # Missing API keys — 503
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        # Catch-all — re-raise so the global handler formats it consistently
        logger.exception("Unexpected error processing inquiry")
        raise

    logger.info(
        "Inquiry processed | type=%s route=%s priority=%s escalate=%s",
        result.inquiry_type, result.route_to, result.priority, result.escalate,
    )
    return result


# ── Dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("ENV", "development") == "development",
        log_level="info",
    )
