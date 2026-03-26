"""
streamlit_app.py

Streamlit demo UI for the Inquiry Processor.
Calls run_inquiry_crew() directly — no separate FastAPI server needed.

Run with:
    streamlit run streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from streamlit.components.v1 import html as components_html
from dotenv import load_dotenv

load_dotenv()

from models import InquiryFormInput
from crew import run_inquiry_crew
from utils.email_composer import compose_routing_email, send_email, smtp_configured

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Inquiry Processor",
    page_icon="📬",
    layout="wide",
)

# ── Styles ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.result-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    border-left: 4px solid #4A90E2;
}
.priority-high   { border-left-color: #e53935; }
.priority-medium { border-left-color: #fb8c00; }
.priority-low    { border-left-color: #43a047; }
.badge {
    display: inline-block;
    padding: 2px 12px;
    border-radius: 12px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-right: 6px;
}
.badge-sales       { background:#dbeafe; color:#1d4ed8; }
.badge-support     { background:#fef9c3; color:#854d0e; }
.badge-partnership { background:#ede9fe; color:#6d28d9; }
.badge-general     { background:#f3f4f6; color:#374151; }
.badge-high        { background:#fee2e2; color:#b91c1c; }
.badge-medium      { background:#ffedd5; color:#c2410c; }
.badge-low         { background:#dcfce7; color:#15803d; }
.badge-escalate    { background:#fee2e2; color:#b91c1c; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("📬 Inquiry Processor")
st.caption("AI-powered contact enrichment, classification, summarisation & routing.")
st.divider()

# ── Layout: form left, results right ──────────────────────────────────────────

left, right = st.columns([1, 1.4], gap="large")

with left:
    st.subheader("Inquiry Form")

    c1, c2 = st.columns(2)
    first_name = c1.text_input("First name *", placeholder="FirstName")
    last_name  = c2.text_input("Last name *",  placeholder="LastName")
    email      = st.text_input("Email *", placeholder="you@company.com")
    phone      = st.text_input("Phone", placeholder="+1-800-555-0000")

    subject    = st.text_input("Subject *", placeholder="Interested in enterprise plan")
    message    = st.text_area(
        "Message *",
        placeholder="Tell us what you need…",
        height=140,
    )

    col_a, col_b = st.columns(2)
    inquiry_hint = col_a.selectbox(
        "Inquiry type (optional hint)",
        ["", "Sales", "Support", "Partnership", "General"],
    )
    contact_pref = col_b.selectbox(
        "Preferred contact",
        ["email", "phone", "either"],
    )

    submitted = st.button("🚀 Process Inquiry", type="primary", use_container_width=True)

# ── Processing & results ───────────────────────────────────────────────────────

with right:
    st.subheader("Results")

    if not submitted:
        st.info("Fill in the form and click **Process Inquiry** to run the AI pipeline.")

    else:
        # ── Validate required fields client-side before hitting the LLM ──────
        errors = []
        if not first_name.strip(): errors.append("First name is required.")
        if not last_name.strip():  errors.append("Last name is required.")
        if not email.strip() or "@" not in email: errors.append("Valid email is required.")
        if not subject.strip():    errors.append("Subject is required.")
        if not message.strip():    errors.append("Message is required.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            form = InquiryFormInput(
                first_name=first_name,
                last_name=last_name,
                email=email,
                phone=phone or None,
                inquiry_type=inquiry_hint or None,
                subject=subject,
                message=message,
                preferred_contact_method=contact_pref,
            )

            # ── Run the crew with a progress bar ─────────────────────────────
            progress = st.progress(0, text="Starting AI pipeline…")
            status   = st.empty()

            steps = [
                (20,  "🌐 Web Research — searching & scraping company data…"),
                (40,  "🔍 Research Agent — enriching contact…"),
                (60,  "🏷️  Classification Agent — analysing intent…"),
                (80,  "📝 Summarization Agent — writing brief…"),
                (95,  "🔀 Routing Agent — determining destination…"),
            ]

            import threading

            result_holder = {}
            error_holder  = {}

            def _run():
                try:
                    result_holder["data"] = run_inquiry_crew(form)
                except Exception as exc:
                    error_holder["err"] = exc

            thread = threading.Thread(target=_run)
            thread.start()

            step_idx = 0
            import time
            while thread.is_alive():
                if step_idx < len(steps):
                    pct, label = steps[step_idx]
                    progress.progress(pct, text=label)
                    step_idx += 1
                time.sleep(6)       # advance label every ~6 s

            thread.join()
            progress.progress(100, text="✅ Pipeline complete!")
            time.sleep(0.4)
            progress.empty()
            status.empty()

            if "err" in error_holder:
                st.error(f"Pipeline error: {error_holder['err']}")
                st.stop()

            r = result_holder["data"]

            # ── Compose email draft ───────────────────────────────────────────
            draft = compose_routing_email(r, message)

            # ── Contact card ──────────────────────────────────────────────────
            with st.container():
                st.markdown("#### 👤 Contact")
                ca, cb = st.columns(2)
                ca.markdown(f"**Name**  \n{r.contact['name']}")
                cb.markdown(f"**Email**  \n{r.contact['email']}")
                ca.markdown(f"**Company**  \n{r.contact.get('company', '—')}")
                cb.markdown(f"**Phone**  \n{r.contact.get('phone') or '—'}")
                st.markdown(
                    f"**LinkedIn**  \n[{r.contact.get('linkedin', '—')}]({r.contact.get('linkedin', '#')})"
                )

            st.divider()

            # ── Classification & routing row ──────────────────────────────────
            st.markdown("#### 🏷️ Classification & Routing")

            itype  = r.inquiry_type.lower()
            pri    = r.priority.lower()

            badge_type = f'<span class="badge badge-{itype}">{r.inquiry_type}</span>'
            badge_pri  = f'<span class="badge badge-{pri}">Priority: {r.priority}</span>'
            badge_esc  = ('<span class="badge badge-escalate">🚨 Escalate</span>'
                          if r.escalate else "")

            st.markdown(
                f"{badge_type} {badge_pri} {badge_esc}",
                unsafe_allow_html=True,
            )

            st.markdown(f"**Route to:** `{r.route_to}` — **{r.department}** team")
            if r.routing_notes:
                st.caption(f"📌 {r.routing_notes}")

            # Classification reasoning
            with st.expander("Classification reasoning"):
                st.write(r.classification.reasoning)
                st.caption(f"Confidence: {r.classification.confidence}")

            st.divider()

            # ── Research card ─────────────────────────────────────────────────
            st.markdown("#### 🔍 Company Research")

            # Overview (if available from web research)
            if r.research.overview:
                st.markdown(
                    f'<div class="result-card">{r.research.overview}</div>',
                    unsafe_allow_html=True,
                )

            # Row 1: core identity metrics
            ra, rb, rc = st.columns(3)
            ra.metric("Company",    r.research.company_name)
            rb.metric("Industry",   r.research.industry)
            rc.metric("Size",       r.research.company_size)

            # Row 2: web-researched KPIs
            rd, re_, rf = st.columns(3)
            rd.metric("Revenue",      r.research.revenue      or "—")
            re_.metric("Founded",     r.research.founded       or "—")
            rf.metric("Headquarters", r.research.headquarters  or "—")

            # Website + confidence on same line
            ws_col, conf_col = st.columns([2, 1])
            if r.research.website:
                ws_col.markdown(
                    f"**Website:** [{r.research.website}]({r.research.website})"
                )
            conf_col.metric("Confidence", r.research.confidence.capitalize())

            # Recent news expander
            if r.research.recent_news:
                with st.expander("📰 Recent News"):
                    for headline in r.research.recent_news:
                        st.markdown(f"- {headline}")

            st.divider()

            # ── Summary ───────────────────────────────────────────────────────
            st.markdown("#### 📝 Intelligence Brief")
            pri_class = f"priority-{r.priority.lower()}"
            st.markdown(
                f'<div class="result-card {pri_class}">{r.summary}</div>',
                unsafe_allow_html=True,
            )

            st.divider()

            # ── Email preview + send ──────────────────────────────────────────
            st.markdown("#### 📧 Routing Email")

            st.markdown(
                f"**To:** `{draft.to}`  \n"
                f"**Subject:** {draft.subject}"
            )

            tab_html, tab_plain = st.tabs(["HTML preview", "Plain text"])

            with tab_html:
                components_html(draft.body_html, height=520, scrolling=True)

            with tab_plain:
                st.code(draft.body_plain, language=None)

            # Send button — active only when SMTP is configured
            if smtp_configured():
                if st.button("📤 Send Email Now", type="primary"):
                    ok = send_email(draft)
                    if ok:
                        st.success(f"Email sent to {draft.to}")
                    else:
                        st.error("Failed to send — check SMTP settings and logs.")
            else:
                st.info(
                    "Add `SMTP_HOST`, `SMTP_USER`, and `SMTP_PASSWORD` to your `.env` "
                    "to enable real sending. See `.env.example` for details.",
                    icon="ℹ️",
                )

            # ── Raw JSON toggle ───────────────────────────────────────────────
            with st.expander("View raw JSON response"):
                st.json(r.model_dump())
