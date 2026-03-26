"""
utils/email_composer.py

Generates and (optionally) sends the internal routing email that gets
dispatched to the destination team after an inquiry is processed.

Two modes:
  - Preview  : always available, returns subject + HTML + plain-text body
  - Send     : requires SMTP_* env vars; sends via any SMTP server (Gmail, etc.)
"""

from __future__ import annotations

import os
import smtplib
import logging
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

from models import ProcessedInquiry

logger = logging.getLogger(__name__)


# ── Email content builder ─────────────────────────────────────────────────────

@dataclass
class EmailDraft:
    to:          str
    subject:     str
    body_plain:  str
    body_html:   str


def compose_routing_email(result: ProcessedInquiry, form_message: str) -> EmailDraft:
    """
    Build the internal routing email sent to the destination team.

    This email is addressed TO the routed team (e.g. sales@company.com)
    and contains the full intelligence brief so the rep has everything
    they need before the first reply.
    """
    contact  = result.contact
    priority_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(result.priority, "⚪")
    escalate_note  = "⚠️  ESCALATION REQUIRED — respond within 1 hour.\n\n" if result.escalate else ""

    subject = (
        f"[{result.priority} Priority] New {result.inquiry_type} Inquiry "
        f"from {contact['name']} ({result.research.company_name})"
    )

    # ── Plain text ────────────────────────────────────────────────────────────
    body_plain = f"""\
{escalate_note}New inquiry routed to the {result.department} team.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTACT DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name     : {contact['name']}
Email    : {contact['email']}
Phone    : {contact.get('phone') or 'not provided'}
Company  : {result.research.company_name}
Industry : {result.research.industry}
Size     : {result.research.company_size}
LinkedIn : {contact.get('linkedin', '—')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INQUIRY SUMMARY   {priority_emoji} {result.priority} Priority
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Type       : {result.inquiry_type}
Confidence : {result.classification.confidence.capitalize()}
Reasoning  : {result.classification.reasoning}

{result.summary}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ORIGINAL MESSAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{form_message}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROUTING NOTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{result.routing_notes or 'No special handling required.'}

---
Processed by Inquiry Processor AI · {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""

    # ── HTML ──────────────────────────────────────────────────────────────────
    priority_color = {"High": "#e53935", "Medium": "#fb8c00", "Low": "#43a047"}.get(result.priority, "#555")
    escalate_banner = (
        f'<div style="background:#fee2e2;border-left:4px solid #e53935;padding:12px 16px;'
        f'margin-bottom:20px;border-radius:4px;">'
        f'<strong>⚠️ ESCALATION REQUIRED</strong> — please respond within 1 hour.</div>'
    ) if result.escalate else ""

    body_html = f"""\
<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;max-width:640px;margin:0 auto;color:#1f2937;">

  {escalate_banner}

  <div style="background:#f0f4ff;border-radius:8px;padding:16px 20px;margin-bottom:20px;">
    <p style="margin:0;font-size:13px;color:#6b7280;">
      New inquiry routed to the <strong>{result.department}</strong> team
    </p>
    <h2 style="margin:8px 0 0;">
      {contact['name']}
      <span style="font-size:14px;font-weight:normal;color:#6b7280;">
        · {result.research.company_name}
      </span>
    </h2>
  </div>

  <!-- Contact details -->
  <h3 style="color:#374151;border-bottom:1px solid #e5e7eb;padding-bottom:6px;">
    👤 Contact Details
  </h3>
  <table style="width:100%;border-collapse:collapse;font-size:14px;">
    <tr><td style="padding:4px 0;color:#6b7280;width:100px;">Email</td>
        <td><a href="mailto:{contact['email']}">{contact['email']}</a></td></tr>
    <tr><td style="padding:4px 0;color:#6b7280;">Phone</td>
        <td>{contact.get('phone') or '—'}</td></tr>
    <tr><td style="padding:4px 0;color:#6b7280;">Company</td>
        <td>{result.research.company_name}</td></tr>
    <tr><td style="padding:4px 0;color:#6b7280;">Industry</td>
        <td>{result.research.industry}</td></tr>
    <tr><td style="padding:4px 0;color:#6b7280;">Size</td>
        <td>{result.research.company_size}</td></tr>
    <tr><td style="padding:4px 0;color:#6b7280;">LinkedIn</td>
        <td><a href="{contact.get('linkedin','#')}">{contact.get('linkedin','—')}</a></td></tr>
  </table>

  <!-- Classification -->
  <h3 style="color:#374151;border-bottom:1px solid #e5e7eb;padding-bottom:6px;margin-top:24px;">
    🏷️ Classification &nbsp;
    <span style="font-size:13px;background:#dbeafe;color:#1d4ed8;
                 padding:2px 10px;border-radius:12px;">{result.inquiry_type}</span>
    &nbsp;
    <span style="font-size:13px;background:{priority_color}22;color:{priority_color};
                 padding:2px 10px;border-radius:12px;">{priority_emoji} {result.priority} Priority</span>
  </h3>
  <p style="font-size:14px;color:#374151;font-style:italic;">
    {result.classification.reasoning}
  </p>

  <!-- Summary -->
  <h3 style="color:#374151;border-bottom:1px solid #e5e7eb;padding-bottom:6px;margin-top:24px;">
    📝 Intelligence Brief
  </h3>
  <div style="background:#f9fafb;border-left:4px solid {priority_color};
              padding:14px 18px;border-radius:4px;font-size:14px;line-height:1.6;">
    {result.summary}
  </div>

  <!-- Original message -->
  <h3 style="color:#374151;border-bottom:1px solid #e5e7eb;padding-bottom:6px;margin-top:24px;">
    💬 Original Message
  </h3>
  <div style="background:#f9fafb;padding:14px 18px;border-radius:4px;
              font-size:14px;line-height:1.6;white-space:pre-wrap;">{form_message}</div>

  <!-- Routing notes -->
  {'<h3 style="color:#374151;border-bottom:1px solid #e5e7eb;padding-bottom:6px;margin-top:24px;">📌 Routing Notes</h3><p style="font-size:14px;">' + result.routing_notes + '</p>' if result.routing_notes else ''}

  <hr style="border:none;border-top:1px solid #e5e7eb;margin-top:32px;">
  <p style="font-size:12px;color:#9ca3af;text-align:center;">
    Processed by Inquiry Processor AI · {datetime.now().strftime('%Y-%m-%d %H:%M')}
  </p>

</body>
</html>
"""

    return EmailDraft(
        to=result.route_to,
        subject=subject,
        body_plain=body_plain,
        body_html=body_html,
    )


# ── SMTP sender ───────────────────────────────────────────────────────────────

def send_email(draft: EmailDraft) -> bool:
    """
    Send `draft` via SMTP using credentials from environment variables.

    Required env vars:
      SMTP_HOST      e.g. smtp.gmail.com
      SMTP_PORT      e.g. 587
      SMTP_USER      your sending email address
      SMTP_PASSWORD  your email password or app password

    Optional:
      SMTP_FROM_NAME  display name (defaults to "Inquiry Processor")

    Returns True on success, False on failure.

    Gmail users: generate an App Password at
    https://myaccount.google.com/apppasswords (2FA must be enabled).
    """
    host     = os.getenv("SMTP_HOST")
    port     = int(os.getenv("SMTP_PORT", "587"))
    user     = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    from_name = os.getenv("SMTP_FROM_NAME", "Inquiry Processor")

    if not all([host, user, password]):
        logger.warning("SMTP not configured — email not sent.")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = draft.subject
    msg["From"]    = f"{from_name} <{user}>"
    msg["To"]      = draft.to

    msg.attach(MIMEText(draft.body_plain, "plain"))
    msg.attach(MIMEText(draft.body_html,  "html"))

    try:
        with smtplib.SMTP(host, port) as server:
            server.ehlo()
            server.starttls()
            server.login(user, password)
            server.sendmail(user, draft.to, msg.as_string())
        logger.info("Email sent to %s", draft.to)
        return True
    except Exception as exc:
        logger.error("Failed to send email: %s", exc)
        return False


def smtp_configured() -> bool:
    """Return True if SMTP credentials are present in the environment."""
    return all([
        os.getenv("SMTP_HOST"),
        os.getenv("SMTP_USER"),
        os.getenv("SMTP_PASSWORD"),
    ])
