"""
Run notifications — a scheduled job that fails silently is worse than one that fails loudly.

  * logs/last_run.json always records the outcome of the latest run (for monitoring / the dashboard);
  * a FAILED run always sends an alert; a successful run sends the morning report only when
    ETL_NOTIFY_SUCCESS=1;
  * channels are configured by environment variables and every channel is optional:
      e-mail   SMTP_HOST [SMTP_PORT=587] SMTP_USER SMTP_PASSWORD ALERT_EMAIL_TO [ALERT_EMAIL_FROM]
      webhook  ETL_WEBHOOK_URL   (Slack / Teams / Discord incoming webhook: posts {"text", "content"})
  * notifying never raises: an unreachable mail server must not turn a good run into a failed one.
"""
import json
import logging
import os
import smtplib
import urllib.request
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

STATUS_FILE = Path(__file__).resolve().parent.parent / "logs" / "last_run.json"


def write_status(status: str, detail: str = "", duration_s: Optional[float] = None, path: Path = STATUS_FILE) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": status, "detail": detail, "duration_s": duration_s,
                                    "finished_at": datetime.now().isoformat(timespec="seconds")}, indent=2),
                        encoding="utf-8")
    except OSError as e:
        logger.warning(f"Could not write {path}: {e}")


def latest_audit_error(audit_db_path: str) -> str:
    """Error text recorded for the most recent run in the audit database ('' when unknown)."""
    import duckdb
    try:
        with duckdb.connect(audit_db_path, read_only=True) as c:
            row = c.execute("SELECT error_message FROM etl.audit_log ORDER BY start_time DESC LIMIT 1").fetchone()
        return (row[0] or "") if row else ""
    except Exception:
        return ""


def _send_email(subject: str, text: str, html: Optional[str], env) -> bool:
    host, to = env.get("SMTP_HOST"), env.get("ALERT_EMAIL_TO")
    if not host or not to:
        return False
    user, password = env.get("SMTP_USER"), env.get("SMTP_PASSWORD")
    msg = EmailMessage()
    msg["Subject"], msg["To"] = subject, to
    msg["From"] = env.get("ALERT_EMAIL_FROM") or user or "etl@localhost"
    msg.set_content(text)
    if html:
        msg.add_alternative(html, subtype="html")
    with smtplib.SMTP(host, int(env.get("SMTP_PORT", 587)), timeout=30) as smtp:
        smtp.starttls()
        if user and password:
            smtp.login(user, password)
        smtp.send_message(msg)
    return True


def _send_webhook(subject: str, text: str, env) -> bool:
    url = env.get("ETL_WEBHOOK_URL")
    if not url:
        return False
    body = json.dumps({"text": f"{subject}\n{text}", "content": f"{subject}\n{text}"[:1900]}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15):
        pass
    return True


def notify(subject: str, text: str, html: Optional[str] = None, env=None) -> list:
    """Send through every configured channel → names of the channels that delivered."""
    env = os.environ if env is None else env
    delivered = []
    for name, send in (("email", lambda: _send_email(subject, text, html, env)),
                       ("webhook", lambda: _send_webhook(subject, text, env))):
        try:
            if send():
                delivered.append(name)
        except Exception as e:
            logger.warning(f"Notification via {name} failed: {e}")
    if not delivered:
        logger.warning(f"No notification channel configured or reachable: {subject}")
    return delivered


def report_outcome(ok: bool, detail: str = "", duration_s: Optional[float] = None, report_html=None,
                   env=None, status_path: Path = STATUS_FILE) -> list:
    """Record the run and alert on failure (or send the morning report when ETL_NOTIFY_SUCCESS=1)."""
    env = os.environ if env is None else env
    write_status("SUCCESS" if ok else "FAILED", detail, duration_s, status_path)
    if not ok:
        return notify("❌ Stock ETL failed", detail or "The pipeline did not complete; production data is unchanged.", env=env)
    if env.get("ETL_NOTIFY_SUCCESS") == "1":
        try:
            html = report_html() if callable(report_html) else report_html
        except Exception as e:                       # a broken report must not hide that the run succeeded
            logger.warning(f"Could not build the e-mail report: {e}")
            html = None
        return notify("✅ Stock ETL completed", detail or "The pipeline completed.", html, env=env)
    return []
