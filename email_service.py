# ─── CONFIGURATION ────────────────────────────────────────────────────────────

from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging
import os
import smtplib
from typing import List

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
RECIPIENTS = [e.strip() for e in os.getenv("RECIPIENTS", "").split(",")]

# Map your DataFrame's column names here if they differ
COLUMN_MAP = {
    "date": "date",
    "home": "hometeam",
    "away": "awayteam",
    "tip":  "predicted_winner",
    "conf": "confidence",   # numeric, e.g. 69 or 69.0
}

# ─── LOGGING ──────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def confidence_color(pct: float) -> str:
    if pct >= 70:
        return "#1a7f37"   # strong green
    elif pct >= 65:
        return "#2d8fd4"   # blue
    elif pct >= 55:
        return "#e07b00"   # amber
    else:
        return "#cc3333"   # red
    
def confidence_badge(pct: float):
    if pct >= 70:
        return '<span style="background:#1a7f37;color:#fff;padding:2px 7px;border-radius:10px;font-size:11px;margin-left:6px;">STRONG</span>'
    elif pct >= 65:
        return '<span style="background:#2d8fd4;color:#fff;padding:2px 7px;border-radius:10px;font-size:11px;margin-left:6px;">GOOD</span>'
    return ""

# ─── HTML BUILDER ─────────────────────────────────────────────────────────────

def build_html(df: pd.DataFrame, generated_at: str) -> str:
    c = COLUMN_MAP  # shorthand
    
    table_rows_html = ""
    for _, row in df.iterrows():
        pct   = float(row[c["conf"]])
        color = confidence_color(pct)
        badge = confidence_badge(pct)
        table_rows_html += f"""
          <tr>
            <td>{row[c['date']]}</td>
            <td>{row[c['home']]}</td>
            <td>{row[c['away']]}</td>
            <td style="font-weight:600;">{row[c['tip']]}{badge}</td>
            <td style="color:{color};font-weight:700;">{pct:.0f}%</td>
          </tr>
        """

    return f"""
      <html>
        <head>
          <style>
            body        {{ margin:0; padding:0; background:#f0f2f5; font-family:Arial,sans-serif; }}
            .wrap       {{ max-width:720px; margin:32px auto; background:#fff;
                          border-radius:10px; overflow:hidden;
                          box-shadow:0 3px 12px rgba(0,0,0,0.12); }}
            .header     {{ background:#003087; padding:22px 28px; }}
            .header h1  {{ margin:0; color:#fff; font-size:22px; letter-spacing:0.5px; }}
            .header p   {{ margin:4px 0 0; color:#aac4f5; font-size:13px; }}
            .body       {{ padding:24px 28px; }}
            table       {{ width:100%; border-collapse:collapse; font-size:14px; }}
            th          {{ background:#003087; color:#fff; padding:11px 13px;
                          text-align:left; font-weight:600; letter-spacing:0.3px; }}
            td          {{ padding:10px 13px; border-bottom:1px solid #eaecef; color:#222; }}
            tr:last-child td {{ border-bottom:none; }}
            tr:nth-child(even) td {{ background:#f7f9fc; }}
            tr:hover td {{ background:#eef3fb; }}
            .legend     {{ margin-top:16px; padding:12px 16px; background:#f7f9fc;
                          border-radius:6px; font-size:12px; color:#555; }}
            .legend span {{ margin-right:18px; }}
            .footer     {{ padding:14px 28px; background:#f0f2f5; font-size:11px;
                          color:#999; text-align:center; }}
          </style>
        </head>
        <body>
          <div class="wrap">
            <div class="header">
              <h1>🏈 AFL Predictions</h1>
              <p>Generated {generated_at}</p>
            </div>
            <div class="body">
              <table>
                <tr>
                  <th>Date</th>
                  <th>Home</th>
                  <th>Away</th>
                  <th>Tip</th>
                  <th>Confidence</th>
                </tr>
                {table_rows_html}
              </table>
              <div class="legend">
                <span><strong style="color:#1a7f37;">■ STRONG</strong> &nbsp;70%+ confidence</span>
                <span><strong style="color:#2d8fd4;">■ GOOD</strong> &nbsp;65%+ confidence</span>
                <span><strong style="color:#e07b00;">■ AVERAGE</strong> &nbsp;55–64%</span>
                <span><strong style="color:#cc3333;">■ WEAK</strong> &nbsp;&lt;55%</span>
              </div>
            </div>
            <div class="footer">Sent automatically by your ML pipeline</div>
          </div>
        </body>
      </html>"""

def send_predictions_email(
  df: pd.DataFrame,
  recipients: List[str] = RECIPIENTS,
  sender: str | None = GMAIL_ADDRESS,
  app_password: str | None = GMAIL_APP_PASSWORD
) -> bool:
  """_summary_

    Args:
        df (pd.DataFrame): _description_
        recipients (List[str]): _description_
        send (_type_, optional): _description_. Defaults to GMAIL_ADDRESSapp_password:str=GMAIL_APP_PASSWORD.

    Returns:
        bool: _description_
    """
  assert recipients is not None, "Must set recipients in .env file"
  assert sender is not None, "Must set sender in .env file"
  assert app_password is not None, "Must set app password in .env file"

  now = datetime.now()
  generated_at = now.strftime("%A %d %B %Y, %I:%M %p")
  subject = f"AFL Predicitons - {now.strftime('%d %b %Y')}"

  html_body = build_html(df, generated_at)
  plain_text = df.to_string(index=False)

  msg = MIMEMultipart("alternative")
  msg["From"] = sender
  msg["To"] = ", ".join(recipients)
  msg["Subject"] = subject
  msg.attach(MIMEText(plain_text, "plain"))
  msg.attach(MIMEText(html_body, "html"))

  try:
      with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, app_password)
        server.sendmail(sender, recipients, msg.as_string())
      
      log.info(f"Predictions email sent to {recipients}")
      return True
  except smtplib.SMTPAuthenticationError as e:
    log.error(f"Gmail auth failed: {e}")
  except smtplib.SMTPException as e:
    log.error(f"SMTP Error: {e}")
  except Exception as e:
    log.error(f"Unexpected error: {e}")
  return False


# ─── QUICK TEST ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
  sample_df = pd.DataFrame([
    {"date": "2026-03-05", "home": "Sydney",               "away": "Carlton",          "tip": "Sydney",               "conf": 69},
    {"date": "2026-03-06", "home": "Gold Coast",           "away": "Geelong",          "tip": "Gold Coast",           "conf": 64},
    {"date": "2026-03-07", "home": "Greater Western Sydney","away": "Hawthorn",        "tip": "Greater Western Sydney","conf": 54},
    {"date": "2026-03-07", "home": "Brisbane Lions",       "away": "Western Bulldogs", "tip": "Brisbane Lions",       "conf": 63},
    {"date": "2026-03-08", "home": "St Kilda",             "away": "Collingwood",      "tip": "Collingwood",          "conf": 58},
  ])

  send_predictions_email(sample_df)
          