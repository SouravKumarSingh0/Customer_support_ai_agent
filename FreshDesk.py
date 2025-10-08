# Fetches Freshdesk tickets and generates a summary using Gemini,
# with performance & PII safeguards and robust fallbacks.

import os
import re
import json
import pandas as pd
from databricks import sql
from dotenv import load_dotenv

# Gemini is optional at runtime; don't crash import if not installed
try:
    import google.generativeai as genai  # pip install google-generativeai
except Exception:
    genai = None

load_dotenv()

# --- Config ---
DBX_HOST = os.getenv("DATABRICKS_SERVER_HOSTNAME") or os.getenv("DATABRICKS_HOST")
DBX_HTTP = os.getenv("DATABRICKS_HTTP_PATH")
DBX_TOKEN = os.getenv("DATABRICKS_TOKEN")

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

TABLE = "bronze.datamart.freshdesk_tickets"

PHONE_CANDIDATES = ["phone_number", "phone", "mobile", "contact_number", "customer_phone"]
ORDER_CANDIDATES = ["created_at", "updated_at", "createdon", "updatedon", "created_ts", "updated_ts"]

# --- Utility Functions ---

def normalize_phone(phone: str) -> str:
    """Keep only digits."""
    return re.sub(r"\D", "", str(phone or ""))

def sanitize_text(text: str) -> str:
    """Redact obvious PII (emails/phones) before sending to external APIs."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", text)
    text = re.sub(r"\+?\d[\d\s\-]{7,}\d", "[REDACTED_PHONE]", text)
    return text

def _fetchall_to_df(cursor) -> pd.DataFrame:
    rows = cursor.fetchall()
    cols = [c[0] for c in cursor.description] if cursor.description else []
    return pd.DataFrame(rows, columns=cols)

def _execute_to_df(cursor, query: str) -> pd.DataFrame:
    cursor.execute(query)
    try:
        return cursor.fetchall_arrow().to_pandas()
    except Exception:
        return _fetchall_to_df(cursor)

def _list_columns(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(f"DESCRIBE {TABLE}")
        rows = cur.fetchall()
    cols = []
    for r in rows:
        name = str(r[0]).strip()
        if name and not name.startswith("#") and name.lower() not in {"partition"}:
            cols.append(name)
    return cols

def _choose_case_insensitive(candidates: list[str], available: list[str]) -> str | None:
    amap = {c.lower(): c for c in available}
    for cand in candidates:
        if cand.lower() in amap:
            return amap[cand.lower()]
    return None

# --- Core Logic ---

def fetch_customer_journey(phone: str) -> pd.DataFrame:
    """Fetch all tickets for a given phone number from Databricks."""
    normalized_phone = normalize_phone(phone)
    if not normalized_phone:
        raise ValueError("Phone must contain at least one digit.")

    with sql.connect(server_hostname=DBX_HOST, http_path=DBX_HTTP, access_token=DBX_TOKEN) as conn:
        existing_cols = _list_columns(conn)

        phone_col = _choose_case_insensitive(PHONE_CANDIDATES, existing_cols)
        if not phone_col:
            raise ValueError(f"No recognizable phone column found. Available: {existing_cols}")

        order_col = _choose_case_insensitive(ORDER_CANDIDATES, existing_cols) or phone_col

        query = f"""
            SELECT *
            FROM {TABLE}
            WHERE REGEXP_REPLACE(COALESCE({phone_col}, ''), '[^0-9]', '') LIKE '%{normalized_phone}%'
            ORDER BY {order_col} ASC
        """

        with conn.cursor() as cursor:
            df = _execute_to_df(cursor, query)

    return df

def summarize_with_gemini(df: pd.DataFrame, original_phone: str) -> str:
    """Generate a human-readable summary with PII sanitization, using the original prompt."""
    if genai is None or not GEMINI_KEY:
        return f"(Gemini not configured) Tickets found: {len(df)}"

    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

    # Build sanitized ticket list
    journey_data = []
    for _, row in df.iterrows():
        ticket_info = {
            "created_at": str(row.get("created_at", "N/A")),
            "subject": sanitize_text(row.get("subject", "N/A")),
            "status": str(row.get("status", "N/A")),
            "priority": str(row.get("priority", "N/A")),
            "description_preview": sanitize_text(str(row.get("description", "N/A"))[:250])
        }
        journey_data.append(ticket_info)

    journey_json = json.dumps(journey_data, indent=2)

    # --- YOUR ORIGINAL PROMPT KEPT AS IS ---
    prompt = f"""
You are a customer support analyst. Your task is to summarize a customer's support journey based on the provided ticket data in JSON format.
The summary should be a clear, professional, and concise narrative.

Please analyze the data below and highlight:
1.  The timeline of key events.
2.  Any recurring or unresolved issues.
3.  The overall sentiment or pattern of their interactions.
4.  The current status of their latest ticket.

Customer Phone (masked): XXXX-{original_phone[-4:]}

Ticket Data:
{journey_json}

Provide the summary below:
"""

    try:
        response = model.generate_content(prompt)
        return getattr(response, "text", None) or "(Gemini returned no text)"
    except Exception as e:
        return f"(Gemini error) {e}"

# --- Main Execution ---

if __name__ == "__main__":
    try:
        phone_input = input("📞 Enter customer phone number: ").strip()
        if not phone_input:
            print("❌ Phone number cannot be empty")
            raise SystemExit(1)

        print(f"\n🔍 Searching tickets for: {phone_input} ...")
        journey_df = fetch_customer_journey(phone_input)

        if journey_df.empty:
            print("✅ No tickets found for this number.")
            raise SystemExit(0)

        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        pd.set_option("display.max_colwidth", 120)

        print(f"✅ Found {len(journey_df)} ticket(s).")
        preview_cols = [c for c in ["ticket_id","subject","status","priority","created_at","updated_at"] if c in journey_df.columns]
        if preview_cols:
            print("\n🗂 Preview (first 10):")
            print(journey_df[preview_cols].head(10).reset_index(drop=True))
        else:
            print("\n🗂 Preview (first 10):")
            print(journey_df.head(10).reset_index(drop=True))

        print("\n🤖 Generating summary with Gemini...\n")
        summary = summarize_with_gemini(journey_df, phone_input)

        print("=" * 80)
        print("📋 CUSTOMER JOURNEY SUMMARY")
        print("=" * 80)
        print(summary)
        print("=" * 80)

        base = normalize_phone(phone_input)
        journey_df.to_csv(f"freshdesk_tickets_{base}.csv", index=False)
        with open(f"summary_{base}.txt", "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"\n💾 Saved: freshdesk_tickets_{base}.csv and summary_{base}.txt")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
