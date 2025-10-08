# TeleCalling_Databricks.py
# Prompts for an identifier, fetches ALL telecalling data from Databricks (no year filter),
# resolves columns case-insensitively (incl. remark/remarks), and summarizes via Gemini.

import os
import sys
import pandas as pd
from datetime import datetime
from databricks import sql
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Optional, List, Tuple

# ---------------------------
# Setup & Config
# ---------------------------
load_dotenv()
pd.set_option("display.max_columns", None)

# Required env
HOST = os.getenv("DATABRICKS_HOST")
HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
TOKEN = os.getenv("DATABRICKS_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Optional env (defaults)
CATALOG = os.getenv("DATABRICKS_CATALOG", "bronze")
SCHEMA = os.getenv("DATABRICKS_SCHEMA", "datamart")
TABLE = os.getenv("TELECALLING_TABLE", "telecalling_data")
TAG_FILTER = os.getenv("TELECALLING_TAG", "onboarding")  # set to "" to disable tag filter

# Column hints
COL_AGENT = os.getenv("COL_AGENT", "agent_name")
COL_DISP = os.getenv("COL_DISP", "disposition")
COL_REMARK1 = os.getenv("COL_REMARK", "remark")
COL_REMARK2 = os.getenv("COL_REMARK_ALT", "remarks")
COL_DATE = os.getenv("COL_DATE", "call_date")
COL_TAG = os.getenv("COL_TAG", "tag")
COL_APPID = os.getenv("COL_APPID", "application_id")
COL_PHONE = os.getenv("COL_PHONE", "phone_number")

# Validate env
REQUIRED_ENV = {
    "DATABRICKS_HOST": HOST,
    "DATABRICKS_HTTP_PATH": HTTP_PATH,
    "DATABRICKS_TOKEN": TOKEN,
    "GEMINI_API_KEY": GEMINI_API_KEY,
}
missing = [k for k, v in REQUIRED_ENV.items() if not v]
if missing:
    raise ValueError(f"Missing required env vars: {', '.join(missing)}")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # <-- added
model = genai.GenerativeModel(GEMINI_MODEL)                   # <-- updated

# ---------------------------
# Database Helpers
# ---------------------------
def _fqtn() -> str:
    """Return fully qualified table name."""
    return f"{CATALOG}.{SCHEMA}.{TABLE}"

def _list_columns(conn) -> List[str]:
    query = f"""
        SELECT column_name
        FROM {CATALOG}.information_schema.columns
        WHERE table_catalog = ? AND table_schema = ? AND table_name = ?
    """
    df_cols = pd.read_sql(query, conn, params=[CATALOG, SCHEMA, TABLE])
    return df_cols["column_name"].tolist()

def _resolve(colnames: List[str], candidates: List[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in colnames}
    for cand in candidates:
        if cand and cand.lower() in lowered:
            return lowered[cand.lower()]
    return None

def _build_where_clause_and_params(
    col_tag: Optional[str],
    id_col: str,
    identifier_value: str
) -> Tuple[str, list]:
    """Build WHERE clause and parameters for the query (no date filter)."""
    where_parts = []
    params = []

    if TAG_FILTER and col_tag:
        where_parts.append(f"lower({col_tag}) = lower(?)")
        params.append(TAG_FILTER)

    where_parts.append(f"{id_col} = ?")
    params.append(identifier_value)

    return " AND ".join(where_parts), params

def _build_select_columns(
    col_agent: Optional[str],
    col_disp: Optional[str],
    col_remark: Optional[str],
    col_date: str
) -> List[str]:
    select_cols = []
    if col_agent:
        select_cols.append(f"{col_agent} AS agent_name")
    if col_disp:
        select_cols.append(f"{col_disp} AS disposition")
    if col_remark:
        select_cols.append(f"{col_remark} AS remark")
    else:
        select_cols.append("CAST(NULL AS STRING) AS remark")
    select_cols.append(f"{col_date} AS call_date")
    return select_cols

def fetch_telecalling_data(identifier_type: str, identifier_value: str, latest_n: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch ALL telecalling data (no year filter).
    identifier_type: 'phone' or 'application' (email not supported at DB level)
    latest_n: optional cap on returned rows (most recent first)
    """
    limit_sql = ""
    if isinstance(latest_n, int) and latest_n > 0:
        limit_sql = f" LIMIT {int(latest_n)}"

    with sql.connect(server_hostname=HOST, http_path=HTTP_PATH, access_token=TOKEN) as conn:
        existing = _list_columns(conn)

        col_agent = _resolve(existing, [COL_AGENT, "agent_name"])
        col_disp = _resolve(existing, [COL_DISP, "disposition"])
        col_remark = _resolve(existing, [COL_REMARK1, COL_REMARK2, "remark", "remarks", "notes"])
        col_date = _resolve(existing, [COL_DATE, "call_date", "call_dt", "timestamp", "created_at"])
        col_tag = _resolve(existing, [COL_TAG, "tag"])
        col_appid = _resolve(existing, [COL_APPID, "application_id", "app_id"])
        col_phone = _resolve(existing, [COL_PHONE, "phone_number", "msisdn", "mobile"])

        if col_date is None:
            raise RuntimeError(f"Required date column not found. Available: {existing}")

        id_col = col_phone if identifier_type == "phone" else col_appid
        if id_col is None:
            # try the other way around if hinted column missing
            id_col = col_appid if identifier_type == "phone" else col_phone
        if id_col is None:
            raise RuntimeError("Neither phone nor application_id column found on the table.")

        where_sql, params = _build_where_clause_and_params(col_tag, id_col, identifier_value)
        select_cols = _build_select_columns(col_agent, col_disp, col_remark, col_date)

        query = f"""
            SELECT {', '.join(select_cols)}
            FROM {_fqtn()}
            WHERE {where_sql}
            ORDER BY {col_date} DESC{limit_sql}
        """

        # Using pandas read_sql here is fine; Databricks connector returns DB-API2.
        df = pd.read_sql(query, conn, params=params)

    if not df.empty:
        # normalize datetime
        df["call_date"] = pd.to_datetime(df["call_date"], errors="coerce", utc=True)
        for col in ["agent_name", "disposition", "remark"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
    return df

# ---------------------------
# Summary Helpers
# ---------------------------
def _format_timeline(df: pd.DataFrame) -> str:
    if df.empty:
        return "No calls found."
    lines = []
    for _, row in df.iterrows():
        dt = row.get("call_date")
        date_str = (dt.strftime("%Y-%m-%d") if isinstance(dt, (datetime, pd.Timestamp)) and not pd.isna(dt) else "unknown date")
        agent = (row.get("agent_name") or "Agent").strip() or "Agent"
        disp = (row.get("disposition") or "N/A").strip()
        rem = (row.get("remark") or "no remarks").strip()
        lines.append(f"- {date_str}: {agent} — disposition: {disp}. Remark: {rem}")
    return "\n".join(lines)

def summarize_with_gemini(df: pd.DataFrame, who: str) -> str:
    if df.empty:
        return "No telecalling records found for this customer."
    timeline = _format_timeline(df)
    prompt = f"""You are a support QA analyst. Summarize the telecalling history for {who}.
Rules:
- 2–4 short sentences, strictly factual.
- Mention agent names, dates, and clear outcomes.
- Then add a compact bullet timeline (≤1 line per call).
- Entire output ≤ 900 characters.

Calls:
{timeline}"""
    try:
        resp = model.generate_content(prompt)
        return (getattr(resp, "text", "") or "").strip() if resp else "[No summary returned]"
    except Exception as e:
        return f"[Gemini error] {e}"

def _determine_identifier_type(identifier: str, forced_type: Optional[str]) -> str:
    if forced_type in {"phone", "application"}:
        return forced_type
    return "phone" if identifier.isdigit() and len(identifier) >= 10 else "application"

def _parse_command_line_args() -> Tuple[str, Optional[str]]:
    forced_type = None
    if len(sys.argv) >= 3 and sys.argv[1] == "--type":
        forced_type = sys.argv[2].strip().lower()
        identifier = input("Enter the identifier value: ").strip()
    else:
        identifier = input("Enter the customer's Application ID or Phone Number: ").strip()
    return identifier, forced_type

# ---------------------------
# Main
# ---------------------------
def main():
    identifier, forced_type = _parse_command_line_args()
    id_type = _determine_identifier_type(identifier, forced_type)
    print(f"Searching by {id_type}... (tag={'none' if not TAG_FILTER else TAG_FILTER})")
    df_records = fetch_telecalling_data(id_type, identifier)
    who = f"{COL_APPID}={identifier}" if id_type == "application" else f"{COL_PHONE}={identifier}"
    summary = summarize_with_gemini(df_records, who)
    print("\n--- AI Summary ---")
    print(summary)
    print("\n--- Records Found ---")
    print(f"rows={len(df_records)}")
    if not df_records.empty:
        print(df_records.head(20).to_string(index=False))

# --- Normalized entry point for the orchestrator (place above the main-guard) ---
def summarize_calls(app_id=None, phone=None, email=None, latest_n=None):
    """
    Returns:
      {
        "summary": "<high-level summary>",
        "calls": [ { ...raw telecall row dicts... } ]
      }
    Uses your existing fetch_telecalling_data(...) and summarize_with_gemini(...).
    """
    # Decide which identifier to use (priority: app_id > phone > email)
    identifier = app_id or phone or email
    if not identifier:
        return {"summary": "No identifier provided", "calls": []}

    # Figure out id_type similar to CLI
    if app_id:
        id_type = "application"
    elif phone:
        id_type = "phone"
    else:
        # Email not supported in current DB schema; bail gracefully
        return {"summary": "Email-based lookup is not supported for telecalling data.", "calls": []}

    # Pull rows
    try:
        df_records = fetch_telecalling_data(id_type, str(identifier), latest_n=latest_n)
    except Exception as e:
        return {"summary": f"Fetch error: {type(e).__name__}: {e}", "calls": []}

    # Handle empty
    try:
        is_empty = df_records is None or getattr(df_records, "empty", False)
    except Exception:
        is_empty = True
    if is_empty:
        return {"summary": "No telecalling records found", "calls": []}

    # Build the 'who' label like your main() does
    who = f"{COL_APPID}={identifier}" if id_type == "application" else f"{COL_PHONE}={identifier}"

    # Summarize with your existing Gemini summarizer
    try:
        summary = summarize_with_gemini(df_records, who)
    except Exception as e:
        summary = f"[summarize error] {type(e).__name__}: {e}"

    # Return normalized dict; calls = raw rows
    try:
        calls = df_records.to_dict(orient="records")
    except Exception:
        calls = []

    return {"summary": summary or "", "calls": calls}
# --- end wrapper ---

if __name__ == "__main__":
    main()
