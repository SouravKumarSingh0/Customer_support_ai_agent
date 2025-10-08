# OnboardingTime.py (drop-in replacement)
# Purpose: Ask for application_table_id or phone, resolve to app id,
# fetch TAT timeline, compute the longest *forward* delay using status+substatus order,
# ignore regressions, and summarize with Gemini (GEMINI_API_KEY).

import os
import math
import argparse
import json
import logging
from typing import Optional, List, Dict, Any
import pandas as pd
from databricks import sql
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
# Match the logger name you saw in your logs
logger = logging.getLogger("OnboardingTime")

# ========================== Configuration ==========================
TABLE_TAT = os.getenv("TABLE_TAT", "silver.retail.application_tat_data")
TABLE_CUSTOMER = os.getenv("TABLE_CUSTOMER", "silver.online.customer_fact")
CUSTOMER_PHONE_COL = os.getenv("CUSTOMER_PHONE_COL", "phone_number")
CUSTOMER_APPID_COL = os.getenv("CUSTOMER_APPID_COL", "application_id")  # maps to application_table_id

# Canonical forward order (override with JSON in env if needed)
DEFAULT_STATUS_ORDER = [
    "in_progress", "applied", "approved_for_submission", "ops_approved"
]
DEFAULT_SUBSTATUS_ORDER = [
    "initiated", "tnc_accepted", "document_upload_in_progress",
    "document_uploaded", "questionnaire_submitted",
    "business_registration_details_updated", "ready_for_ops",
    "nbfc_approved", "ops_approved"
]

# Load order configurations from environment
ENV_STATUS_ORDER = os.getenv("STATUS_ORDER")
ENV_SUBSTATUS_ORDER = os.getenv("SUBSTATUS_ORDER")
STATUS_ORDER = json.loads(ENV_STATUS_ORDER) if ENV_STATUS_ORDER else DEFAULT_STATUS_ORDER
SUBSTATUS_ORDER = json.loads(ENV_SUBSTATUS_ORDER) if ENV_SUBSTATUS_ORDER else DEFAULT_SUBSTATUS_ORDER

# Create rank mappings for efficient lookups
STATUS_RANK = {s.lower(): i for i, s in enumerate(STATUS_ORDER, 1)}
SUBSTATUS_RANK = {s.lower(): i for i, s in enumerate(SUBSTATUS_ORDER, 1)}

# Terminal (final) states: if last row matches, we don't claim it's "stalled"
TERMINAL_STATUSES = set(json.loads(os.getenv("TERMINAL_STATUSES", '["ops_approved"]')))
TERMINAL_SUBSTATUSES = set(json.loads(os.getenv("TERMINAL_SUBSTATUSES", '["ops_approved"]')))

logger.info(f"Configured with {len(STATUS_ORDER)} status levels and {len(SUBSTATUS_ORDER)} substatus levels")

# ========================== Time Formatting Utilities ==========================
def format_minutes(minutes: Optional[float]) -> str:
    """Convert minutes to human-readable format (days/hours/minutes)."""
    if minutes is None:
        return "N/A"
    minutes = int(minutes)
    if minutes < 60:
        return f"{minutes} min"
    elif minutes < 1440:  # Less than 24 hours
        hours = minutes // 60
        remaining_mins = minutes % 60
        if remaining_mins == 0:
            return f"{hours} hr"
        return f"{hours} hr {remaining_mins} min"
    else:  # 24+ hours
        days = minutes // 1440
        remaining_hours = (minutes % 1440) // 60
        remaining_mins = minutes % 60
        result = f"{days} day{'s' if days != 1 else ''}"
        if remaining_hours > 0:
            result += f" {remaining_hours} hr"
        if remaining_mins > 0:
            result += f" {remaining_mins} min"
        return result

# ========================== Databricks Connection Helpers ==========================
def get_conn():
    """Establish Databricks connection with proper error handling."""
    host = os.getenv("DATABRICKS_HOST") or os.getenv("DATABRICKS_SERVER_HOSTNAME")
    http_path = os.getenv("DATABRICKS_HTTP_PATH")
    token = os.getenv("DATABRICKS_TOKEN")

    if not host or not http_path or not token:
        logger.error("Missing required Databricks environment variables")
        raise RuntimeError("Missing Databricks env vars (HOST/SERVER_HOSTNAME, HTTP_PATH, TOKEN).")

    logger.debug(f"Connecting to Databricks at {host}")
    return sql.connect(server_hostname=host, http_path=http_path, access_token=token)

def query_df(connection, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
    """Execute query and return DataFrame with proper error handling."""
    logger.debug(f"Executing query with {len(params) if params else 0} parameters")

    with connection.cursor() as cur:
        cur.execute(query, params or ())
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []

    df = pd.DataFrame.from_records(rows, columns=cols)
    logger.debug(f"Query returned {len(df)} rows")
    return df

# ========================== Phone Number Resolution ==========================
def normalize_phone_variants(raw: str) -> List[str]:
    """Generate all possible phone number variants for lookup."""
    digits = "".join(ch for ch in raw if ch.isdigit())
    variants = set()
    if len(digits) >= 10:
        core = digits[-10:]  # Last 10 digits
        variants.update({core, "0" + core, "91" + core, "+91" + core})
    if digits:
        variants.add(digits)  # Original digits
    result = [v for v in variants if v]
    logger.debug(f"Generated {len(result)} phone variants from '{raw}'")
    return result

def lookup_app_ids_by_phone(phone_input: str) -> List[str]:
    """Look up application IDs associated with a phone number."""
    logger.info(f"Looking up application IDs for phone: {phone_input}")
    variants = normalize_phone_variants(phone_input)
    if not variants:
        logger.warning("No valid phone number variants generated")
        return []
    placeholders = ",".join(["?"] * len(variants))
    query = f"""
        SELECT DISTINCT {CUSTOMER_APPID_COL} AS application_id
        FROM {TABLE_CUSTOMER}
        WHERE {CUSTOMER_PHONE_COL} IN ({placeholders})
          AND {CUSTOMER_APPID_COL} IS NOT NULL
    """
    try:
        with get_conn() as conn:
            df = query_df(conn, query, params=tuple(variants))
        if df.empty:
            logger.info("No application IDs found for the given phone number")
            return []
        app_ids = [str(x) for x in df["application_id"].dropna().astype(str).unique()]
        logger.info(f"Found {len(app_ids)} application ID(s)")
        return app_ids
    except Exception as e:
        logger.error(f"Error looking up application IDs: {e}")
        return []

# ========================== Timeline Data Retrieval ==========================
def get_application_timeline(application_table_id: str) -> pd.DataFrame:
    """Fetch the complete timeline for an application."""
    logger.info(f"Fetching timeline for application: {application_table_id}")
    query = f"""
    SELECT
      application_table_id,
      prev_status,
      prev_substatus,
      CAST(prev_updated_time AS TIMESTAMP) AS prev_updated_time,
      status,
      substatus,
      CAST(updated_time   AS TIMESTAMP)    AS updated_time,
      CAST(time_lapse_minutes AS DOUBLE)   AS time_lapse_minutes
    FROM {TABLE_TAT}
    WHERE application_table_id = ?
    ORDER BY updated_time ASC
    """
    try:
        with get_conn() as conn:
            df = query_df(conn, query, params=(application_table_id,))
        logger.info(f"Retrieved {len(df)} timeline records")
        return df
    except Exception as e:
        logger.error(f"Error fetching timeline: {e}")
        return pd.DataFrame()

# ========================== Timeline Analysis Functions ==========================
def _rank_status(s: str) -> Optional[int]:
    """Get numeric rank for a status value."""
    return STATUS_RANK.get((s or "").strip().lower(), None)

def _rank_substatus(s: str) -> Optional[int]:
    """Get numeric rank for a substatus value."""
    return SUBSTATUS_RANK.get((s or "").strip().lower(), None)

def _step_label(status: str, substatus: str) -> str:
    """Create a readable step label from status and substatus."""
    s = (status or "").strip()
    ss = (substatus or "").strip()
    if s and ss:
        return f"{s}/{ss}"
    return s or ss or "(unknown)"

def enrich_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich timeline data with additional calculated fields."""
    if df.empty:
        logger.warning("Empty timeline provided for enrichment")
        return df

    logger.info("Enriching timeline data with calculated fields")
    out = df.copy()

    # Add sequence numbers and step labels
    out["seq"] = range(1, len(out) + 1)
    out["from_step"] = out.apply(
        lambda r: _step_label(r.get("prev_status"), r.get("prev_substatus")), axis=1
    )
    out["to_step"] = out.apply(
        lambda r: _step_label(r.get("status"), r.get("substatus")), axis=1
    )
    out["transition"] = out["from_step"] + " → " + out["to_step"]

    # Calculate gap in minutes
    def gap_minutes(r):
        tlm = r.get("time_lapse_minutes")
        if tlm is not None and not (isinstance(tlm, float) and math.isnan(tlm)):
            return float(tlm)
        try:
            t0 = pd.to_datetime(r.get("prev_updated_time"), utc=True)
            t1 = pd.to_datetime(r.get("updated_time"), utc=True)
            if pd.notnull(t0) and pd.notnull(t1):
                return max(0.0, (t1 - t0).total_seconds() / 60.0)
        except Exception:
            pass
        return None

    out["gap_minutes"] = out.apply(gap_minutes, axis=1)

    # Calculate journey ranks for status and substatus
    out["prev_status_rank"] = out["prev_status"].map(_rank_status)
    out["status_rank"] = out["status"].map(_rank_status)
    out["prev_substatus_rank"] = out["prev_substatus"].map(_rank_substatus)
    out["substatus_rank"] = out["substatus"].map(_rank_substatus)

    # Determine direction of each transition
    def direction(r):
        prev_ranks = (r.get("prev_status_rank"), r.get("prev_substatus_rank"))
        curr_ranks = (r.get("status_rank"), r.get("substatus_rank"))
        if prev_ranks[0] is None or curr_ranks[0] is None:
            return "unknown"
        if prev_ranks[0] < curr_ranks[0]:
            return "forward"
        elif prev_ranks[0] > curr_ranks[0]:
            return "backward"
        else:  # Same status, check substatus
            if prev_ranks[1] is None or curr_ranks[1] is None:
                return "same"
            elif prev_ranks[1] < curr_ranks[1]:
                return "forward"
            elif prev_ranks[1] > curr_ranks[1]:
                return "backward"
            else:
                return "same"

    out["direction"] = out.apply(direction, axis=1)

    # Select and order columns
    cols = [
        "seq", "application_table_id",
        "prev_status", "prev_substatus", "prev_updated_time",
        "status", "substatus", "updated_time",
        "time_lapse_minutes", "gap_minutes",
        "direction", "transition"
    ]

    result = out[[c for c in cols if c in out.columns]]
    logger.info(f"Timeline enriched with {len(result)} transitions")
    return result

def is_terminal(status: str, substatus: str) -> bool:
    """Check if a status/substatus combination represents a terminal state."""
    s = (status or "").strip()
    ss = (substatus or "").strip()
    return (s in TERMINAL_STATUSES) or (ss in TERMINAL_SUBSTATUSES)

def summarize_timeline(df_enriched: pd.DataFrame) -> Dict[str, Any]:
    """Generate summary statistics from enriched timeline data."""
    if df_enriched.empty:
        logger.warning("Empty enriched timeline for summarization")
        return {
            "longest_transition": None, "longest_minutes": None,
            "longest_prev_time": None, "longest_updated_time": None,
            "current_status": None, "current_substatus": None,
            "current_since_minutes": None, "regression_count": 0, "is_terminal": False
        }

    logger.info("Analyzing timeline for longest forward delays")

    # Find longest *forward* transition (ignore backward movements)
    forward_transitions = df_enriched[
        df_enriched["direction"].isin(["forward", "same", "unknown"])
    ].copy()
    if forward_transitions.empty:
        forward_transitions = df_enriched.copy()

    # Use gap_minutes, fallback to time_lapse_minutes
    forward_transitions["gap_used"] = forward_transitions["gap_minutes"]
    forward_transitions.loc[
        forward_transitions["gap_used"].isna(), "gap_used"
    ] = forward_transitions["time_lapse_minutes"]

    # Handle all-NaN gap edge case
    gap_series = forward_transitions["gap_used"].fillna(-1)
    idx = gap_series.idxmax()
    longest = forward_transitions.loc[idx] if isinstance(idx, (int, float)) else forward_transitions.iloc[-1]

    # Analyze current state
    last = df_enriched.iloc[-1]
    last_time = pd.to_datetime(last.get("updated_time"), utc=True) if pd.notnull(last.get("updated_time")) else None
    now = pd.Timestamp.utcnow()
    term = is_terminal(last.get("status"), last.get("substatus"))

    current_since = None
    if not term and last_time is not None:
        current_since = int((now - last_time).total_seconds() // 60)

    # Count regressions
    regression_count = int((df_enriched["direction"] == "backward").sum())

    summary = {
        "longest_transition": longest.get("transition"),
        "longest_minutes": None if pd.isna(longest.get("gap_used")) else float(longest.get("gap_used")),
        "longest_prev_time": longest.get("prev_updated_time"),
        "longest_updated_time": longest.get("updated_time"),
        "current_status": last.get("status"),
        "current_substatus": last.get("substatus"),
        "current_since_minutes": current_since,
        "regression_count": regression_count,
        "is_terminal": term,
    }

    logger.info(
        f"Summary: longest delay={format_minutes(summary['longest_minutes'])}, "
        f"regressions={regression_count}, terminal={term}"
    )
    return summary

# ========================== Gemini AI Summary ==========================
def _truncate(s: str, n: int = 200) -> str:
    """Truncate string to specified length with ellipsis."""
    return s if s is not None and len(s) <= n else (s[:n-1] + "…")

def summarize_with_gemini(summary: Dict[str, Any]) -> str:
    """Generate AI summary using Gemini or fallback to structured summary."""
    def fallback():
        """Generate structured fallback summary."""
        lt, lm = summary.get("longest_transition"), summary.get("longest_minutes")
        cs, css = summary.get("current_status"), summary.get("current_substatus")
        cmins, reg, term = (
            summary.get("current_since_minutes"),
            summary.get("regression_count"),
            summary.get("is_terminal")
        )
        parts = []
        if lt and lm is not None:
            parts.append(f"Longest forward step: {lt} ({format_minutes(lm)}).")
        else:
            parts.append("No forward delays found.")
        if cs or css:
            step = "/".join([x for x in [cs, css] if x])
            if term:
                parts.append(f"Final step: {step}.")
            elif cmins is not None:
                parts.append(f"Current: {step} for {format_minutes(cmins)}.")
            else:
                parts.append(f"Current: {step}.")
        if reg:
            parts.append(f"{reg} regression step(s) ignored.")
        return _truncate(" ".join(parts))

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.info("GEMINI_API_KEY not found, using fallback summary")
        return fallback()

    try:
        logger.info("Generating AI summary with Gemini")
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))  # updated default
        prompt = (
            "Write one factual sentence (<=200 chars) summarizing onboarding delays. "
            f"Longest forward transition: {summary.get('longest_transition')} "
            f"({format_minutes(summary.get('longest_minutes'))}). "
            f"Current: {summary.get('current_status')}/{summary.get('current_substatus')}. "
            f"{'Final step reached.' if summary.get('is_terminal') else ''} "
            f"Regressions ignored: {summary.get('regression_count')}."
        )
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        if text:
            logger.info("Generated AI summary successfully")
            return _truncate(text)
        else:
            logger.warning("Empty response from Gemini, using fallback")
            return fallback()
    except Exception as e:
        logger.warning(f"Gemini API error: {e}, using fallback summary")
        return fallback()

# ========================== Application Resolution ==========================
def resolve_application_id(user_input: str) -> Optional[str]:
    """Resolve user input to an application_table_id."""
    logger.info(f"Resolving input: {user_input}")

    # First, try as application_table_id
    df_try = get_application_timeline(user_input)
    if not df_try.empty:
        logger.info("Input resolved as application_table_id")
        return user_input

    # If not found, treat as phone number
    logger.info("Treating input as phone number")
    app_ids = lookup_app_ids_by_phone(user_input)

    if not app_ids:
        logger.error("No application IDs found")
        return None

    if len(app_ids) == 1:
        logger.info(f"Single application ID found: {app_ids[0]}")
        return app_ids[0]

    # Multiple IDs found - let user choose
    print(f"\n📋 Multiple application IDs found for this phone:")
    for i, aid in enumerate(app_ids, 1):
        print(f"  {i}) {aid}")

    while True:
        try:
            sel = input("👉 Select an application (number): ").strip()
            idx = int(sel)
            if 1 <= idx <= len(app_ids):
                selected_id = app_ids[idx-1]
                logger.info(f"User selected application ID: {selected_id}")
                return selected_id
            else:
                print(f"❌ Please enter a number between 1 and {len(app_ids)}")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n👋 Operation cancelled by user")
            return None

# ========================== Canonical Metrics Builder (NEW) ==========================
def _to_iso(ts_val) -> str:
    try:
        if ts_val is None or (isinstance(ts_val, float) and math.isnan(ts_val)):
            return ""
        ts = pd.to_datetime(ts_val, utc=True)
        if pd.isna(ts):
            return ""
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return ""

def compute_metrics_for_customer(app_id: Optional[str] = None,
                                 phone: Optional[str] = None,
                                 email: Optional[str] = None) -> Dict[str, Any]:
    """
    Return a canonical dict used by the orchestrator:

    {
      "summary": "<<=200 chars sentence about onboarding delays>",
      "timeline": [ { "at": "...", "stage": "...", "substatus": "...", "transition": "...",
                      "gap_minutes": <float or None>, "status": "...", "prev_status": "...",
                      "prev_substatus": "..."} , ... ],
      "bottlenecks": [ { "stage": "<from→to>", "duration_minutes": <float>,
                         "from": "...", "to": "...", "start": "...", "end": "..." } ],
      "application_id": "...",
      "phone_number": "..."
    }
    """
    # 1) Resolve application_id if needed
    resolved_app_id: Optional[str] = None
    if app_id:
        resolved_app_id = str(app_id).strip()
    elif phone:
        ids = lookup_app_ids_by_phone(str(phone).strip())
        resolved_app_id = ids[0] if ids else None
    elif email:
        # no direct email resolver here; stay graceful
        resolved_app_id = None

    if not resolved_app_id:
        return {
            "summary": "No onboarding timeline available (missing application_id).",
            "timeline": [],
            "bottlenecks": [],
            "application_id": None,
            "phone_number": str(phone).strip() if phone else None
        }

    # 2) Fetch raw timeline
    raw = get_application_timeline(resolved_app_id)
    if raw.empty:
        return {
            "summary": "No onboarding timeline rows found.",
            "timeline": [],
            "bottlenecks": [],
            "application_id": resolved_app_id,
            "phone_number": str(phone).strip() if phone else None
        }

    # 3) Enrich, summarize
    enriched = enrich_timeline(raw)
    s = summarize_timeline(enriched)

    # 4) Build normalized timeline objects expected by the orchestrator
    timeline: List[Dict[str, Any]] = []
    for _, r in enriched.iterrows():
        timeline.append({
            "at": _to_iso(r.get("updated_time")),
            "time": _to_iso(r.get("updated_time")),
            "timestamp": _to_iso(r.get("updated_time")),
            "stage": (r.get("status") or "") if pd.notna(r.get("status")) else "",
            "status": (r.get("status") or "") if pd.notna(r.get("status")) else "",
            "substatus": (r.get("substatus") or "") if pd.notna(r.get("substatus")) else "",
            "prev_status": (r.get("prev_status") or "") if pd.notna(r.get("prev_status")) else "",
            "prev_substatus": (r.get("prev_substatus") or "") if pd.notna(r.get("prev_substatus")) else "",
            "transition": (r.get("transition") or "") if pd.notna(r.get("transition")) else "",
            "gap_minutes": None if pd.isna(r.get("gap_minutes")) else float(r.get("gap_minutes")),
        })

    # 5) Bottleneck list (from longest forward step)
    bottlenecks: List[Dict[str, Any]] = []
    if s.get("longest_transition"):
        trans = str(s["longest_transition"])
        parts = [p.strip() for p in trans.split("→")]
        frm = parts[0] if parts else ""
        to = parts[1] if len(parts) > 1 else ""
        bottlenecks.append({
            "stage": trans,
            "name": trans,
            "from": frm,
            "to": to,
            "duration_minutes": s.get("longest_minutes"),
            "start": _to_iso(s.get("longest_prev_time")),
            "end": _to_iso(s.get("longest_updated_time")),
        })

    # 6) Summary sentence (<=200 chars)
    summary_text = summarize_with_gemini(s)

    return {
        "summary": summary_text,
        "timeline": timeline,
        "bottlenecks": bottlenecks,
        "application_id": resolved_app_id,
        "phone_number": str(phone).strip() if phone else None
    }

# ========================== Main Application Logic ==========================
def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Onboarding TAT analyzer - analyze application timeline delays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input APP123456789          # Analyze by application ID
  %(prog)s --input +919876543210         # Analyze by phone number
  %(prog)s --input APP123 --save-csv timeline.csv  # Save timeline data
  %(prog)s --verbose                     # Enable debug logging
        """
    )

    parser.add_argument(
        "--input",
        help="Application table ID or phone number (will prompt if not provided)"
    )
    parser.add_argument(
        "--save-csv",
        help="Save enriched timeline data to CSV file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Get user input
    user_input = args.input
    if not user_input:
        print("🔍 Onboarding TAT Analyzer")
        print("=" * 40)
        user_input = input("📝 Enter application_table_id OR phone number: ").strip()

    if not user_input:
        logger.error("No input provided")
        raise SystemExit("❌ No input provided.")

    # Resolve to application ID
    app_id = resolve_application_id(user_input)
    if not app_id:
        logger.error("Could not resolve application ID")
        raise SystemExit("❌ Could not resolve an application_table_id from your input.")

    # Fetch and process timeline
    logger.info("Processing timeline data")
    raw = get_application_timeline(app_id)
    if raw.empty:
        logger.error(f"No timeline data found for {app_id}")
        raise SystemExit(f"❌ No timeline rows found for application_table_id={app_id}")

    # Enrich and analyze
    enriched = enrich_timeline(raw)
    summary = summarize_timeline(enriched)
    sentence = summarize_with_gemini(summary)

    # Display results
    print(f"\n✅ Resolved application_table_id: {app_id}")
    print(f"📊 Timeline contains {len(enriched)} transitions")

    print("\n📋 Timeline (chronological order):")
    print("=" * 80)

    # Create display version with formatted times
    display_df = enriched.copy()
    if 'gap_minutes' in display_df.columns:
        display_df['gap_formatted'] = display_df['gap_minutes'].apply(format_minutes)

    print(display_df.to_string(index=False))

    print("\n📈 Analysis Summary:")
    print("=" * 40)
    for key, value in summary.items():
        if key == 'longest_minutes':
            print(f"  {key}: {format_minutes(value)}")
        elif key == 'current_since_minutes':
            print(f"  {key}: {format_minutes(value)}")
        else:
            print(f"  {key}: {value}")

    print(f"\n🤖 AI Summary:")
    print("=" * 40)
    print(f"  {sentence}")

    # Save CSV if requested
    if args.save_csv:
        try:
            enriched.to_csv(args.save_csv, index=False)
            logger.info(f"Timeline saved to {args.save_csv}")
            print(f"\n💾 Saved timeline to {args.save_csv}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            print(f"❌ Failed to save CSV: {e}")

    logger.info("Analysis completed successfully")

# ========================== Orchestrator-Facing Adapter ==========================
def get_onboarding_metrics(app_id=None, phone=None, email=None):
    """
    Expected shape: {"summary":"...", "timeline":[...], "bottlenecks":[...]}
    This now calls compute_metrics_for_customer(...) implemented above.
    """
    try:
        res = compute_metrics_for_customer(app_id=app_id, phone=phone, email=email)
    except Exception as e:
        logger.warning(f"compute_metrics_for_customer failed: {e}")
        return {"summary": f"[Onboarding error] {type(e).__name__}: {e}", "timeline": [], "bottlenecks": []}

    # Ensure keys exist and in expected names
    if not isinstance(res, dict):
        return {"summary": str(res), "timeline": [], "bottlenecks": []}

    res.setdefault("summary", "")
    # back-compat aliases if callers expect different keys
    res.setdefault("timeline", res.get("stages") or [])
    res.setdefault("bottlenecks", res.get("delays") or [])

    return {
        "summary": res.get("summary", ""),
        "timeline": res.get("timeline", []) or [],
        "bottlenecks": res.get("bottlenecks", []) or [],
    }

if __name__ == "__main__":
    main()
