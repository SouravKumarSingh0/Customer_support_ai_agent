#!/usr/bin/env python3
"""
Partner Dashboard -> Databricks client with Gemini summarization.
Fetches and summarizes partner dashboard comments for a customer using either
an application ID or a phone number.

This version adds a normalized "comments" list in the returned dict so the
orchestrator can consume timestamps + comment text consistently.
"""

import os
import re
import sys
import time
import json
import logging
import argparse
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager
from functools import lru_cache
from dataclasses import dataclass

import pandas as pd
from databricks import sql

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# ---------- Configuration ----------
TABLE_COMMENTS = "bronze.retail.application_comment"
TABLE_CUSTOMER = "silver.online.customer_fact"
MAX_LIMIT = 5000
DEFAULT_TIMEOUT_SEC = 90
DEFAULT_RETRIES = 2
DEFAULT_LATEST_N = 5

GEMINI_SYSTEM_PROMPT = (
    "You are a support summarizer. Given a chronological set of internal partner-dashboard comments, "
    "produce ONE short paragraph (<= 120 words) that captures the customer's issue(s), key actions, "
    "current status, and next steps if any. Avoid PII. Be precise and factual."
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class for database and API settings."""
    server_hostname: str
    http_path: str
    token: str
    gemini_api_key: str
    gemini_model_name: str = "gemini-2.5-flash"  # updated default
    timeout_sec: int = DEFAULT_TIMEOUT_SEC
    retries: int = DEFAULT_RETRIES

    @classmethod
    def from_env(cls) -> 'Config':
        """Create config from environment variables."""
        server_hostname = _clean_host(os.getenv("DATABRICKS_HOST"))
        http_path = os.getenv("DATABRICKS_HTTP_PATH")
        token = os.getenv("DATABRICKS_TOKEN")
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        gemini_model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        missing = [k for k, v in {
            "DATABRICKS_HOST": server_hostname,
            "DATABRICKS_HTTP_PATH": http_path,
            "DATABRICKS_TOKEN": token,
            "GEMINI_API_KEY": gemini_api_key,
        }.items() if not v]

        if missing:
            raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

        return cls(
            server_hostname=server_hostname,
            http_path=http_path,
            token=token,
            gemini_api_key=gemini_api_key,
            gemini_model_name=gemini_model_name
        )

def _clean_host(h: Optional[str]) -> Optional[str]:
    """Clean hostname by removing protocol and trailing slashes."""
    if not h:
        return None
    return h.replace("https://", "").replace("http://", "").strip("/")

def _only_digits(s: str) -> str:
    """Extract only digits from string."""
    return re.sub(r"\D+", "", s or "")

class DatabaseError(Exception):
    """Custom exception for database-related errors."""
    pass

class PartnerDashboardClient:
    """Client for fetching and summarizing partner dashboard comments."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.from_env()
        self._validate_dependencies()
        self._setup_gemini()
        self._columns_cache: Optional[List[str]] = None
        self._ts_col_cache: Optional[str] = None

        logger.info(f"[Databricks] host={self.config.server_hostname} http_path={self.config.http_path}")

    # ---------- Setup ----------
    def _validate_dependencies(self) -> None:
        """Validate required dependencies are available."""
        if genai is None:
            raise RuntimeError(
                "google-generativeai is not installed. Run: pip install google-generativeai"
            )

    def _setup_gemini(self) -> None:
        """Configure Gemini API."""
        genai.configure(api_key=self.config.gemini_api_key)

    @contextmanager
    def _db_connection(self):
        """Context manager for database connections with proper resource cleanup."""
        conn = None
        try:
            conn = sql.connect(
                server_hostname=self.config.server_hostname,
                http_path=self.config.http_path,
                access_token=self.config.token,
                _session_params={"statement_timeout_in_seconds": self.config.timeout_sec},
            )
            yield conn
        finally:
            if conn:
                try:
                    conn.close()
                except Exception as e:
                    logger.warning(f"Error closing connection: {e}")

    # ---------- DB Helpers ----------
    def _run(self, query: str, retries: Optional[int] = None, timeout_sec: Optional[int] = None) -> pd.DataFrame:
        """Execute SQL query with retry logic and proper error handling."""
        retries = retries or self.config.retries
        timeout_sec = timeout_sec or self.config.timeout_sec

        last_err = None
        for attempt in range(retries + 1):
            try:
                with self._db_connection() as conn, conn.cursor() as cur:
                    cur.execute(query)
                    rows = cur.fetchall()
                    cols = [c[0] for c in cur.description] if cur.description else []
                    return pd.DataFrame.from_records(rows, columns=cols)
            except Exception as e:
                last_err = e
                logger.warning(f"Query attempt {attempt + 1} failed: {e}")
                if attempt < retries:
                    sleep_time = 1.2 * (attempt + 1)
                    logger.info(f"Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                else:
                    break

        error_msg = f"Query failed after {retries + 1} attempts. Last error: {last_err}"
        logger.error(f"{error_msg}\nSQL: {query}")
        raise DatabaseError(error_msg) from last_err

    @lru_cache(maxsize=10)
    def _list_columns(self, table_fqn: str) -> Tuple[str, ...]:
        """List columns for a table with caching."""
        if table_fqn == TABLE_COMMENTS and self._columns_cache is not None:
            return tuple(self._columns_cache)

        df = self._run(f"SELECT * FROM {table_fqn} LIMIT 0")
        cols = list(df.columns)

        if table_fqn == TABLE_COMMENTS:
            self._columns_cache = cols

        return tuple(cols)

    def _timestamp_column(self) -> str:
        """Determine the best timestamp column to use for ordering."""
        if self._ts_col_cache:
            return self._ts_col_cache

        available = set(self._list_columns(TABLE_COMMENTS))
        timestamp_candidates = [
            "created_time", "created_at", "comment_time",
            "insert_time", "event_time", "updated_time", "updated_at", "id"
        ]

        for candidate in timestamp_candidates:
            if candidate in available:
                self._ts_col_cache = candidate
                return candidate

        # Fallback to id if no timestamp columns found
        self._ts_col_cache = "id"
        return self._ts_col_cache

    def _select_cols_comments(self) -> str:
        """Build SELECT clause for comments query."""
        available = set(self._list_columns(TABLE_COMMENTS))
        ts_col = self._timestamp_column()
        preferred = ["id", "application_id", "comment", "status", "remarks", "resolved_by", "team_id"]

        cols = [c for c in preferred if c in available]

        # Handle timestamp column
        if ts_col not in cols:
            cols.append(f"{ts_col} as ts")
        else:
            # ensure alias 'ts'
            cols[cols.index(ts_col)] = f"{ts_col} as ts"

        return ", ".join(cols) if cols else "*"

    # ---------- Resolvers ----------
    def resolve_app_ids_by_phone(self, phone_number: str) -> List[int]:
        """Resolve application IDs by phone number."""
        phone = _only_digits(phone_number)
        if not phone:
            return []

        query = f"SELECT DISTINCT application_id FROM {TABLE_CUSTOMER} WHERE phone_number = '{phone}'"
        try:
            df = self._run(query)
            if "application_id" not in df or df.empty:
                return []
            app_ids = df["application_id"].dropna().unique().tolist()
            return [int(x) for x in app_ids if pd.notna(x)]
        except Exception as e:
            logger.error(f"Error resolving app IDs for phone {phone}: {e}")
            return []

    def resolve_phone_by_app(self, application_id: int) -> Optional[str]:
        """Resolve phone number by application ID."""
        query = f"SELECT phone_number FROM {TABLE_CUSTOMER} WHERE application_id = '{int(application_id)}' LIMIT 1"
        try:
            df = self._run(query)
            if "phone_number" not in df or df.empty:
                return None
            phone = str(df.iloc[0]["phone_number"])
            return _only_digits(phone) if phone != 'nan' else None
        except Exception as e:
            logger.error(f"Error resolving phone for app ID {application_id}: {e}")
            return None

    # ---------- Fetch / Summarize ----------
    def fetch_comments_for_app(self, application_id: int, limit: int = 2000) -> pd.DataFrame:
        """Fetch comments for a specific application ID."""
        limit = min(max(1, int(limit)), MAX_LIMIT)
        select_cols = self._select_cols_comments()
        query = f"""
            SELECT {select_cols}
            FROM {TABLE_COMMENTS}
            WHERE application_id = {int(application_id)}
              AND comment IS NOT NULL
              AND length(trim(comment)) >= 2
              AND lower(trim(comment)) NOT IN ('.', '..', '...', 'na', 'n/a', '-')
            ORDER BY ts ASC, id ASC
            LIMIT {limit}
        """
        try:
            return self._run(query)
        except Exception as e:
            logger.error(f"Error fetching comments for app ID {application_id}: {e}")
            return pd.DataFrame()

    def summarize_thread_with_gemini(self, comments_df: pd.DataFrame) -> str:
        """Summarize comment thread using Gemini AI."""
        if comments_df is None or comments_df.empty:
            return "No partner dashboard discussion found for this customer."

        # Extract and format comments
        lines = []
        for _, row in comments_df.iterrows():
            comment_text = str(row.get('comment') or row.get('remarks') or '').strip()
            if comment_text:
                timestamp = row.get('ts') or ''
                clean_comment = re.sub(r'\s+', ' ', comment_text)
                lines.append(f"[{timestamp}] {clean_comment}")

        if not lines:
            return "No valid comments found to summarize."

        # Limit input size for API efficiency
        joined = "\n".join(lines[-1200:])
        prompt = f"{GEMINI_SYSTEM_PROMPT}\n\nCOMMENTS:\n{joined}"

        try:
            model = genai.GenerativeModel(self.config.gemini_model_name)
            resp = model.generate_content(prompt)
            summary = (resp.text or "").strip() if getattr(resp, "text", None) else ""
            return summary or "Summary unavailable."
        except Exception as e:
            logger.error(f"Error generating summary with Gemini: {e}")
            return "Summary generation failed."

    # ---------- Normalization ----------
    @staticmethod
    def _normalize_comment_row(row: pd.Series) -> Dict[str, Any]:
        """
        Shape each comment row for the orchestrator's timeline builder.
        Fields chosen to match its expectations:
          - timestamp/created_at/time
          - comment/application_comment
          - user/author/created_by (best-effort)
          - id (if present)
        """
        ts = str(row.get("ts") or "").strip()
        txt = str(row.get("comment") or row.get("remarks") or "").strip()
        user = (str(row.get("resolved_by")) if row.get("resolved_by") is not None else "").strip()
        status = (str(row.get("status")) if row.get("status") is not None else "").strip()

        out = {
            "id": int(row["id"]) if "id" in row and pd.notna(row["id"]) else None,
            "timestamp": ts,
            "created_at": ts,  # duplicate for convenience
            "time": ts,
            "comment": txt,
            "application_comment": txt,
        }
        if user:
            out["user"] = user
            out["author"] = user
            out["created_by"] = user
        if status:
            out["status"] = status
        return out

    # ---------- Public summaries ----------
    def summarize_by_application_id(self, application_id: int, latest_n: int = DEFAULT_LATEST_N) -> Dict[str, Any]:
        """Summarize comments by application ID and include normalized 'comments' list."""
        app_id = int(application_id)
        phone = self.resolve_phone_by_app(app_id)
        df = self.fetch_comments_for_app(app_id)
        summary = self.summarize_thread_with_gemini(df)

        # Latest normalized comments
        comments_norm: List[Dict[str, Any]] = []
        if not df.empty:
            latest_df = df.sort_values("ts", ascending=False).head(latest_n)
            for _, row in latest_df.iterrows():
                comments_norm.append(self._normalize_comment_row(row))

        return {
            "application_id": app_id,
            "phone_number": phone,
            "summary": summary,
            "latest_comments": comments_norm,  # kept for compatibility
            "comments": comments_norm,         # orchestrator reads this
        }

    def summarize_by_phone(self, phone_number: str, latest_n: int = DEFAULT_LATEST_N) -> Dict[str, Any]:
        """Summarize comments by phone number."""
        app_ids = self.resolve_app_ids_by_phone(phone_number)
        phone = _only_digits(phone_number)

        if not app_ids:
            return {
                "application_id": None,
                "phone_number": phone,
                "summary": "No application found for this phone number.",
                "latest_comments": [],
                "comments": [],
                "choices": []
            }

        if len(app_ids) > 1:
            # Let caller choose; we still return choices
            return {
                "application_id": None,
                "phone_number": phone,
                "summary": "Multiple applications found. Please choose an application_id.",
                "latest_comments": [],
                "comments": [],
                "choices": app_ids
            }

        return self.summarize_by_application_id(app_ids[0], latest_n=latest_n)

# ---------- CLI ----------
def create_cli_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Partner Dashboard Comment Summarizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --interactive                    # Interactive mode
  %(prog)s --app-id 12345                   # Summarize by application ID
  %(prog)s --phone 1234567890               # Summarize by phone number
  %(prog)s --app-id 12345 --latest 10       # Show 10 latest comments
  %(prog)s --phone 1234567890 --json-output # Output in JSON format
        """
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode (default)"
    )

    parser.add_argument(
        "--app-id", "--application-id",
        type=int,
        help="Application ID to summarize"
    )

    parser.add_argument(
        "--phone", "--phone-number",
        type=str,
        help="Phone number to lookup and summarize"
    )

    parser.add_argument(
        "--latest", "-n",
        type=int,
        default=DEFAULT_LATEST_N,
        help=f"Number of latest comments to show (default: {DEFAULT_LATEST_N})"
    )

    parser.add_argument(
        "--json-output", "-j",
        action="store_true",
        help="Output results in JSON format"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser

def format_output(result: Dict[str, Any], json_output: bool = False) -> str:
    """Format output for display."""
    if json_output:
        return json.dumps(result, indent=2)

    lines = []
    lines.append(f"Application ID: {result.get('application_id', 'N/A')}")
    lines.append(f"Phone Number: {result.get('phone_number', 'N/A')}")

    if result.get('choices'):
        lines.append(f"Multiple applications found: {result['choices']}")

    lines.append(f"\nSummary:\n{result.get('summary', 'No summary available')}")

    latest = result.get('latest_comments') or result.get('comments') or []
    if latest:
        lines.append(f"\nLatest Comments ({len(latest)}):")
        for i, comment in enumerate(latest, 1):
            lines.append(f"  {i}. [{comment.get('timestamp', 'N/A')}] {comment.get('comment', 'N/A')}")

    return '\n'.join(lines)

def interactive_mode(client: PartnerDashboardClient) -> None:
    """Run interactive mode."""
    print("\n✅ Client ready.")
    print("Enter Application ID or Phone Number (or 'quit' to exit)")

    while True:
        try:
            identifier = input("\n> ").strip()

            if identifier.lower() in ['quit', 'q', 'exit']:
                print("Exiting...")
                break

            if not identifier or not identifier.isdigit():
                print("Invalid input. Please enter digits only.")
                continue

            print(f"\nSearching for '{identifier}'...")
            result = client.summarize_by_phone(identifier)

            # If no app was found by phone, it might have been an application ID
            if result.get("summary") == "No application found for this phone number.":
                print(f"No results for phone. Trying as application ID...")
                result = client.summarize_by_application_id(int(identifier))

            print(json.dumps(result, indent=2))

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            print(f"❌ An error occurred: {e}", file=sys.stderr)

def main() -> None:
    """Main entry point."""
    parser = create_cli_parser()
    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        client = PartnerDashboardClient()

        # Handle different modes
        if args.app_id:
            result = client.summarize_by_application_id(args.app_id, latest_n=args.latest)
            print(format_output(result, args.json_output))

        elif args.phone:
            result = client.summarize_by_phone(args.phone, latest_n=args.latest)
            print(format_output(result, args.json_output))

        else:
            # Default to interactive mode
            interactive_mode(client)

    except Exception as e:
        logger.error(f"Error initializing client: {e}")
        print(f"❌ Error initializing client: {e}", file=sys.stderr)
        sys.exit(1)

# ---------- Orchestrator-facing normalized entry ----------
def fetch_partner_comments(app_id=None, phone=None, email=None, latest_n=None):
    """
    Normalized entry point consumed by the orchestrator.

    Returns a dict with at least:
      {
        "summary": "<string>",
        "comments": [ { "timestamp": "...", "comment": "...", ...}, ... ]
      }
    """
    client = PartnerDashboardClient()

    def _safe_int(x):
        try:
            return int(str(x).strip())
        except Exception:
            return None

    if app_id is not None:
        app_int = _safe_int(app_id)
        if app_int is None:
            return {"summary": f"Invalid app_id: {app_id}", "comments": []}
        res = client.summarize_by_application_id(app_int, latest_n=latest_n if latest_n else DEFAULT_LATEST_N)

    elif phone:
        res = client.summarize_by_phone(str(phone).strip(), latest_n=latest_n if latest_n else DEFAULT_LATEST_N)

    elif email and hasattr(client, "summarize_by_email"):
        # Not implemented, but kept for forward-compatibility
        res = client.summarize_by_email(str(email).strip(), latest_n=latest_n if latest_n else DEFAULT_LATEST_N)

    else:
        return {"summary": "No identifier provided", "comments": []}

    # Already normalized in client methods; just ensure keys exist
    if isinstance(res, dict):
        return {
            "summary": res.get("summary", ""),
            "comments": res.get("comments") or res.get("latest_comments") or [],
        }

    # Fallback (shouldn't happen)
    return {"summary": "", "comments": []}

if __name__ == "__main__":
    main()
