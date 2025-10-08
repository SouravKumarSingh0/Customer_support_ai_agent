from __future__ import annotations

import os
import logging
from typing import List, Optional, Sequence, Tuple
from contextlib import contextmanager
from zoneinfo import ZoneInfo

import pandas as pd
from databricks import sql as dbsql
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("transaction_database")
if __name__ == "__main__" and not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

DBX_HOST = os.getenv("DATABRICKS_SERVER_HOSTNAME") or os.getenv("DATABRICKS_HOST")
DBX_HTTP = os.getenv("DATABRICKS_HTTP_PATH")
DBX_TOKEN = os.getenv("DATABRICKS_TOKEN")
TABLE_FQN = "silver.online.transaction_fact"
DEFAULT_TZ = os.getenv("DEFAULT_TIMEZONE", "Asia/Kolkata")
DATE_COLS = ("transaction_date", "delivered_date", "confirmed_date", "due_date", "created_at", "updated_at")

try:
    import google.generativeai as genai
except Exception:
    genai = None

def _ensure_gemini(model: str = "gemini-2.5-flash"):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or genai is None:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model)
    except Exception as e:
        logger.warning("Gemini init failed: %s", e)
        return None

@contextmanager
def dbx_conn():
    if not all([DBX_HOST, DBX_HTTP, DBX_TOKEN]):
        missing = [k for k, v in {
            "DATABRICKS_SERVER_HOSTNAME": DBX_HOST,
            "DATABRICKS_HTTP_PATH": DBX_HTTP,
            "DATABRICKS_TOKEN": DBX_TOKEN,
        }.items() if not v]
        raise RuntimeError(f"Missing Databricks env vars: {', '.join(missing)}")
    conn = None
    try:
        conn = dbsql.connect(
            server_hostname=DBX_HOST,
            http_path=DBX_HTTP,
            access_token=DBX_TOKEN,
            _user_agent_entry="CSA-TransactionDatabase/1.0"
        )
        yield conn
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

def _build_sql(
    *,
    columns: Optional[Sequence[str]] = None,
    extra_where: Optional[str] = None,
    extra_params: Optional[Tuple] = None,
    order_by: str = "transaction_date DESC",
    limit: int = 1000,
) -> Tuple[str, list]:
    if columns:
        for c in columns:
            if not c.strip() or any(ch in c for ch in " ;()'\""):
                raise ValueError(f"Unsafe column name: {c!r}")
        cols = ", ".join(c.strip() for c in columns)
    else:
        cols = "*"

    sql = [f"SELECT {cols} FROM {TABLE_FQN} WHERE 1=1"]
    params: list = []

    if extra_where:
        sql.append(f"AND ({extra_where})")
        if extra_params:
            params.extend(extra_params)

    if order_by and any(ch in order_by for ch in ";()'"):
        raise ValueError("Unsafe order_by clause.")
    if order_by:
        sql.append(f"ORDER BY {order_by}")

    if limit > 0:
        sql.append("LIMIT ?")
        params.append(limit)

    return " ".join(sql), params

def _rows_to_df(cur, tz: str = DEFAULT_TZ) -> pd.DataFrame:
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=cols)
    
    df = pd.DataFrame.from_records(rows, columns=cols)
    
    tz_obj = ZoneInfo(tz)
    for col in DATE_COLS:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_convert(tz_obj)
            except Exception as e:
                logger.warning("Datetime convert failed for %s: %s", col, e)
    return df

def fetch_transactions_by_customer(customer_id: str, *, limit: int = 1000) -> pd.DataFrame:
    if not customer_id:
        raise ValueError("customer_id required")
    sql, params = _build_sql(
        extra_where="customer_id = ?",
        extra_params=(customer_id,),
        limit=limit
    )
    logger.info("DBX query: %s | params=%s", sql, tuple(params))
    with dbx_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return _rows_to_df(cur)

CUSTOMER_TO_APP_SQL = """
SELECT DISTINCT tf.customer_id, tf.application_id
FROM silver.online.transaction_fact tf
LEFT JOIN silver.online.customer_fact cf
  ON tf.customer_id = cf.customer_id
WHERE tf.customer_id = ?
ORDER BY tf.application_id
""".strip()

def resolve_application_ids_by_customer(customer_id: str) -> List[str]:
    if not customer_id:
        return []
    with dbx_conn() as conn, conn.cursor() as cur:
        cur.execute(CUSTOMER_TO_APP_SQL, [customer_id])
        rows = cur.fetchall() or []
    return sorted({r[1] for r in rows if len(r) >= 2 and r[1] is not None})

def summarize_brief_30_40_words(df: pd.DataFrame, customer_id: str) -> str:
    if df.empty:
        return "No transactions found for the provided customer ID."

    cols_pref = [
        "transaction_id", "application_id", "transaction_date",
        "transaction_status", "net_amount", "currency", "payment_method"
    ]
    preview_cols = [c for c in cols_pref if c in df.columns]
    head = df[preview_cols].head(12) if preview_cols else df.head(12)

    sample_lines = "\n".join(
        "; ".join(f"{k}={row[k]}" for k in row.index if pd.notna(row[k]))
        for _, row in head.iterrows()
    )

    model = _ensure_gemini()
    if not model:
        total = len(df)
        statuses = df["transaction_status"].value_counts().to_dict() if "transaction_status" in df.columns else {}
        amt = None
        if "net_amount" in df.columns:
            try:
                amt = float(df["net_amount"].sum())
            except Exception:
                amt = None
        parts = [f"Customer {customer_id} has {total} transaction(s)"]
        if statuses:
            parts.append(f"; status mix: {statuses}")
        if amt is not None:
            parts.append(f"; total amount ~{amt:,.2f}")
        return "".join(parts)[:220]

    prompt = f"""
You are a precise financial ops summarizer. Given a compact sample of a single customer's transactions,
write a strictly factual summary in 30–40 words. Do not invent fields. Prefer counts, recency, amounts, and status mix.

Identifier:
- customer_id: {customer_id}

Sample rows (semicolon key=value pairs, one per line):
{sample_lines}

Now produce the 30–40 word summary.
""".strip()

    try:
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        return text[:320]
    except Exception as e:
        logger.warning("Gemini summary failed: %s", e)
        return "Summary unavailable due to model error."

def print_banner():
    print("\n" + "="*60)
    print("  TRANSACTION DATABASE QUERY TOOL")
    print("="*60 + "\n")

def print_results(df: pd.DataFrame, summary: str):
    print("\n" + "-"*60)
    print("SUMMARY")
    print("-"*60)
    print(summary)
    print("\n" + "-"*60)
    print(f"TRANSACTION COUNT: {len(df)}")
    print("-"*60)
    
    if not df.empty:
        print("\nFirst 5 transactions:")
        display_cols = [c for c in ["transaction_id", "transaction_date", "transaction_status", "net_amount"] if c in df.columns]
        if display_cols:
            print(df[display_cols].head().to_string(index=False))
        else:
            print(df.head().to_string(index=False))
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    print_banner()
    
    while True:
        try:
            customer_id = input("Enter customer_id (or 'quit' to exit): ").strip()
            
            if customer_id.lower() in ('quit', 'exit', 'q'):
                print("\nExiting. Goodbye!")
                break
                
            if not customer_id:
                print("❌ Error: customer_id cannot be empty. Please try again.\n")
                continue
            
            print(f"\n🔍 Fetching transactions for customer: {customer_id}...")
            
            try:
                df = fetch_transactions_by_customer(customer_id, limit=1000)
                summary = summarize_brief_30_40_words(df, customer_id)
                print_results(df, summary)
                
            except ValueError as e:
                print(f"❌ Validation Error: {e}\n")
            except RuntimeError as e:
                print(f"❌ Configuration Error: {e}\n")
                break
            except Exception as e:
                logger.error("Query failed: %s", e, exc_info=True)
                print(f"❌ Error: Failed to fetch transactions: {e}\n")
                
        except (EOFError, KeyboardInterrupt):
            print("\n\nInterrupted. Exiting gracefully...")
            break
        except Exception as e:
            logger.error("Unexpected error: %s", e, exc_info=True)
            print(f"❌ Unexpected error: {e}\n")