from __future__ import annotations
import os, sys, json, argparse, logging, csv, re, asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache, wraps
from contextlib import asynccontextmanager, contextmanager
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

try:
    from databricks import sql as dbx_sql
except Exception:
    dbx_sql = None

try:
    import slack_agent as Slack
except ModuleNotFoundError:
    try:
        import Slack as Slack
    except ModuleNotFoundError:
        Slack = None

try:
    import TeleCalling_Databricks as TeleCalling_Databricks
except ModuleNotFoundError:
    import TeleCalling_Databricks

import PartnerDB_Databrics
import OnboardingTime

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("orchestrator")

USE_GEMINI = bool(os.getenv("GEMINI_API_KEY"))
if USE_GEMINI:
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        logger.info("Gemini integration enabled")
    except Exception as e:
        USE_GEMINI = False
        logger.warning(f"Failed to initialize Gemini: {e}")

THREAD_POOL_MAX_WORKERS = int(os.getenv("THREAD_POOL_MAX_WORKERS", "8"))
OPERATION_TIMEOUT = int(os.getenv("OPERATION_TIMEOUT", "60"))
ORCH_TIMEOUT_GLOBAL_SEC = int(os.getenv("ORCH_TIMEOUT_GLOBAL_SEC", str(max(OPERATION_TIMEOUT, 180))))
ORCH_TIMEOUT_PER_TASK_SEC = int(os.getenv("ORCH_TIMEOUT_PER_TASK_SEC", "45"))

DBX_HOST = os.getenv("DATABRICKS_SERVER_HOSTNAME") or os.getenv("DATABRICKS_HOST")
DBX_HTTP = os.getenv("DATABRICKS_HTTP_PATH")
DBX_TOKEN = os.getenv("DATABRICKS_TOKEN")
FRESHDESK_TABLE = os.getenv("FRESHDESK_TABLE", "bronze.datamart.freshdesk_tickets")
FD_DEFAULT_PAGE = int(os.getenv("FRESHDESK_PAGE_DEFAULT", "1"))
FD_DEFAULT_PAGE_SIZE = int(os.getenv("FRESHDESK_PAGE_SIZE_DEFAULT", "50"))
FD_MAX_PAGE_SIZE = int(os.getenv("FRESHDESK_PAGE_SIZE_MAX", "200"))
FD_HARD_CAP_ROWS = int(os.getenv("FRESHDESK_HARD_CAP_ROWS", "500"))

_dbx_connection_pool = []
_dbx_pool_lock = asyncio.Lock()

def timeout_decorator(seconds):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.error(f"{func.__name__} timed out after {seconds}s")
                raise
        return wrapper
    return decorator

@contextmanager
def timer(operation_name: str):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"'{operation_name}' completed in {duration:.2f}s")

def iso_now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def _safe(obj: Any, *keys, default=None):
    cur = obj
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

_dt_parse_cache = {}
def _parse_dt(val: Any) -> Optional[datetime]:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    if not isinstance(val, str):
        try:
            if isinstance(val, (int, float)):
                return datetime.fromtimestamp(float(val), tz=timezone.utc)
        except (ValueError, OSError):
            pass
        return None
    
    s = val.strip()
    if not s:
        return None
    
    if s in _dt_parse_cache:
        return _dt_parse_cache[s]
    
    s = s.replace("Z", "+00:00") if s.endswith("Z") else s
    try:
        dt = datetime.fromisoformat(s)
        result = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        _dt_parse_cache[s] = result
        return result
    except ValueError:
        for f in ("%Y-%m-%d %H:%M:%S%z","%Y-%m-%d %H:%M:%S","%Y-%m-%dT%H:%M:%S%z",
                  "%Y-%m-%dT%H:%M:%S","%d-%m-%Y %H:%M:%S%z","%d-%m-%Y %H:%M:%S"):
            try:
                dt = datetime.strptime(s, f)
                result = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                _dt_parse_cache[s] = result
                return result
            except ValueError:
                continue
    return None

def _fmt_dt(dt: Optional[datetime]) -> str:
    if not dt:
        return ""
    try:
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except (ValueError, OSError):
        return ""

def _first_non_empty(*vals) -> str:
    for v in vals:
        if v:
            s = str(v).strip()
            if s:
                return s
    return ""

def _limit(s: str, n: int) -> str:
    if not s:
        return ""
    s = s.strip()
    return s if len(s) <= n else s[: n - 1] + "…"

def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    if hasattr(obj, 'isoformat'):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    return obj

def save_timeline_as_csv(events: List[Dict], output_path: str):
    if not events:
        return
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['timestamp', 'source', 'title', 'details'])
            writer.writeheader()
            for event in events:
                writer.writerow({
                    'timestamp': event.get('ts_str', ''),
                    'source': event.get('source', ''),
                    'title': event.get('title', ''),
                    'details': event.get('detail', '')
                })
        logger.info(f"Saved timeline with {len(events)} events to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write CSV: {e}")

def extract_tldr_and_key_events(summary_text: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    tldr_bullets: List[str] = []
    key_events: List[Tuple[str, str, str]] = []

    def _section(regexes: List[str]) -> Optional[str]:
        for rx in regexes:
            m = re.search(rx, summary_text, flags=re.S | re.I)
            if m:
                return m.group(1).strip()
        return None

    tldr_body = _section([
        r"##\s*1\.\s*Executive Summary\s*\(TL;DR\)\s*(.+?)\n##\s*[\d#]",
        r"##\s*TL;DR\s*(.+?)\n##\s*[\d#]"
    ]) or _section([
        r"##\s*1\.\s*Executive Summary\s*\(TL;DR\)\s*(.+)$",
        r"##\s*TL;DR\s*(.+)$"
    ])

    if tldr_body:
        for line in tldr_body.splitlines():
            line = line.strip()
            if line.startswith(("-", "•")):
                tldr_bullets.append(line.lstrip("-• ").strip())

    key_body = _section([
        r"##\s*3\.\s*Key Events\s*&\s*Timeline\s*(.+?)\n##\s*[\d#]",
        r"##\s*Key Events\s*&\s*Timeline\s*(.+?)\n##\s*[\d#]"
    ]) or _section([
        r"##\s*3\.\s*Key Events\s*&\s*Timeline\s*(.+)$",
        r"##\s*Key Events\s*&\s*Timeline\s*(.+)$"
    ])

    def _unbold(s: str) -> str:
        return re.sub(r"\*\*([^*]+)\*\*", r"\1", s).strip()

    if key_body:
        for raw in key_body.splitlines():
            line = raw.strip().lstrip("- ").strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("·")]
            if len(parts) >= 3 and re.match(r"^\d{4}-\d{2}-\d{2}", parts[0]):
                when = parts[0]
                src = _unbold(parts[1]).rstrip(":")
                desc = "·".join(parts[2:]).strip()
                key_events.append((when, src, desc))
            else:
                m = re.match(r"^(\d{4}-\d{2}-\d{2}[^\s]*)\s*[·\-]\s*\*{0,2}([^*]+?)\*{0,2}[:\-–]\s*(.+)$", line)
                if m:
                    key_events.append((m.group(1).strip(), m.group(2).strip(), m.group(3).strip()))
    return tldr_bullets, key_events

def _events_to_key_events(events: List[Dict], limit: int = 60) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    for e in events[:limit]:
        when = e.get("ts_str") or ""
        src = e.get("source") or ""
        title = e.get("title") or ""
        detail = e.get("detail") or ""
        desc = (title + (" — " if title and detail else "") + detail).strip()
        rows.append((when, src, desc))
    return rows

def write_summary_csv(tldr_bullets: List[str], path: str) -> None:
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Summary"])
            for b in tldr_bullets:
                w.writerow([b])
    except Exception as e:
        logger.error(f"Failed to write summary CSV: {e}")

def write_key_events_csv(key_events: List[Tuple[str, str, str]], path: str) -> None:
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["When", "Source", "Event"])
            for when, src, desc in key_events:
                w.writerow([when, src, desc])
    except Exception as e:
        logger.error(f"Failed to write key events CSV: {e}")

@lru_cache(maxsize=500)
def _resolve_phone_by_app_cached(app_id: str) -> Optional[str]:
    try:
        client = PartnerDB_Databrics.PartnerDashboardClient()
        try:
            p = client.resolve_phone_by_app(int(app_id))
        except ValueError:
            p = client.resolve_phone_by_app(str(app_id))
        return str(p) if p else None
    except Exception as e:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Failed to resolve phone for app_id {app_id}: {e}")
        return None

@lru_cache(maxsize=500)
def _resolve_app_by_phone_cached(phone: str) -> Optional[str]:
    try:
        client = PartnerDB_Databrics.PartnerDashboardClient()
        ids = client.resolve_app_ids_by_phone(str(phone)) or []
        if ids:
            return str(ids[0])
    except Exception as e:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Failed to resolve app_id for phone {phone}: {e}")

    try:
        aid = OnboardingTime.resolve_application_id(str(phone))
        return str(aid) if aid else None
    except Exception as e:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Failed OnboardingTime resolve for phone {phone}: {e}")
        return None

def _resolve_identifiers_initial(app_id: Optional[str], phone: Optional[str], email: Optional[str]
                                 ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    new_app, new_phone, new_email = app_id, phone, email
    with timer("identifier_resolution"):
        if new_app and not new_phone:
            new_phone = _resolve_phone_by_app_cached(new_app)
        if new_phone and not new_app:
            new_app = _resolve_app_by_phone_cached(new_phone)
    return new_app, new_phone, new_email

def _extract_identifiers_from_slices(slices: Dict[str, Dict]) -> Dict[str, str]:
    found = {"app_id": None, "phone": None, "email": None}
    def try_set(key, val):
        if val and not found[key]:
            s = str(val).strip()
            if s:
                found[key] = s

    slice_configs = [
        ("Slack", "events", [
            ("app_id", ["application_id", "app_id", "applicationId", "appId"]),
            ("phone", ["phone", "mobile", "msisdn"]),
            ("email", ["email", "customer_email"])
        ]),
        ("Telecalling", "calls", [
            ("app_id", ["application_id", "app_id"]),
            ("phone", ["phone", "mobile", "customer_phone"]),
            ("email", ["email"])
        ]),
        ("PartnerDB", "comments", [
            ("app_id", ["application_id", "app_id"]),
            ("phone", ["phone", "mobile"]),
            ("email", ["email"])
        ]),
        ("Onboarding", "timeline", [
            ("app_id", ["application_id", "app_id"]),
            ("phone", ["phone", "mobile"]),
            ("email", ["email"])
        ]),
        ("Freshdesk", "tickets", [
            ("phone", ["phone","phone_number","mobile","contact_number","customer_phone"]),
        ]),
    ]
    for slice_name, items_key, field_mappings in slice_configs:
        items = slices.get(slice_name, {}).get(items_key, []) or []
        if not items:
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            for target_key, source_keys in field_mappings:
                if found[target_key]:
                    continue
                for source_key in source_keys:
                    val = item.get(source_key)
                    if val:
                        try_set(target_key, val)
                        break
    return {k: v for k, v in found.items() if v}

async def _get_dbx_connection():
    async with _dbx_pool_lock:
        if _dbx_connection_pool:
            return _dbx_connection_pool.pop()
        
        if dbx_sql is None:
            raise RuntimeError("databricks-sql-connector not installed")
        if not (DBX_HOST and DBX_HTTP and DBX_TOKEN):
            raise RuntimeError("Databricks env missing")
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: dbx_sql.connect(server_hostname=DBX_HOST, http_path=DBX_HTTP, access_token=DBX_TOKEN)
        )

async def _release_dbx_connection(conn):
    async with _dbx_pool_lock:
        if len(_dbx_connection_pool) < 3:
            _dbx_connection_pool.append(conn)
        else:
            try:
                conn.close()
            except:
                pass

def _execute_to_dicts(cur, query: str) -> List[Dict[str, Any]]:
    cur.execute(query)
    try:
        df = cur.fetchall_arrow().to_pandas()
        return [{k: (v if v == v else None) for k,v in row.items()} for _, row in df.iterrows()]
    except Exception:
        rows = cur.fetchall()
        cols = [c[0] for c in cur.description] if cur.description else []
        return [{c: (r[i] if i < len(r) else None) for i, c in enumerate(cols)} for r in rows]

def _digits_only(s: Optional[str]) -> str:
    return "".join(ch for ch in (s or "") if ch.isdigit())

_fd_column_cache = {}
async def _freshdesk_discover_columns(cur) -> Dict[str, str]:
    if FRESHDESK_TABLE in _fd_column_cache:
        return _fd_column_cache[FRESHDESK_TABLE]
    
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, cur.execute, f"DESCRIBE {FRESHDESK_TABLE}")
    rows = await loop.run_in_executor(None, cur.fetchall)
    
    cols = [str(r[0]).strip() for r in rows if r and str(r[0]).strip() and not str(r[0]).startswith("#")]
    low = {c.lower(): c for c in cols}

    phone_candidates = ["phone_number","phone","mobile","contact_number","customer_phone"]
    created_candidates = ["created_at","createdon","created_ts","createdtime"]
    updated_candidates = ["updated_at","updatedon","updated_ts","lastupdated","last_activity_at"]

    def pick(cands):
        for c in cands:
            if c in low:
                return low[c]
        return None

    result = {
        "phone_col": pick([c.lower() for c in phone_candidates]),
        "created_col": pick([c.lower() for c in created_candidates]),
        "updated_col": pick([c.lower() for c in updated_candidates]),
    }
    _fd_column_cache[FRESHDESK_TABLE] = result
    return result

@timeout_decorator(ORCH_TIMEOUT_PER_TASK_SEC)
async def _fetch_slack_data(identifiers: Dict[str, str], tier1_channels: List[str]) -> Dict[str, Any]:
    try:
        if Slack is None:
            return {"events": [], "meta": {"error": "Slack module not available"}}
        
        loop = asyncio.get_event_loop()
        with timer("slack_fetch"):
            result = await loop.run_in_executor(
                None,
                Slack.search,
                identifiers,
                30,
                tier1_channels
            )
        return result
    except asyncio.TimeoutError:
        logger.error("Slack fetch timed out")
        return {"events": [], "meta": {"error": "TimeoutError"}}
    except Exception as e:
        logger.error(f"Slack fetch error: {type(e).__name__}: {e}")
        return {"events": [], "meta": {"error": f"{type(e).__name__}: {e}"}}

@timeout_decorator(ORCH_TIMEOUT_PER_TASK_SEC)
async def _fetch_telecalling_data(app_id: Optional[str], phone: Optional[str], email: Optional[str], latest_n: Optional[int]) -> Dict[str, Any]:
    try:
        loop = asyncio.get_event_loop()
        with timer("telecalling_fetch"):
            result = await loop.run_in_executor(
                None,
                TeleCalling_Databricks.summarize_calls,
                None if phone else app_id,
                phone,
                email,
                latest_n
            )
        return result
    except asyncio.TimeoutError:
        logger.error("Telecalling fetch timed out")
        return {"summary": "[Telecalling error] TimeoutError", "calls": []}
    except Exception as e:
        logger.error(f"Telecalling fetch error: {type(e).__name__}: {e}")
        return {"summary": f"[Telecalling error] {type(e).__name__}: {e}", "calls": []}

@timeout_decorator(ORCH_TIMEOUT_PER_TASK_SEC)
async def _fetch_partner_data(app_id: Optional[str], phone: Optional[str], email: Optional[str], latest_n: Optional[int]) -> Dict[str, Any]:
    try:
        loop = asyncio.get_event_loop()
        with timer("partner_fetch"):
            result = await loop.run_in_executor(
                None,
                PartnerDB_Databrics.fetch_partner_comments,
                app_id,
                phone,
                email,
                latest_n
            )
        return result
    except asyncio.TimeoutError:
        logger.error("PartnerDB fetch timed out")
        return {"summary": "[PartnerDB error] TimeoutError", "comments": []}
    except Exception as e:
        logger.error(f"PartnerDB fetch error: {type(e).__name__}: {e}")
        return {"summary": f"[PartnerDB error] {type(e).__name__}: {e}", "comments": []}

@timeout_decorator(ORCH_TIMEOUT_PER_TASK_SEC)
async def _fetch_onboarding_data(app_id: Optional[str], phone: Optional[str], email: Optional[str]) -> Dict[str, Any]:
    try:
        loop = asyncio.get_event_loop()
        with timer("onboarding_fetch"):
            result = await loop.run_in_executor(
                None,
                OnboardingTime.get_onboarding_metrics,
                app_id,
                phone,
                email
            )
        return result
    except asyncio.TimeoutError:
        logger.error("Onboarding fetch timed out")
        return {"summary": "[Onboarding error] TimeoutError", "timeline": [], "bottlenecks": []}
    except Exception as e:
        logger.error(f"Onboarding fetch error: {type(e).__name__}: {e}")
        return {"summary": f"[Onboarding error] {type(e).__name__}: {e}", "timeline": [], "bottlenecks": []}

@timeout_decorator(ORCH_TIMEOUT_PER_TASK_SEC)
async def _fetch_freshdesk_data(app_id: Optional[str], phone: Optional[str], page: int, page_size: int, latest_n: Optional[int]) -> Dict[str, Any]:
    try:
        with timer("freshdesk_fetch"):
            if phone is None and app_id:
                phone = _resolve_phone_by_app_cached(str(app_id))
            if not phone:
                return {"summary": "Freshdesk: phone not available", "tickets": [], "meta": {"warning": "no-phone"}}

            if dbx_sql is None:
                return {"summary": "Freshdesk: Databricks connector not installed", "tickets": [], "meta": {"error": "dbx-missing"}}
            if not (DBX_HOST and DBX_HTTP and DBX_TOKEN):
                return {"summary": "Freshdesk: Databricks env not configured", "tickets": [], "meta": {"error": "dbx-env"}}

            digits = _digits_only(phone)
            page = max(1, int(page or 1))
            page_size = max(1, min(int(page_size or FD_DEFAULT_PAGE_SIZE), FD_MAX_PAGE_SIZE))
            if latest_n is not None:
                page_size = min(page_size, int(latest_n))
            offset = (page - 1) * page_size
            total_limit = min(page_size, FD_HARD_CAP_ROWS)

            conn = await _get_dbx_connection()
            try:
                loop = asyncio.get_event_loop()
                cur = await loop.run_in_executor(None, conn.cursor)
                
                cols = await _freshdesk_discover_columns(cur)
                phone_col = cols.get("phone_col")
                created_col = cols.get("created_col") or phone_col
                updated_col = cols.get("updated_col") or created_col

                if not phone_col:
                    return {"summary": "Freshdesk: phone column not found", "tickets": [], "meta": {"error": "schema"}}

                normalized_phone_expr = f"REGEXP_REPLACE(COALESCE({phone_col}, ''), '[^0-9]', '')"
                q = f"""
                    SELECT *
                    FROM {FRESHDESK_TABLE}
                    WHERE {normalized_phone_expr} = '{digits}'
                    ORDER BY COALESCE({updated_col}, {created_col}) DESC
                    LIMIT {total_limit} OFFSET {offset}
                """
                
                rows = await loop.run_in_executor(None, _execute_to_dicts, cur, q)

                return {
                    "summary": f"Freshdesk: {len(rows)} ticket(s) fetched (page={page}, size={page_size})",
                    "tickets": rows,
                    "meta": {"page": page, "page_size": page_size}
                }
            finally:
                await _release_dbx_connection(conn)
    except asyncio.TimeoutError:
        logger.error("Freshdesk fetch timed out")
        return {"summary": "[Freshdesk error] TimeoutError", "tickets": [], "meta": {}}
    except Exception as e:
        logger.error(f"Freshdesk fetch error: {type(e).__name__}: {e}")
        return {"summary": f"[Freshdesk error] {type(e).__name__}: {e}", "tickets": [], "meta": {"error": str(e)}}

async def collect_slices(app_id=None, phone=None, email=None, latest_n=None, fd_page=None, fd_page_size=None) -> Dict[str, Dict]:
    identifiers = {}
    if app_id: identifiers["app_id"] = str(app_id).strip()
    if phone:  identifiers["phone_digits"] = _digits_only(str(phone))
    if email:  identifiers["email"] = str(email).strip().lower()

    tier1_channels = [c.strip() for c in (os.getenv("SLACK_CHANNELS") or "").split(",") if c.strip()]

    with timer("concurrent_data_collection"):
        tasks = [
            _fetch_slack_data(identifiers, tier1_channels),
            _fetch_telecalling_data(app_id, phone, email, latest_n),
            _fetch_partner_data(app_id, phone, email, latest_n),
            _fetch_onboarding_data(app_id, phone, email),
            _fetch_freshdesk_data(app_id, phone, fd_page or FD_DEFAULT_PAGE, fd_page_size or FD_DEFAULT_PAGE_SIZE, latest_n),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        out = {}
        source_names = ["Slack", "Telecalling", "PartnerDB", "Onboarding", "Freshdesk"]
        
        for i, (source, result) in enumerate(zip(source_names, results)):
            if isinstance(result, Exception):
                logger.error(f"{source} failed: {result}")
                if source == "Slack":
                    out[source] = {"summary": "", "events": [], "meta": {"error": str(result)}}
                elif source == "Freshdesk":
                    out[source] = {"summary": f"[{source} error] {result}", "tickets": [], "meta": {}}
                elif source == "Onboarding":
                    out[source] = {"summary": f"[{source} error] {result}", "timeline": [], "bottlenecks": []}
                else:
                    out[source] = {"summary": f"[{source} error] {result}", "calls" if source == "Telecalling" else "comments": []}
            else:
                if source == "Slack":
                    out[source] = {
                        "summary": "",
                        "events": result.get("events", []) or [],
                        "meta": result.get("meta", {})
                    }
                elif source == "Telecalling":
                    out[source] = {
                        "summary": _safe(result, "summary", default=""),
                        "calls": _safe(result, "calls", default=[]) or []
                    }
                elif source == "PartnerDB":
                    out[source] = {
                        "summary": _safe(result, "summary", default=""),
                        "comments": _safe(result, "comments", default=_safe(result, "latest_comments", default=[])) or []
                    }
                elif source == "Onboarding":
                    out[source] = {
                        "summary": _safe(result, "summary", default=""),
                        "timeline": _safe(result, "timeline", default=_safe(result, "stages", default=[])) or [],
                        "bottlenecks": _safe(result, "bottlenecks", default=_safe(result, "delays", default=[])) or []
                    }
                elif source == "Freshdesk":
                    out[source] = {
                        "summary": _safe(result, "summary", default=""),
                        "tickets": _safe(result, "tickets", default=[]) or [],
                        "meta": _safe(result, "meta", default={})
                    }
    
    return out

def build_timeline(slices: Dict[str, Dict]) -> List[Dict]:
    events: List[Dict] = []

    with timer("timeline_construction"):
        for ev in slices.get("Slack", {}).get("events", []):
            ts = _parse_dt(_first_non_empty(ev.get("ts"), ev.get("timestamp"), ev.get("time"), ev.get("date")))
            title = _limit(ev.get("title") or "Slack message", 100)
            detail = _limit(_first_non_empty(ev.get("text"), ev.get("message")), 600)
            if ts or detail:
                events.append({"ts": ts, "ts_str": _fmt_dt(ts), "source": "Slack", "title": title, "detail": detail, "raw": ev})

        for call in slices.get("Telecalling", {}).get("calls", []):
            ts = _parse_dt(_first_non_empty(call.get("time"), call.get("timestamp"), call.get("created_at"), call.get("call_date")))
            who = _first_non_empty(call.get("agent"), call.get("agent_name"))
            disp = _first_non_empty(call.get("disposition"), call.get("status"))
            title = _limit(f"Telecall: {disp or 'call'}", 100)
            detail = _limit(_first_non_empty(call.get("remark"), call.get("notes"), call.get("comment")), 600)
            if who:
                detail = (f"Agent: {who}. " + detail).strip()
            if ts or detail or title:
                events.append({"ts": ts, "ts_str": _fmt_dt(ts), "source": "Telecalling", "title": title, "detail": detail, "raw": call})

        for c in slices.get("PartnerDB", {}).get("comments", []):
            ts = _parse_dt(_first_non_empty(c.get("created_at"), c.get("time"), c.get("timestamp"), c.get("ts")))
            who = _first_non_empty(c.get("user"), c.get("user_name"), c.get("author"), c.get("created_by"))
            title = _limit("Partner comment", 100)
            comment = _limit(_first_non_empty(c.get("application_comment"), c.get("comment"), c.get("note"), c.get("text"), c.get("remarks")), 600)
            if who:
                comment = (f"{who}: " + comment).strip()
            if ts or comment:
                events.append({"ts": ts, "ts_str": _fmt_dt(ts), "source": "PartnerDB", "title": title, "detail": comment, "raw": c})

        for st in slices.get("Onboarding", {}).get("timeline", []):
            ts = _parse_dt(_first_non_empty(st.get("at"), st.get("time"), st.get("timestamp"), st.get("created_at"), st.get("updated_time")))
            stage = _first_non_empty(st.get("stage"), st.get("status"), st.get("name"))
            meta = []
            for key in ("substatus", "info", "note"):
                val = st.get(key)
                if val:
                    meta.append(str(val))
            detail = _limit("; ".join(meta), 500)
            title = _limit(f"Stage: {stage or 'event'}", 100)
            events.append({"ts": ts, "ts_str": _fmt_dt(ts), "source": "Onboarding", "title": title, "detail": detail, "raw": st})

        for t in slices.get("Freshdesk", {}).get("tickets", []):
            ts = _parse_dt(_first_non_empty(t.get("updated_at"), t.get("created_at"), t.get("createdon"), t.get("updatedon")))
            subj = _first_non_empty(t.get("subject"), t.get("status"), "Freshdesk ticket")
            title = _limit(f"Freshdesk: {subj}", 120)
            desc = _limit(_first_non_empty(t.get("description"), t.get("description_text"), t.get("description_html")), 600)
            events.append({"ts": ts, "ts_str": _fmt_dt(ts), "source": "Freshdesk", "title": title, "detail": desc, "raw": t})

        max_datetime = datetime.max.replace(tzinfo=timezone.utc)
        events.sort(key=lambda e: (e["ts"] is None, e["ts"] or max_datetime))

    return events

SYSTEM_INSTRUCTIONS = """
You are a Customer Experience Analyst. Your job is to take all available customer data (events, conversations, remarks, dispositions, and system logs) and create a clear, human-readable journey report. The report should tell the story of what happened to the customer, where they got stuck, and what needs to happen next.

---

### Report Structure

#### 1. Executive Summary
Summarize the customer’s situation in 3 short bullets. Include their name, application ID, phone number, and email if available.
* **Main Issue/Goal:** What the customer is trying to achieve, and what is blocking progress.
* **Current Status:** Where the customer stands right now in the journey (e.g., “Waiting for verification, stuck for 12 days”).
* **Critical Next Action:** The most important step needed to move forward and who should own it (e.g., “Operations must recheck Aadhaar proof”).

#### 2. Customer Journey Narrative
Write a 150–250 word story that connects all the dots in order:
* Begin with their first interaction or signup.
* Explain what they did at each step and why delays happened.
* Mention the people, teams, or partners involved.
* Write in flowing, story-like prose instead of bullet points.

#### 3. Detailed Timeline
List the 10–15 most important moments, starting with the most recent. Use this format:
`YYYY-MM-DDTHH:MM:SSZ ·` **`[Source]`** `· [Event Description]`

#### 4. Onboarding Journey Analysis
Break down the onboarding into stages:
* **Stage Breakdown:** List each stage (e.g., Registration, KYC, Activation), show how much time was spent, and whether it is completed, in progress, or blocked.  
* **Longest Delay:** Identify the single biggest delay, with start and end events, duration, and likely cause.  
* **Bottlenecks:** Explain where the customer spent the most time compared to normal, and whether the delay was on their side or due to internal processes.

#### 5. Internal Conversations & Context
* **Remarks Summary:** Gather all notes, dispositions, and comments in order, highlighting sentiment, urgency, or special handling.  
* **Agent/Team Involvement:** List the agents or teams involved, describing their actions and any escalations.

#### 6. Root Cause & Recommendations
* **Root Cause Analysis:** Identify the main reason for the issue (e.g., document quality, system error, customer unresponsive). Provide evidence and impact in terms of time lost.  
* **Actionable Next Steps:** Give 3–5 specific steps, each with a priority (CRITICAL / HIGH / MEDIUM) and a clear owner.  
* **Preventive Measures:** Suggest 1–2 improvements that would prevent this kind of issue from happening again.

#### 7. Data Quality Check
* **Data Gaps:** Point out missing information that makes analysis harder (e.g., missing timestamps).  
* **Data Conflicts:** Call out contradictions between sources (e.g., different timestamps or statuses).

---

### Guidelines
* Base your conclusions only on available evidence—if data is missing, say “unclear.”  
* Be precise with time calculations (all in UTC format: `YYYY-MM-DDTHH:MM:SSZ`).  
* Use a professional, empathetic, and clear tone. Avoid jargon.  
* Always focus on the customer’s perspective.  
* Make recommendations actionable and owned by specific teams or roles.  

"""



async def summarize_gemini(payload: Dict, events: List[Dict]) -> str:
    with timer("gemini_summarization"):
        model = genai.GenerativeModel(GEMINI_MODEL)
        sanitized_payload = sanitize_for_json(payload)

        MAX_TIMELINE_EVENTS_FOR_PROMPT = 40
        events_for_prompt = events[-MAX_TIMELINE_EVENTS_FOR_PROMPT:]

        content = {
            "generated_at_utc": iso_now_utc(),
            "input_identifiers": sanitized_payload["identifiers"],
            "timeline": [{"title": e["title"], "timestamp": e["ts_str"], "source": e["source"], "detail": e["detail"]} for e in events_for_prompt],
        }
        
        prompt = [
            {"role": "user", "parts": [{"text": SYSTEM_INSTRUCTIONS}]},
            {"role": "user", "parts": [{"text": json.dumps(content, ensure_ascii=False)}]},
        ]
        
        generation_config = {
            "max_output_tokens": 4096,
            "temperature": 0.3,
        }

        try:
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: model.generate_content(prompt, generation_config=generation_config)
            )
            return getattr(resp, "text", str(resp))
        except Exception as e:
            logger.error(f"Gemini API error: {type(e).__name__}: {e}")
            raise

def summarize_rule_based(payload: Dict, events: List[Dict]) -> str:
    with timer("rule_based_summarization"):
        s = payload["slices"]
        
        tl_lines = []
        for e in events:
            ts = e["ts_str"] or "(no-ts)"
            tl_lines.append(f"- {e['title']} · {ts} · source={e['source']} · {_limit(e['detail'], 300)}")
        tl = "\n".join(tl_lines) if tl_lines else "No dated events available."

        bottlenecks = []
        for b in s.get("Onboarding", {}).get("bottlenecks", []):
            stage = _first_non_empty(b.get("stage"), b.get("name"))
            dur = _first_non_empty(b.get("duration_minutes"), b.get("duration"))
            if stage or dur:
                bottlenecks.append(f"- {stage or 'stage'} · duration={dur}")

        recos = []
        tl_lower = tl.lower()
        if "kyc" in tl_lower:
            recos.append("- Improve KYC document upload reliability (retry logic, clearer error messages).")
        if any("call back" in (e["detail"].lower()) for e in events if e["source"] == "Telecalling"):
            recos.append("- Add automatic reminders for promised call-backs; track an SLA.")
        if any("login" in (e["detail"].lower()) for e in events):
            recos.append("- Investigate auth/session stability for partner & customer portals.")

        return f"""# Customer Journey Summary

## TL;DR
- Unified view built from Slack, Telecalling, PartnerDB, Onboarding, and Freshdesk.
- {_limit(s.get("Slack",{}).get("summary","") or "Slack: timeline entries included.", 160)}
- {_limit(s.get("Telecalling",{}).get("summary","") or "Telecalling: see call notes.", 160)}
- {_limit(s.get("PartnerDB",{}).get("summary","") or "PartnerDB: see comments.", 160)}
- {_limit(s.get("Freshdesk",{}).get("summary","") or "Freshdesk: latest ticket activity included.", 160)}

## Chronological Timeline
{tl}

## Channel Highlights
- **Slack**: {_limit(s.get("Slack",{}).get("summary",""), 500) or "—"}
- **Telecalling**: {_limit(s.get("Telecalling",{}).get("summary",""), 500) or "—"}
- **Partner Portal**: {_limit(s.get("PartnerDB",{}).get("summary",""), 500) or "—"}
- **Onboarding**: {_limit(s.get("Onboarding",{}).get("summary",""), 500) or "—"}
- **Freshdesk**: {_limit(s.get("Freshdesk",{}).get("summary",""), 500) or "—"}

## Bottlenecks & Root Causes
{os.linesep.join(bottlenecks) or "—"}

## Product/Process Recommendations
{os.linesep.join(recos) or "—"}

## Data Completeness Notes
- Any missing timestamps appear as `(no-ts)`.
- Field names vary by source and were normalized best-effort.
"""

async def build_payload(app_id=None, phone=None, email=None, latest_n=None, fd_page=None, fd_page_size=None) -> Tuple[Dict, List[Dict]]:
    with timer("payload_build"):
        slices = await collect_slices(app_id=app_id, phone=phone, email=email, latest_n=latest_n, fd_page=fd_page, fd_page_size=fd_page_size)
        events = build_timeline(slices)
        payload = {
            "generated_at_utc": iso_now_utc(),
            "identifiers": {"app_id": app_id, "phone": phone, "email": email},
            "slices": slices,
            "counts": {
                "events_total": len(events),
                "slack_events": len(slices.get("Slack", {}).get("events", [])),
                "tele_calls": len(slices.get("Telecalling", {}).get("calls", [])),
                "partner_comments": len(slices.get("PartnerDB", {}).get("comments", [])),
                "onboarding_stages": len(slices.get("Onboarding", {}).get("timeline", [])),
                "freshdesk_tickets": len(slices.get("Freshdesk", {}).get("tickets", [])),
            },
        }
    return payload, events

async def run_async(app_id=None, phone=None, email=None, out_prefix="journey", latest_n=None, csv_output_path=None, fd_page=None, fd_page_size=None) -> str:
    app_id, phone, email = _resolve_identifiers_initial(app_id, phone, email)

    if not any([app_id, phone]):
        logger.warning("Could not resolve a primary identifier. Aborting.")
        return "Error: Provide at least an Application ID or Phone Number."

    payload, events = await build_payload(app_id=app_id, phone=phone, email=email, latest_n=latest_n, fd_page=fd_page, fd_page_size=fd_page_size)

    discovered = _extract_identifiers_from_slices(payload["slices"])
    new_app_id = app_id or discovered.get("app_id")
    new_phone  = phone  or discovered.get("phone")
    new_email  = email  or discovered.get("email")
    if (new_app_id, new_phone, new_email) != (app_id, phone, email):
        logger.info("Discovered new identifiers, re-running data collection.")
        payload, events = await build_payload(
            app_id=new_app_id, phone=new_phone, email=new_email, latest_n=latest_n,
            fd_page=fd_page, fd_page_size=fd_page_size
        )

    try:
        with open(f"{out_prefix}_bundle.json", "w", encoding="utf-8") as f:
            sanitized_payload = sanitize_for_json(payload)
            json.dump(sanitized_payload, f, ensure_ascii=False, indent=2)
    except IOError as e:
        logger.error(f"Failed to write bundle file: {e}")
        
    if csv_output_path:
        save_timeline_as_csv(events, csv_output_path)

    if USE_GEMINI:
        try:
            summary = await summarize_gemini(payload, events)
        except Exception as e:
            summary = f"[Gemini error] {type(e).__name__}: {e}\n\n" + summarize_rule_based(payload, events)
    else:
        summary = summarize_rule_based(payload, events)

    try:
        tldr_bullets, key_events = extract_tldr_and_key_events(summary)
        if not key_events:
            key_events = _events_to_key_events(events)
        write_summary_csv(tldr_bullets, f"{out_prefix}_summary.csv")
        write_key_events_csv(key_events, f"{out_prefix}_key_events.csv")
    except Exception as e:
        logger.error(f"Failed to export summary/key events CSVs: {e}")

    try:
        with open(f"{out_prefix}_summary.txt", "w", encoding="utf-8") as f:
            f.write(summary)
    except IOError as e:
        logger.error(f"Failed to write summary file: {e}")

    return summary

def run(app_id=None, phone=None, email=None, out_prefix="journey", latest_n=None, csv_output_path=None, fd_page=None, fd_page_size=None) -> str:
    return asyncio.run(run_async(app_id, phone, email, out_prefix, latest_n, csv_output_path, fd_page, fd_page_size))

def _prompt_for_identifier():
    print("\n🤖 Customer Journey — provide one identifier")
    print("    (Only one is required; the other will be auto-resolved if possible.)")
    app_id = input("  📱 Application ID (press Enter if unknown): ").strip() or None
    phone  = input("  ☎️  Phone Number (press Enter if unknown): ").strip() or None
    if not any([app_id, phone]):
        print("  ⚠️  Please enter at least one of Application ID or Phone Number.")
        app_id = input("  📱 Application ID (optional): ").strip() or None
        phone  = input("  ☎️  Phone Number (optional): ").strip() or None
    return app_id, phone

# -------- Added: user-friendly spinner + log suppression (non-debug) --------
@contextmanager
def user_friendly_progress(message: str = "Preparing unified journey report…"):
    import threading
    import itertools

    spinner_frames = ["⏳", "⌛", "🕒", "🕓", "🕔", "🕕"]
    cycle = itertools.cycle(spinner_frames)
    stop_event = threading.Event()

    def spin():
        while not stop_event.is_set():
            try:
                sys.stdout.write(f"\r{next(cycle)} {message} ")
                sys.stdout.flush()
                time.sleep(0.15)
            except Exception:
                # If stdout is not writable (e.g., piped), just stop gracefully
                break

    # Suppress all logging while spinner is active
    previous_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)

    t = threading.Thread(target=spin, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop_event.set()
        t.join(timeout=0.25)
        # Clear spinner line
        try:
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()
        except Exception:
            pass
        # Restore logging
        logging.disable(previous_disable)
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Customer Journey Orchestrator (Optimized)")
    p.add_argument("--id", dest="app_id", help="Application ID")
    p.add_argument("--phone", help="Phone number")
    p.add_argument("--email", help="Email address")
    p.add_argument("--latest-n", type=int, default=None, help="Per-source row limit")
    p.add_argument("--fd-page", type=int, default=None, help="Freshdesk page")
    p.add_argument("--fd-page-size", type=int, default=None, help=f"Freshdesk page size (max {FD_MAX_PAGE_SIZE})")
    p.add_argument("--out-prefix", default="journey", help="Output file prefix")
    p.add_argument("--csv", dest="csv_output_path", help="Path to save timeline CSV")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = p.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    app_id, phone = args.app_id, args.phone
    email = args.email

    if not any([app_id, phone]):
        try:
            app_id, phone = _prompt_for_identifier()
            if not any([app_id, phone]):
                print("No identifier provided. Exiting.")
                sys.exit(1)
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled. Exiting.")
            sys.exit(1)

    try:
        # Quiet, user-friendly mode by default (spinner + suppressed logs)
        if args.debug:
            with timer("total_runtime"):
                summary = run(
                    app_id=app_id, 
                    phone=phone, 
                    email=email, 
                    out_prefix=args.out_prefix, 
                    latest_n=args.latest_n,
                    csv_output_path=args.csv_output_path,
                    fd_page=args.fd_page,
                    fd_page_size=args.fd_page_size
                )
        else:
            with user_friendly_progress("Compiling data across Slack, Telecalling, PartnerDB, Onboarding & Freshdesk…"):
                summary = run(
                    app_id=app_id, 
                    phone=phone, 
                    email=email, 
                    out_prefix=args.out_prefix, 
                    latest_n=args.latest_n,
                    csv_output_path=args.csv_output_path,
                    fd_page=args.fd_page,
                    fd_page_size=args.fd_page_size
                )

        # Show only the final output cleanly (no extra headers)
        print(summary)
    except Exception as e:
        logger.critical(f"Unhandled error: {e}", exc_info=True)
        print(f"\n💥 Critical error. Check logs. Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
