# customer_support_nlp.py
# Conversational agent that understands free-form prompts with Gemini,
# fetches unified data via your orchestrator, and returns concise summaries.
# - Multi-turn: asks for missing IDs and remembers context
# - Analytical intents: timeline bottleneck (longest gap between events)
# - Centralized Gemini client with retries + graceful fallbacks
# - Privacy redaction; structured fallback output if Gemini is down

from __future__ import annotations
import os, re, sys, json, logging, warnings, time, math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable, Literal
from datetime import datetime, timezone, timedelta
from functools import lru_cache

from dotenv import load_dotenv
load_dotenv()  # .env with GEMINI_API_KEY, GEMINI_MODEL, CSNLP_PRIVACY

# --------------------------
# Quiet logs / warnings
# --------------------------
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    message=r"pandas only supports SQLAlchemy connectable .*",
    category=UserWarning,
)
logging.disable(logging.CRITICAL)

# Ensure sibling imports work even when invoked via absolute path
sys.path.insert(0, os.path.dirname(__file__))

# ===========================
# Orchestrator integration
# ===========================
try:
    import customer_journey_orchestrator as CJO
except Exception as e:
    raise RuntimeError(
        "Customer_Support_nlp: could not import 'customer_journey_orchestrator'."
    ) from e

_resolve_identifiers_initial = getattr(CJO, "_resolve_identifiers_initial", None)
collect_slices = getattr(CJO, "collect_slices", None)
_first_non_empty = getattr(CJO, "_first_non_empty", None)

if not callable(_resolve_identifiers_initial) or not callable(collect_slices):
    raise RuntimeError(
        "Customer_Support_nlp: orchestrator must expose "
        "'_resolve_identifiers_initial' and 'collect_slices'."
    )

if _first_non_empty is None:
    def _first_non_empty(*vals):
        for v in vals:
            if v is not None and str(v).strip() != "":
                return v
        return None

# ===========================
# Utilities
# ===========================
def _digits_only(s: Optional[str]) -> str:
    return "".join(ch for ch in (s or "") if ch.isdigit())

def _parse_ts(val: Any) -> Optional[datetime]:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    if isinstance(val, (int, float)):
        try:
            return datetime.fromtimestamp(float(val), tz=timezone.utc)
        except Exception:
            return None
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return None

def _fmt_ts(ts_iso_utc: Optional[str]) -> str:
    return ts_iso_utc or "unknown time"

def _mask_phone(phone: Optional[str], privacy_mode: str) -> Optional[str]:
    if not phone:
        return None
    if privacy_mode == "internal":
        return phone
    d = _digits_only(phone)
    if not d:
        return phone
    if len(d) <= 4:
        return "x" * len(d)
    return f"{'x'*(len(d)-4)}{d[-4:]}"

def _mask_email(email: Optional[str], privacy_mode: str) -> Optional[str]:
    if not email:
        return None
    if privacy_mode == "internal":
        return email
    try:
        name, domain = email.split("@", 1)
        if not name:
            return f"*@{domain}"
        if len(name) <= 2:
            return f"{name[0]}*@{domain}"
        return f"{name[0]}***@{domain}"
    except Exception:
        return "****"

def _iso_utc(dt: Optional[datetime]) -> Optional[str]:
    return dt and dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

# ===========================
# Gemini Client (centralized)
# ===========================
class GeminiClient:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
        self._ready = False
        self._model = None
        if self.api_key:
            try:
                import google.generativeai as genai  # type: ignore
                genai.configure(api_key=self.api_key)
                self._genai = genai
                self._model = genai.GenerativeModel(self.model_name)
                self._ready = True
            except Exception:
                self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def _retry_call(self, fn: Callable[[], Any], attempts: int = 3, base_delay: float = 0.8) -> Any:
        last_err = None
        for i in range(attempts):
            try:
                return fn()
            except Exception as e:
                last_err = e
                time.sleep(base_delay * (2 ** i))
        raise last_err or RuntimeError("Unknown Gemini error")

    def generate_text(self, prompt: str) -> Optional[str]:
        if not self.ready:
            return None
        try:
            resp = self._retry_call(lambda: self._model.generate_content(prompt))
            txt = getattr(resp, "text", None)
            return txt.strip() if isinstance(txt, str) and txt.strip() else None
        except Exception:
            return None

    def classify_intent_json(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Ask Gemini to classify intent + extract entities with JSON output.
        """
        if not self.ready:
            return None

        schema = """
Respond ONLY with JSON (no prose) matching:
{
  "intent": "get_contact_details|get_latest_status|analyze_timeline|get_slack_conversation|get_unified_any_data|general_info",
  "requires_identifier": true/false,
  "entities": {"phone": "string|null", "app_id": "string|null", "email": "string|null"},
  "confidence": 0.0
}
"""
        classify_prompt = f"""
You are an assistant for a customer support analyst system.

Classify the user's query and extract entities.

Available intents:
- get_contact_details: wants phone/email
- get_latest_status: wants current status/stage/substatus
- analyze_timeline: wants to analyze delays/bottlenecks and longest gap between events
- get_slack_conversation: wants Slack messages
- get_unified_any_data: unified summary of whatever is found
- general_info: generic query about a customer (fallback)

IMPORTANT:
- "requires_identifier" must be true if the intent needs application id or phone/email.
- Extract phone, app_id, email if present; else null.

User Query:
\"\"\"{prompt}\"\"\"

{schema}
"""
        out = self.generate_text(classify_prompt)
        if not out:
            return None
        try:
            # Some models might wrap JSON with ```json fences; strip them.
            out = out.strip()
            out = re.sub(r"^```json\s*|\s*```$", "", out, flags=re.I)
            data = json.loads(out)
            # Lightweight validation
            if "intent" in data and "entities" in data:
                return data
        except Exception:
            return None
        return None

    def summarize(self, context: str, instruction: Optional[str] = None) -> Optional[str]:
        if not self.ready:
            return None
        instr = instruction or (
            "You are a helpful customer support analyst. Brevity is key. "
            "Summarize into a single concise paragraph (2-4 sentences). "
            "Focus on the most recent and important activity. Do not use markdown or lists."
        )
        prompt = f"{instr}\n\n--- DATA ---\n{context}\n--- END DATA ---\n\nConcise Summary:"
        return self.generate_text(prompt)

# Singleton Gemini client
_GEMINI = GeminiClient()

# ===========================
# Regex Fallbacks (NLP)
# ===========================
_GREETING_ONLY_RX = re.compile(r"^\s*(hi|hey|hello|yo|good\s+(morning|afternoon|evening))\s*[!.]?\s*$", re.I)
_COMPANY_KEYWORDS_RX = re.compile(
    r"\b(app(?:lication)?\s*id|onboarding|kyc|cpv|partner|telecalling|freshdesk|ticket|loan|limit|disbursement|portal|substatus|verification|slack)\b", re.I
)
_PHONE_RX = re.compile(r"(?:\+?\d[\s-]?)(?:\d[\s-]?){7,13}\d")
_EMAIL_RX = re.compile(r"[\w\.\-+%]+@[\w\.\-]+\.[A-Za-z]{2,}")
_APPID_RX = re.compile(r"\b(app(?:lication)?(?:\s*[_-]?\s*id)?)[^\d]{0,3}(\d{4,12})\b", re.I)

def _rx_extract_entities(prompt: str) -> Dict[str, Optional[str]]:
    p = prompt.strip()
    app_id = None; phone = None; email = None
    m = _APPID_RX.search(p)
    if m: app_id = m.group(2).strip()
    m = _PHONE_RX.search(p)
    if m:
        phone = re.sub(r"\D", "", m.group(0))
        if len(phone) < 10:
            phone = None
    m = _EMAIL_RX.search(p)
    if m: email = m.group(0).strip()
    return {"app_id": app_id, "phone": phone, "email": email}

def _rx_classify(prompt: str) -> Tuple[str, bool]:
    p = prompt.strip()
    has_any_entity = any(_rx_extract_entities(p).values())
    has_company_kw = bool(_COMPANY_KEYWORDS_RX.search(p))
    if re.search(r"\b(contact|contact\s+details|contact\s+info)\b|\b(email|mail)\b.*\b(phone|mobile|number)\b|\b(phone|mobile)\b.*\b(email|mail)\b", p, re.I):
        return "get_contact_details", True
    if re.search(r"\b(latest|current)\b.*\b(status|stage|substatus)\b|\bwhat'?s\b.*\bstatus\b", p, re.I):
        return "get_latest_status", True
    if re.search(r"\bslack\b.*\b(messages?|conversation|chat|thread|only|data)\b|what.*\bon\s+slack\b|show\s+slack\b", p, re.I):
        return "get_slack_conversation", True
    if re.search(r"\b(longest|bottleneck|delay|took the longest|time between|gap)\b.*\b(onboard|timeline|stage|event)\b", p, re.I):
        return "analyze_timeline", True
    if has_company_kw or has_any_entity:
        return "get_unified_any_data", True
    return "general_info", False

# ===========================
# Conversation State
# ===========================
@dataclass
class TurnState:
    intent: Optional[str] = None
    requires_identifier: bool = False
    app_id: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    # For analytical sub-requests (e.g., longest gap results)
    last_analysis: Optional[Dict[str, Any]] = None

@dataclass
class CSNLPConfig:
    privacy_mode: Literal["redacted", "internal"] = os.getenv("CSNLP_PRIVACY", "redacted")  # mask by default
    latest_n: Optional[int] = 50
    fd_page: Optional[int] = None
    fd_page_size: Optional[int] = None
    show_activity: bool = True

@dataclass
class CSNLPAnswer:
    intent: str
    identifiers: Dict[str, Optional[str]]
    answer: Dict[str, Any]
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    message: Optional[str] = None

# ===========================
# Orchestrator wrappers + cache
# ===========================
def _activity(msg: str, cfg: CSNLPConfig) -> None:
    if getattr(cfg, "show_activity", False):
        print(f"\n… {msg}")

@lru_cache(maxsize=128)
def _collect_slices_cached(key: str) -> Dict[str, Dict[str, Any]]:
    # Key is a stable serialization of (app_id, phone, email, latest_n, fd_page, fd_page_size)
    payload = json.loads(key)
    return collect_slices(**payload) or {}

def _collect_all(app_id: Optional[str], phone: Optional[str], email: Optional[str], cfg: CSNLPConfig) -> Dict[str, Dict[str, Any]]:
    key = json.dumps({
        "app_id": app_id, "phone": phone, "email": email,
        "latest_n": cfg.latest_n, "fd_page": cfg.fd_page, "fd_page_size": cfg.fd_page_size
    }, sort_keys=True)
    return _collect_slices_cached(key)

# ===========================
# Analytical: timeline durations
# ===========================
def _event_time(ev: Dict[str, Any]) -> Optional[datetime]:
    for k in ("at","time","timestamp","created_at","updated_time","ts"):
        if ev.get(k):
            return _parse_ts(ev[k])
    return None

def _event_label(ev: Dict[str, Any]) -> str:
    return _first_non_empty(ev.get("stage"), ev.get("status"), ev.get("name"), ev.get("substatus"), ev.get("info"), ev.get("note")) or "event"

def calculate_stage_durations(timeline_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    1) sort by timestamp
    2) compute duration between consecutive events
    3) return longest gap and per-gap details
    """
    events = [(e, _event_time(e)) for e in (timeline_events or [])]
    events = [x for x in events if x[1] is not None]
    events.sort(key=lambda x: x[1])

    gaps: List[Dict[str, Any]] = []
    for (e1, t1), (e2, t2) in zip(events, events[1:]):
        dur = (t2 - t1).total_seconds()
        gaps.append({
            "from_label": _event_label(e1),
            "to_label": _event_label(e2),
            "from_ts_utc": _iso_utc(t1),
            "to_ts_utc": _iso_utc(t2),
            "duration_hours": round(dur/3600.0, 3),
            "duration_days": round(dur/86400.0, 3),
        })
    longest = max(gaps, key=lambda g: g["duration_hours"]) if gaps else None
    return {"gaps": gaps, "longest": longest}

# ===========================
# Answer builders
# ===========================
def _answer_contact_details(slices: Dict[str, Dict[str, Any]], cfg: CSNLPConfig) -> Tuple[Dict[str, Any], str]:
    phones: List[Tuple[str, Dict[str, Any]]] = []
    emails: List[Tuple[str, Dict[str, Any]]] = []

    # PartnerDB
    for c in slices.get("PartnerDB", {}).get("comments", []) or []:
        candidates = {
            "phone": _first_non_empty(c.get("phone"), c.get("mobile"), c.get("contact_number"), c.get("customer_phone")),
            "email": _first_non_empty(c.get("email"), c.get("customer_email")),
        }
        text = _first_non_empty(c.get("application_comment"), c.get("comment"), c.get("note"), c.get("text"), c.get("remarks")) or ""
        m_phone = re.search(r"(\+?\d[\d\-\s]{6,}\d)", text)
        m_mail = re.search(r"[\w\.\-+%]+@[\w\.\-]+\.[A-Za-z]{2,}", text)
        phone = candidates["phone"] or (m_phone.group(1) if m_phone else None)
        email = candidates["email"] or (m_mail.group(0) if m_mail else None)
        ts = _parse_ts(_first_non_empty(c.get("created_at"), c.get("time"), c.get("timestamp"), c.get("ts")))
        if phone: phones.append((str(phone).strip(), {"source": "PartnerDB", "ts_utc": _iso_utc(ts)}))
        if email: emails.append((str(email).strip(), {"source": "PartnerDB", "ts_utc": _iso_utc(ts)}))

    # Telecalling
    for call in slices.get("Telecalling", {}).get("calls", []) or []:
        phone = _first_non_empty(call.get("customer_phone"), call.get("phone"), call.get("mobile"))
        email = _first_non_empty(call.get("email"))
        ts = _parse_ts(_first_non_empty(call.get("time"), call.get("timestamp"), call.get("created_at"), call.get("call_date")))
        if phone: phones.append((str(phone).strip(), {"source": "Telecalling", "ts_utc": _iso_utc(ts)}))
        if email: emails.append((str(email).strip(), {"source": "Telecalling", "ts_utc": _iso_utc(ts)}))

    # Freshdesk
    for t in slices.get("Freshdesk", {}).get("tickets", []) or []:
        phone = _first_non_empty(t.get("phone_number"), t.get("phone"), t.get("mobile"), t.get("contact_number"), t.get("customer_phone"))
        email = _first_non_empty(t.get("email"), t.get("requester_email"), t.get("customer_email"))
        ts = _parse_ts(_first_non_empty(t.get("updated_at"), t.get("created_at"), t.get("createdon"), t.get("updatedon")))
        if phone: phones.append((str(phone).strip(), {"source": "Freshdesk", "ts_utc": _iso_utc(ts)}))
        if email: emails.append((str(email).strip(), {"source": "Freshdesk", "ts_utc": _iso_utc(ts)}))

    # Slack
    for ev in slices.get("Slack", {}).get("events", []) or []:
        text = _first_non_empty(ev.get("text"), ev.get("message")) or ""
        m_phone = re.search(r"(\+?\d[\d\-\s]{6,}\d)", text)
        m_mail = re.search(r"[\w\.\-+%]+@[\w\.\-]+\.[A-Za-z]{2,}", text)
        ts = _parse_ts(_first_non_empty(ev.get("ts"), ev.get("timestamp"), ev.get("time"), ev.get("date")))
        if m_phone: phones.append((m_phone.group(1), {"source": "Slack", "ts_utc": _iso_utc(ts)}))
        if m_mail: emails.append((m_mail.group(0), {"source": "Slack", "ts_utc": _iso_utc(ts)}))

    def _rank(src: str) -> int:
        order = {"PartnerDB": 0, "Freshdesk": 1, "Telecalling": 2, "Slack": 3}
        return order.get(src, 9)

    def _select_best(pairs: List[Tuple[str, Dict[str, Any]]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        if not pairs:
            return None, []
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for val, prov in pairs:
            key = val.strip().lower()
            buckets.setdefault(key, []).append(prov)
        best_val, best_provs = None, []
        best_score = (99, datetime.min.replace(tzinfo=timezone.utc))
        for key, provs in buckets.items():
            best_prov = sorted(
                provs,
                key=lambda p: (_rank(p.get("source","")), _parse_ts(p.get("ts_utc") or "") or datetime.min.replace(tzinfo=timezone.utc))
            )[0]
            score = (_rank(best_prov.get("source","")), _parse_ts(best_prov.get("ts_utc") or "") or datetime.min.replace(tzinfo=timezone.utc))
            if score < best_score:
                best_score = score
                best_val = key
                best_provs = provs
        return best_val, best_provs

    best_phone, _ = _select_best(phones)
    best_email, _ = _select_best(emails)
    masked_phone = _mask_phone(best_phone, cfg.privacy_mode) if best_phone else None
    masked_email = _mask_email(best_email, cfg.privacy_mode) if best_email else None

    message = (
        f"Contact: phone {masked_phone}, email {masked_email}." if (masked_phone and masked_email)
        else f"Contact: phone {masked_phone}. No email found." if masked_phone
        else f"Contact: email {masked_email}. No phone found." if masked_email
        else "No contact details found."
    )
    return {"phone": masked_phone, "email": masked_email}, message

def _answer_latest_status(slices: Dict[str, Dict[str, Any]], cfg: CSNLPConfig) -> Tuple[Dict[str, Any], str]:
    timeline = slices.get("Onboarding", {}).get("timeline", []) or []

    def _ts_of(ev: Dict[str, Any]) -> Optional[datetime]:
        return _event_time(ev)

    rows = sorted([(ev, _ts_of_tl := _ts_of(ev)) for ev in timeline], key=lambda r: (r[1] is None, r[1] or datetime.min.replace(tzinfo=timezone.utc)))
    if rows:
        ev, ts = rows[-1]
        status = _first_non_empty(ev.get("stage"), ev.get("status"), ev.get("name"))
        substatus = _first_non_empty(ev.get("substatus"), ev.get("info"), ev.get("note"))
        answer = {"status": status, "substatus": substatus or None, "as_of_utc": _iso_utc(ts)}
        msg = f"Latest status: {status}" + (f" — {substatus}" if substatus else "") + (f" (as of { _fmt_ts(answer['as_of_utc']) })" if answer["as_of_utc"] else "")
        return answer, msg

    comments = slices.get("PartnerDB", {}).get("comments", []) or []
    if comments:
        comments_sorted = sorted(
            comments,
            key=lambda c: (_parse_ts(_first_non_empty(c.get("created_at"), c.get("time"), c.get("timestamp"), c.get("ts"))) or datetime.min.replace(tzinfo=timezone.utc))
        )
        last = comments_sorted[-1]
        ts = _parse_ts(_first_non_empty(last.get("created_at"), last.get("time"), last.get("timestamp"), last.get("ts")))
        text = _first_non_empty(last.get("application_comment"), last.get("comment"), last.get("note"), last.get("text"))
        answer = {
            "status": "See latest partner comment",
            "substatus": _first_non_empty(last.get("status"), last.get("substatus")),
            "as_of_utc": _iso_utc(ts),
            "narrative": text and text[:240]
        }
        msg = f"Latest known from PartnerDB: {(text or '').strip()[:180]}"
        return answer, msg

    return ({"status": None, "substatus": None, "as_of_utc": None}, "No latest status found in Onboarding or PartnerDB.")

def _answer_slack_conversation(slices: Dict[str, Dict[str, Any]], max_items: int = 10) -> Tuple[Dict[str, Any], str]:
    events = slices.get("Slack", {}).get("events", []) or []

    def ts_of(e: Dict[str, Any]) -> Optional[datetime]:
        return _event_time(e)

    events_sorted = sorted([e for e in events if ts_of(e)], key=lambda e: ts_of(e) or datetime.min.replace(tzinfo=timezone.utc))
    items = []
    for e in events_sorted[-max_items:]:
        t = ts_of(e); tstr = _iso_utc(t)
        user = _first_non_empty(e.get("user_name"), e.get("user"), e.get("author"), e.get("username"))
        channel = _first_non_empty(e.get("channel_name"), e.get("channel"))
        text = (_first_non_empty(e.get("text"), e.get("message")) or "").strip().replace("\n", " ")
        if len(text) > 180: text = text[:179] + "…"
        items.append({"ts_utc": tstr, "channel": channel, "user": user, "text": text})

    if not items:
        return ({"count": 0, "items": []}, "No Slack messages found for this customer.")

    first_ts = items[0]["ts_utc"]; last_ts = items[-1]["ts_utc"]
    msg = (
        f"Slack: {len(events)} messages"
        + (f" from {first_ts}" if first_ts else "")
        + (f" to {last_ts}" if last_ts else "")
        + f". Showing last {min(max_items, len(events_sorted))}:\n"
        + "\n".join(
            f"• {it['ts_utc'] or 'unknown'}{(' | #' + (it['channel'] or '')) if it['channel'] else ''}"
            f"{(' | ' + it['user']) if it['user'] else ''}: {it['text']}"
            for it in items
        )
    )
    return ({"count": len(events), "items": items}, msg)

def _answer_unified_any_data(slices: Dict[str, Dict[str, Any]], identifiers: Dict[str, Optional[str]], cfg: CSNLPConfig, slack_snippets: int = 3) -> Tuple[Dict[str, Any], str]:
    provenance: List[Dict[str, Any]] = []
    slack_events = slices.get("Slack", {}).get("events", []) or []
    tele_calls  = slices.get("Telecalling", {}).get("calls", []) or []
    partner_cmts= slices.get("PartnerDB", {}).get("comments", []) or []
    onb_tl      = slices.get("Onboarding", {}).get("timeline", []) or []
    fd_tickets  = slices.get("Freshdesk", {}).get("tickets", []) or []

    total_hits = sum([len(slack_events), len(tele_calls), len(partner_cmts), len(onb_tl), len(fd_tickets)])
    if total_hits == 0:
        ident_str = identifiers.get("app_id") or identifiers.get("phone") or identifiers.get("email") or "the customer"
        if identifiers.get("phone"):
            ident_str = _mask_phone(identifiers["phone"], cfg.privacy_mode) or ident_str
        if identifiers.get("email") and cfg.privacy_mode == "redacted":
            ident_str = "customer's email"
        return ({"found": False, "counts": {"slack":0,"telecalling":0,"partnerdb":0,"onboarding":0,"freshdesk":0}},
                f"No data found for {ident_str} across Slack, Telecalling, PartnerDB, Onboarding, or Freshdesk.")

    contact_ans, contact_msg = _answer_contact_details(slices, cfg)
    status_ans,  status_msg  = _answer_latest_status(slices, cfg)

    def _ts_of(e: Dict[str, Any]) -> Optional[datetime]:
        return _event_time(e)

    slack_sorted = sorted([e for e in slack_events if _ts_of(e)], key=lambda e: _ts_of(e) or datetime.min.replace(tzinfo=timezone.utc))
    tail = slack_sorted[-slack_snippets:] if slack_sorted else []

    def _snip(s: str, n: int = 160) -> str:
        s = (s or "").strip().replace("\n", " ")
        return s if len(s) <= n else s[:n-1] + "…"

    slack_lines = []
    for e in tail:
        t = _ts_of(e); tstr = _iso_utc(t)
        channel = _first_non_empty(e.get("channel_name"), e.get("channel"))
        user = _first_non_empty(e.get("user_name"), e.get("user"), e.get("author"))
        text = _snip(_first_non_empty(e.get("text"), e.get("message")) or "")
        slack_lines.append(f"{tstr or 'unknown'}{(' | #' + channel) if channel else ''}{(' | ' + user) if user else ''}: {text}")

    counts_line = f"Found data — Slack: {len(slack_events)}, Telecalling: {len(tele_calls)}, PartnerDB: {len(partner_cmts)}, Onboarding: {len(onb_tl)}, Freshdesk: {len(fd_tickets)}."
    details_parts: List[str] = [counts_line]
    if contact_ans.get("phone") or contact_ans.get("email"): details_parts.append(contact_msg)
    if status_ans.get("status") or status_ans.get("narrative"): details_parts.append(status_msg)
    if slack_lines: details_parts += ["Recent Slack:"] + [f"• {ln}" for ln in slack_lines]
    details_text = "\n".join(details_parts).strip()

    summary = _GEMINI.summarize(details_text)
    msg = summary or details_text
    answer = {
        "found": True,
        "counts": {"slack": len(slack_events), "telecalling": len(tele_calls), "partnerdb": len(partner_cmts), "onboarding": len(onb_tl), "freshdesk": len(fd_tickets)},
        "contact": contact_ans,
        "status": status_ans,
        "slack_recent": slack_lines
    }
    return answer, msg

def _answer_timeline_analysis(slices: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
    timeline = slices.get("Onboarding", {}).get("timeline", []) or []
    if not timeline:
        return {"analysis": None}, "No onboarding timeline found."
    result = calculate_stage_durations(timeline)
    longest = result.get("longest")
    if not longest:
        return {"analysis": result}, "Not enough events to compute gaps."
    # Summarize naturally (Gemini) if available
    raw = (
        f"Longest gap: {longest['duration_hours']} hours ({longest['duration_days']} days) "
        f"from '{longest['from_label']}' at {longest['from_ts_utc']} "
        f"to '{longest['to_label']}' at {longest['to_ts_utc']}."
    )
    summary = _GEMINI.summarize(raw, instruction="Explain in one sentence the bottleneck between two stages for a customer onboarding timeline.")
    return {"analysis": result}, (summary or raw)

# ===========================
# Agent (multi-turn)
# ===========================
class CSNLPAgent:
    def __init__(self, config: Optional[CSNLPConfig] = None):
        self.cfg = config or CSNLPConfig()
        self.state = TurnState()

    def _classify_and_extract(self, prompt: str) -> Tuple[str, bool, Dict[str, Optional[str]], float]:
        """
        Try Gemini classification; fall back to regex.
        Returns: (intent, requires_identifier, entities, confidence)
        """
        # Gemini-first
        data = _GEMINI.classify_intent_json(prompt) if _GEMINI.ready else None
        if data:
            ent = data.get("entities") or {}
            entities = {
                "app_id": ent.get("app_id"),
                "phone": ent.get("phone"),
                "email": ent.get("email"),
            }
            return data.get("intent") or "general_info", bool(data.get("requires_identifier", False)), entities, float(data.get("confidence") or 0.0)

        # Fallback — regex
        intent, req = _rx_classify(prompt)
        entities = _rx_extract_entities(prompt)
        return intent, req, entities, 0.5

    def _resolve_ids(self, app_id: Optional[str], phone: Optional[str], email: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        try:
            return _resolve_identifiers_initial(app_id, phone, email)
        except Exception:
            # fallback: pass-through
            return app_id, phone, email

    def _need_identifier(self, intent: str) -> bool:
        return intent in {"get_contact_details", "get_latest_status", "get_slack_conversation", "get_unified_any_data", "analyze_timeline"}

    def _slices_with_activity(self, app_id: Optional[str], phone: Optional[str], email: Optional[str], reason: str) -> Dict[str, Dict[str, Any]]:
        _activity(f"Searching {reason} across Slack, Telecalling, PartnerDB, Onboarding, Freshdesk…", self.cfg)
        return _collect_all(app_id, phone, email, self.cfg)

    def ask(self, prompt: str, app_id: Optional[str] = None, phone: Optional[str] = None, email: Optional[str] = None) -> CSNLPAnswer:
        p = prompt.strip()
        # Greeting short-circuit
        if _GREETING_ONLY_RX.match(p):
            return CSNLPAnswer(
                intent="small_talk",
                identifiers={"app_id": None, "phone": None, "email": None},
                answer={},
                message="Hi! How can I help you with this customer?"
            )

        # Classify + extract
        intent, requires_id, ents, _conf = self._classify_and_extract(p)

        # Merge with explicit args and prior state (conversation memory)
        app_id = app_id or ents.get("app_id") or self.state.app_id
        phone  = phone  or ents.get("phone")  or self.state.phone
        email  = email  or ents.get("email")  or self.state.email

        # Allow follow-up messages that are just identifiers (e.g., "9622506497")
        if not (app_id or phone or email):
            # try regex again to see if user pasted only an ID
            ids_only = _rx_extract_entities(p)
            app_id = app_id or ids_only.get("app_id")
            phone  = phone  or ids_only.get("phone")
            email  = email  or ids_only.get("email")

        # Resolve via orchestrator helper
        app_id, phone, email = self._resolve_ids(app_id, phone, email)

        # Persist to state for multi-turn continuity
        self.state.intent = intent
        self.state.requires_identifier = requires_id
        self.state.app_id = app_id
        self.state.phone = phone
        self.state.email = email

        identifiers = {"app_id": app_id, "phone": phone, "email": email}

        # If identifier is required but missing, ask for it
        if self._need_identifier(intent) and not any([app_id, phone, email]):
            return CSNLPAnswer(
                intent=intent,
                identifiers=identifiers,
                answer={},
                notes=["Identifier required."],
                message="I can help. Please share the customer's application ID or phone number (or email)."
            )

        # Route intents
        try:
            if intent == "get_contact_details":
                slices = self._slices_with_activity(app_id, phone, email, "for contact details")
                answer, msg = _answer_contact_details(slices, self.cfg)
                return CSNLPAnswer(intent=intent, identifiers=identifiers, answer=answer, message=msg)

            if intent == "get_latest_status":
                slices = self._slices_with_activity(app_id, phone, email, "for latest status")
                answer, msg = _answer_latest_status(slices, self.cfg)
                return CSNLPAnswer(intent=intent, identifiers=identifiers, answer=answer, message=msg)

            if intent == "get_slack_conversation":
                slices = self._slices_with_activity(app_id, phone, email, "for Slack messages")
                answer, msg = _answer_slack_conversation(slices, 10)
                return CSNLPAnswer(intent=intent, identifiers=identifiers, answer=answer, message=msg)

            if intent == "analyze_timeline":
                slices = self._slices_with_activity(app_id, phone, email, "onboarding timeline for bottlenecks")
                answer, msg = _answer_timeline_analysis(slices)
                self.state.last_analysis = answer
                return CSNLPAnswer(intent=intent, identifiers=identifiers, answer=answer, message=msg)

            if intent == "get_unified_any_data":
                # Try orchestrator's own summary if provided
                try:
                    fn = getattr(CJO, "summarize_customer_journey", None) or getattr(CJO, "run_orchestrator_summary", None)
                    if callable(fn):
                        payload = fn(
                            app_id=app_id, phone=phone, email=email,
                            latest_n=self.cfg.latest_n, fd_page=self.cfg.fd_page,
                            fd_page_size=self.cfg.fd_page_size, privacy_mode=self.cfg.privacy_mode
                        )
                        if isinstance(payload, dict):
                            msg = payload.get("summary_text") or payload.get("message")
                            if isinstance(msg, str) and msg.strip():
                                return CSNLPAnswer(intent=intent, identifiers=identifiers, answer=payload, message=msg.strip())
                except Exception:
                    pass

                # Else compose unified summary here
                slices = self._slices_with_activity(app_id, phone, email, "for a unified summary")
                answer, msg = _answer_unified_any_data(slices, identifiers, self.cfg)
                return CSNLPAnswer(intent=intent, identifiers=identifiers, answer=answer, message=msg)

            # Fallback: treat as unified info
            slices = self._slices_with_activity(app_id, phone, email, "for a unified summary")
            answer, msg = _answer_unified_any_data(slices, identifiers, self.cfg)
            return CSNLPAnswer(intent=intent, identifiers=identifiers, answer=answer, message=msg)

        except Exception as e:
            # Robust error surfaced as user-friendly message
            return CSNLPAnswer(
                intent=intent,
                identifiers=identifiers,
                answer={},
                notes=[f"Internal error: {type(e).__name__}"],
                message="Something went wrong while processing the request. Try again or provide a different identifier."
            )

# ===========================
# CLI / REPL
# ===========================
def _stdin_prompt(msg: str) -> str:
    try:
        return input(msg)
    except (EOFError, KeyboardInterrupt):
        return ""

def _print_answer(ans: CSNLPAnswer):
    print("\n— Answer —")
    print(ans.message or "(no message)")

def main():
    import argparse
    p = argparse.ArgumentParser(description="Customer Support NLP — Gemini-powered conversational agent")
    p.add_argument("--prompt", required=False, help="One-shot prompt. If omitted, runs REPL.")
    p.add_argument("--privacy", choices=["redacted","internal"], default=os.getenv("CSNLP_PRIVACY","redacted"))
    p.add_argument("--latest-n", type=int, default=None)
    p.add_argument("--fd-page", type=int, default=None)
    p.add_argument("--fd-page-size", type=int, default=None)
    p.add_argument("--loop", action="store_true", help="Force REPL even if --prompt is provided.")
    p.add_argument("--no-activity", action="store_true", help="Hide the 'Searching…' line.")
    args = p.parse_args()

    cfg = CSNLPConfig(
        privacy_mode=args.privacy,
        latest_n=args.latest_n,
        fd_page=args.fd_page,
        fd_page_size=args.fd_page_size,
        show_activity=not args.no_activity,
    )
    agent = CSNLPAgent(cfg)

    run_repl = args.loop or (args.prompt is None)
    if run_repl:
        print("\n💬 Enter prompts (type 'exit' or 'quit' to stop).")
        while True:
            user_in = _stdin_prompt("> ").strip()
            if not user_in:
                continue
            if user_in.lower() in {"exit", "quit"}:
                print("Bye!")
                break
            ans = agent.ask(prompt=user_in)
            _print_answer(ans)
        return

    ans = agent.ask(prompt=args.prompt)
    _print_answer(ans)

if __name__ == "__main__":
    main()
