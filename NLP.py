#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NLP.py — Production-ready orchestrator-then-Q&A shell

IMPROVEMENTS:
✅ Caching layer (avoid redundant orchestrator calls)
✅ Progress indicators with elapsed time
✅ Input validation (prevent invalid orchestrator calls)
✅ Proper logging (file + console)
✅ Smart context hints (user always knows active customer)
✅ Retry logic with exponential backoff + jitter
✅ Streaming Gemini responses
✅ Graceful degradation
✅ On-disk summary persistence (survives restarts)
✅ Stronger identifier regex + ANSI stripping + token clipping
"""

from __future__ import annotations

import os
import re
import sys
import subprocess
import logging
import time
import random
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime, timedelta

# -------- .env load --------
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

# -------- Config --------
ORCH_PATH = Path(os.getenv("ORCHESTRATOR_PATH", Path(__file__).parent / "customer_journey_orchestrator.py")).resolve()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CACHE_DURATION_MIN = int(os.getenv("CACHE_DURATION_MIN", "10"))  # Cache summaries for 10 minutes
ORCHESTRATOR_TIMEOUT = int(os.getenv("ORCHESTRATOR_TIMEOUT", "180"))
MAX_RETRIES = 3

# On-disk persistence for summaries
BASE_DIR = Path(os.getenv("JOURNEYS_DIR", Path(__file__).parent / "journeys")).resolve()
BASE_DIR.mkdir(parents=True, exist_ok=True)

# -------- Logging Setup --------
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"nlp_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress verbose logs from other libraries
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# -------- Gemini (initialize once) --------
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=GEMINI_API_KEY)
    MODEL = genai.GenerativeModel(GEMINI_MODEL)
    _gemini_ok = True
    logger.info(f"✓ Gemini initialized: {GEMINI_MODEL}")
except Exception as e:
    logger.error(f"Gemini initialization failed: {e}")
    MODEL = None
    _gemini_ok = False

# -------- Regex (compiled once) --------
PHONE_RE = re.compile(r"\b(\d{10})\b")
# more forgiving: application id / application-id / application_id
APPID_RE = re.compile(r"(?i)(?:app\s*id|application[ _-]?id)\s*[:\-]?\s*(\d+)\b")
ANSI_RE  = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

# -------- Token control --------
MAX_CHARS = int(os.getenv("SUMMARY_MAX_CHARS", "200000"))
def _clip_summary(s: str) -> str:
    return s if len(s) <= MAX_CHARS else (s[:MAX_CHARS] + "\n\n... [truncated]")

# -------- Cache --------
class CacheEntry:
    def __init__(self, summary: str):
        self.summary = summary
        self.timestamp = datetime.now()
    def is_expired(self) -> bool:
        return datetime.now() - self.timestamp > timedelta(minutes=CACHE_DURATION_MIN)

CACHE: dict[str, CacheEntry] = {}

# -------- In-memory context --------
class Context:
    summary: Optional[str] = None
    identifier: Optional[str] = None  # "phone:1234567890" or "app_id:12345"
    customer_name: Optional[str] = None  # Extracted from summary

CTX = Context()

# -------- Input Validation --------
def validate_phone(phone: str) -> tuple[bool, Optional[str]]:
    if not phone.isdigit():
        return False, "Phone must contain only digits"
    if len(phone) != 10:
        return False, "Phone must be exactly 10 digits"
    return True, None

def validate_app_id(app_id: str) -> tuple[bool, Optional[str]]:
    if not app_id.isdigit():
        return False, "App ID must be a number"
    if int(app_id) <= 0:
        return False, "App ID must be positive"
    return True, None

# -------- Parse identifier --------
def parse_identifier(text: str) -> Optional[tuple[str, str]]:
    """Returns (type, value) or None. Type is 'app_id' or 'phone'."""
    app = APPID_RE.search(text)
    if app:
        return ("app_id", app.group(1))
    ph = PHONE_RE.search(text)
    if ph:
        return ("phone", ph.group(1))
    return None

# -------- Extract customer name from summary --------
def extract_customer_name(summary: str) -> Optional[str]:
    lines = summary.split('\n')[:10]  # first 10 lines are enough
    for line in lines:
        if 'M/S' in line or 'Customer:' in line or 'Business Name:' in line:
            m = re.search(r'(?:M/S|Customer:|Business Name:)\s*([A-Z][A-Z\s&/\.]+?)(?:\s*[-—]|$)', line)
            if m:
                return m.group(1).strip()
    # Title pattern: "## Customer Journey Report: Application ID 2922969 - M/S SUNIL ENTERPRISES"
    m2 = re.search(r'Customer Journey Report.*-\s*(M/S\s+[A-Z][A-Z\s/&\.]+)', lines[0] if lines else "")
    return m2.group(1).strip() if m2 else None

# -------- Disk persistence helpers --------
def _summary_path(id_type: str, id_value: str) -> Path:
    slug = f"{'app' if id_type=='app_id' else 'phone'}_{id_value}"
    p = BASE_DIR / slug
    p.mkdir(parents=True, exist_ok=True)
    return p / "summary.md"

def _load_summary_from_disk(id_type: str, id_value: str) -> Optional[str]:
    sp = _summary_path(id_type, id_value)
    if sp.exists():
        try:
            return sp.read_text(encoding="utf-8")
        except Exception:
            return None
    return None

def _save_summary_to_disk(id_type: str, id_value: str, summary: str) -> None:
    _summary_path(id_type, id_value).write_text(summary, encoding="utf-8")

# -------- Run orchestrator with progress --------
def run_orchestrator(id_type: str, id_value: str) -> tuple[Optional[str], Optional[str]]:
    """Returns (summary_markdown, error_message). Shows progress."""
    if not ORCH_PATH.exists():
        return None, f"Orchestrator not found at {ORCH_PATH}"

    args = [sys.executable, str(ORCH_PATH)]
    if id_type == "app_id":
        args += ["--id", id_value]
    else:  # phone
        args += ["--phone", id_value]

    start_time = time.time()
    logger.info(f"Running orchestrator for {id_type}:{id_value}")
    print("⏳ Running orchestrator", end="", flush=True)

    try:
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # progress dots + timeout
        while proc.poll() is None:
            elapsed = int(time.time() - start_time)
            if elapsed < ORCHESTRATOR_TIMEOUT:
                print(".", end="", flush=True)
                time.sleep(1)
            else:
                proc.kill()
                print(" ❌ Timeout")
                logger.error(f"Orchestrator timeout after {ORCHESTRATOR_TIMEOUT}s")
                return None, f"Orchestrator timed out after {ORCHESTRATOR_TIMEOUT}s"

        stdout, stderr = proc.communicate()
        elapsed = time.time() - start_time
        print(f" ✓ ({elapsed:.1f}s)")

    except Exception as e:
        print(" ❌ Error")
        logger.error(f"Orchestrator execution failed: {e}")
        return None, f"Failed to run orchestrator: {e}"

    # Strip ANSI and basic validation
    summary = ANSI_RE.sub("", (stdout or "").strip())
    looks_md = ("## " in summary) or ("### " in summary) or ("Executive Summary" in summary) or ("Customer Journey Report" in summary)
    if not summary or not looks_md:
        stderr_preview = (stderr or "").strip()[:200]
        logger.warning(f"Invalid summary generated. Stderr: {stderr_preview}")
        return None, "No valid summary generated. Check logs for details."

    logger.info(f"Orchestrator completed successfully in {elapsed:.1f}s")
    return summary, None

# -------- Gemini Q&A with retry and streaming --------
QNA_PROMPT = """You are an expert customer experience analyst.
Answer the user's question ONLY using the provided customer summary below.

Rules:
- If the answer is not in the summary, reply: "Information not available in summary"
- Use UTC ISO timestamps when citing events
- Be concise and action-focused

# CUSTOMER SUMMARY
"""

def answer_question_with_retry(question: str, summary: str, stream: bool = True) -> str:
    if not _gemini_ok:
        return "⚠️ Gemini unavailable"

    prompt = f"{QNA_PROMPT}\n{_clip_summary(summary)}\n\n# QUESTION\n{question}"

    for attempt in range(MAX_RETRIES):
        try:
            if stream:
                print("💬 ", end="", flush=True)
                response = MODEL.generate_content(prompt, stream=True)
                full_text = ""
                for chunk in response:
                    if getattr(chunk, "text", None):
                        print(chunk.text, end="", flush=True)
                        full_text += chunk.text
                print()
                logger.info("Answer generated (streamed) on attempt %d", attempt + 1)
                return full_text or "Information not available in summary"
            else:
                response = MODEL.generate_content(prompt)
                text = (response.text or "Information not available in summary").strip()
                logger.info("Answer generated on attempt %d", attempt + 1)
                return text

        except Exception as e:
            wait_time = (2 ** attempt) + random.uniform(0, 0.7)  # backoff + jitter
            logger.warning("Gemini attempt %d failed: %s. Retrying in %.1fs...", attempt + 1, e, wait_time)
            if attempt < MAX_RETRIES - 1:
                print(f"⚠️ Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                logger.error("Gemini failed after %d attempts", MAX_RETRIES)
                return f"⚠️ Gemini error after {MAX_RETRIES} attempts: {str(e)[:150]}"

# -------- Generic Gemini (no context) --------
def generic_answer(question: str) -> str:
    if not _gemini_ok:
        return "⚠️ No customer context and Gemini unavailable. Provide phone/app id first."
    try:
        response = MODEL.generate_content(question)
        return (response.text or "No response").strip()
    except Exception as e:
        logger.error(f"Generic answer failed: {e}")
        return f"⚠️ Error: {str(e)[:150]}"

# -------- Load customer (cache + disk) --------
class CacheEntry:
    def __init__(self, summary: str):
        self.summary = summary
        self.timestamp = datetime.now()
    def is_expired(self) -> bool:
        return datetime.now() - self.timestamp > timedelta(minutes=CACHE_DURATION_MIN)

def load_customer(id_type: str, id_value: str, force_refresh: bool = False) -> tuple[Optional[str], Optional[str]]:
    cache_key = f"{id_type}:{id_value}"

    # In-memory cache
    if not force_refresh and cache_key in CACHE:
        entry = CACHE[cache_key]
        if not entry.is_expired():
            logger.info("Cache hit for %s", cache_key)
            ttl = CACHE_DURATION_MIN - int((datetime.now() - entry.timestamp).total_seconds() / 60)
            print(f"📦 Using cached summary (expires in ~{max(ttl, 0)} min)")
            return entry.summary, None
        else:
            logger.info("Cache expired for %s", cache_key)
            del CACHE[cache_key]

    # On-disk cache
    if not force_refresh:
        disk = _load_summary_from_disk(id_type, id_value)
        if disk:
            CACHE[cache_key] = CacheEntry(disk)
            logger.info("Disk cache hit for %s", cache_key)
            print("💾 Using on-disk summary")
            return disk, None

    # Orchestrator
    summary, err = run_orchestrator(id_type, id_value)
    if err:
        return None, err

    # Cache + persist
    CACHE[cache_key] = CacheEntry(summary)
    _save_summary_to_disk(id_type, id_value, summary)
    logger.info("Cached + persisted summary for %s", cache_key)
    return summary, None

# -------- Commands/help --------
def show_help() -> str:
    return (
        "Commands:\n"
        "  customer 9876543210       - Load customer by phone\n"
        "  app id 12345              - Load customer by app ID\n"
        "  refresh                   - Reload current customer (bypass cache)\n"
        "  show summary              - Display current summary\n"
        "  show context              - Display active customer context\n"
        "  clear                     - Clear context\n"
        "  clear cache               - Clear all cached summaries\n"
        "  help                      - Show this help\n"
        "  exit                      - Exit shell\n\n"
        "Or just ask: 'did customer 9876543210 complete application?'\n"
    )

# -------- Context display --------
def show_context_hint():
    if CTX.identifier:
        id_type, id_val = CTX.identifier.split(":", 1)
        name_part = f" ({CTX.customer_name})" if CTX.customer_name else ""
        print(f"📋 Active: {id_type.replace('_', ' ').title()} {id_val}{name_part}")

# -------- REPL --------
def main() -> None:
    logger.info("=" * 50)
    logger.info("NLP Shell started")
    print("💬 Customer Support NLP Shell (Production)")
    print(f"📝 Logging to: {LOG_FILE}")
    print(f"⚡ Cache duration: {CACHE_DURATION_MIN} minutes")
    print("Type 'help' for commands\n")

    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            logger.info("NLP Shell exited")
            break

        if not q:
            continue

        low = q.lower()

        # Exit
        if low in {"exit", "quit", "q"}:
            print("Goodbye!")
            logger.info("NLP Shell exited")
            break

        # Help
        if low in {"help", "?"}:
            print(show_help())
            continue

        # Context
        if low in {"who", "context", "show context"}:
            show_context_hint()
            continue

        # Clear context
        if low in {"clear", "reset"}:
            CTX.summary = None
            CTX.identifier = None
            CTX.customer_name = None
            print("✓ Context cleared")
            logger.info("Context cleared")
            continue

        # Clear cache (fix count)
        if low == "clear cache":
            n = len(CACHE)
            CACHE.clear()
            print(f"✓ Cleared {n} cached summaries")
            logger.info("Cache cleared (%d entries)", n)
            continue

        # Show summary
        if low == "show summary":
            if not CTX.summary:
                print("⚠️ No active context. Provide phone/app id first.")
                continue
            show_context_hint()
            lines = CTX.summary.splitlines()
            print("\n".join(lines[:60]))
            if len(lines) > 60:
                print(f"\n... (showing 60/{len(lines)} lines)")
            continue

        # Parse identifier from input
        ident = parse_identifier(q)

        # Validate identifier if found
        if ident:
            id_type, id_val = ident
            if id_type == "phone":
                valid, err = validate_phone(id_val)
                if not valid:
                    print(f"⚠️ Invalid phone number: {err}")
                    logger.warning("Invalid phone: %s", id_val)
                    continue
            else:  # app_id
                valid, err = validate_app_id(id_val)
                if not valid:
                    print(f"⚠️ Invalid app ID: {err}")
                    logger.warning("Invalid app_id: %s", id_val)
                    continue

        # Refresh command
        if low.startswith("refresh"):
            if ident:
                id_type, id_val = ident
                print(f"🔄 Force refreshing {id_type.replace('_', ' ')} {id_val}...")
                logger.info("Force refresh requested for %s:%s", id_type, id_val)
                summary, err = load_customer(id_type, id_val, force_refresh=True)
                if err:
                    print(f"⚠️ {err}")
                    continue
                CTX.summary = summary
                CTX.identifier = f"{id_type}:{id_val}"
                CTX.customer_name = extract_customer_name(summary)
                show_context_hint()
                print("✓ Summary refreshed. Ask your questions.")
            elif CTX.identifier:
                id_type, id_val = CTX.identifier.split(":", 1)
                print("🔄 Force refreshing current context...")
                logger.info("Force refresh for current context: %s", CTX.identifier)
                summary, err = load_customer(id_type, id_val, force_refresh=True)
                if err:
                    print(f"⚠️ {err}")
                    continue
                CTX.summary = summary
                CTX.customer_name = extract_customer_name(summary)
                show_context_hint()
                print("✓ Summary refreshed.")
            else:
                print("⚠️ No active context to refresh. Provide phone/app id.")
            continue

        # If identifier found in question, load it and answer
        if ident:
            id_type, id_val = ident
            print(f"🔍 Loading {id_type.replace('_', ' ')} {id_val}...")
            logger.info("Loading customer: %s:%s", id_type, id_val)

            summary, err = load_customer(id_type, id_val)
            if err:
                print(f"⚠️ {err}")
                if CTX.summary:
                    print("⚙️ Using existing context instead...")
                    answer_question_with_retry(q, CTX.summary)
                continue

            CTX.summary = summary
            CTX.identifier = f"{id_type}:{id_val}"
            CTX.customer_name = extract_customer_name(summary)
            show_context_hint()

            # Answer with fresh summary
            answer_question_with_retry(q, summary)
            continue

        # No identifier → use existing context
        if CTX.summary:
            show_context_hint()
            answer_question_with_retry(q, CTX.summary)
            continue

        # No context → generic answer
        print("ℹ️ No customer context. Answering generically...")
        ans = generic_answer(q)
        print(ans)

if __name__ == "__main__":
    main()
