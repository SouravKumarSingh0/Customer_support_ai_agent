import os
import re
import time
import pickle
import logging
import asyncio
import threading # Import the threading module
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional, Set
from html import unescape
from dotenv import load_dotenv, find_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----- Environment / Constants -----
load_dotenv(find_dotenv(".env"))
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNELS = [c.strip() for c in (os.getenv("SLACK_CHANNELS") or "").split(",") if c.strip()]
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # <-- added
MAX_TRANSCRIPT_CHARS = 18000
RATE_LIMIT_RETRY_DELAY = 2
MAX_RETRIES = 3
PAGINATION_LIMIT = 1000  # Increased from 200 to 1000 (Slack's max)
CACHE_FILE = "slack_cache.pkl"
CACHE_TTL = 3600  # 1 hour cache validity
MAX_WORKERS = 5  # Parallel processing workers

# ----- Logging Setup -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("slack_agent")

# ----- Dependency Imports -----
try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    HAS_SLACK = True
except ImportError as e:
    logger.warning(f"Slack SDK not available: {e}")
    HAS_SLACK = False

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError as e:
    logger.warning(f"Google Generative AI not available: {e}")
    HAS_GENAI = False

# ----- Custom Exceptions -----
class SlackAgentError(Exception):
    pass

# ----- Message Cache Implementation -----
class MessageCache:
    def __init__(self, cache_file: str = CACHE_FILE):
        self.cache_file = cache_file
        self.lock = threading.Lock() # Add a lock for thread safety
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        with self.lock: # Acquire lock before file access
            try:
                if os.path.exists(self.cache_file):
                    with open(self.cache_file, 'rb') as f:
                        return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
            return {}

    def _save_cache(self):
        with self.lock: # Acquire lock before file access
            try:
                with open(self.cache_file, 'wb') as f:
                    # Create a copy to prevent "dictionary changed size during iteration"
                    cache_copy = dict(self.cache)
                    pickle.dump(cache_copy, f)
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

    def get_cached_messages(self, channel_id: str, oldest_ts: float) -> Optional[List[Dict]]:
        key = f"{channel_id}_{oldest_ts}"
        # No lock needed for simple dict read if writes are locked
        if key in self.cache:
            cached_data, timestamp = self.cache[key]
            if time.time() - timestamp < CACHE_TTL:
                logger.debug(f"Cache hit for channel {channel_id}")
                return cached_data
        return None

    def cache_messages(self, channel_id: str, oldest_ts: float, messages: List[Dict]):
        key = f"{channel_id}_{oldest_ts}"
        with self.lock: # Lock before modifying the dictionary
            self.cache[key] = (messages, time.time())
        # The save operation is now also thread-safe
        self._save_cache()

message_cache = MessageCache()

# ----- Core Client and Validators -----
def validate_env() -> Tuple[bool, str]:
    # Validates that required environment variables and dependencies are available.
    if not HAS_SLACK:
        return False, "slack_sdk not installed. Run: pip install slack_sdk"
    if not SLACK_BOT_TOKEN:
        return False, "SLACK_BOT_TOKEN environment variable is missing"
    if len(SLACK_BOT_TOKEN) < 10:
        return False, "SLACK_BOT_TOKEN appears to be invalid"
    return True, ""

def make_client() -> "WebClient":
    # Creates and returns an authenticated Slack WebClient instance.
    return WebClient(token=SLACK_BOT_TOKEN)

# ----- Helper Functions -----
def _is_channel_id(s: str) -> bool:
    # Checks if a string matches the Slack channel ID format (e.g., C12345, G12345).
    return bool(re.fullmatch(r"[CG][A-Z0-9]+", s))

def _slack_error_code(e: SlackApiError) -> str:
    # Safely extracts the specific error string (e.g., 'not_in_channel') from a SlackApiError.
    try:
        if hasattr(e, "response"):
            if hasattr(e.response, "data") and isinstance(e.response.data, dict):
                return e.response.data.get("error") or ""
            try:
                return e.response.get("error", "")
            except Exception:
                pass
    except Exception:
        pass
    return ""

def _handle_rate_limit(func):
    # Decorator to automatically retry Slack API calls on rate limits with exponential backoff.
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except SlackApiError as e:
                if getattr(e, "response", None) and getattr(e.response, "status_code", None) == 429:
                    retry_after = int(e.response.headers.get("Retry-After", RATE_LIMIT_RETRY_DELAY))
                    delay = retry_after * (2 ** attempt)
                    logger.warning(f"Rate limited, retrying in {delay}s (attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(delay)
                    continue
                raise
        raise SlackAgentError("Max retries exceeded for rate-limited request")
    return wrapper

# ----- Rate-Limited Slack API Wrappers -----
@_handle_rate_limit
def _conversations_list(client: WebClient, **kwargs):
    return client.conversations_list(**kwargs)

@_handle_rate_limit
def _conversations_history(client: WebClient, **kwargs):
    return client.conversations_history(**kwargs)

@_handle_rate_limit
def _conversations_replies(client: WebClient, **kwargs):
    return client.conversations_replies(**kwargs)

@_handle_rate_limit
def _users_info(client: WebClient, **kwargs):
    return client.users_info(**kwargs)

# ----- Channel and Message Fetching -----
def list_channels(client: WebClient, types: str = "public_channel,private_channel") -> List[Dict]:
    # Retrieves a complete list of channels, handling pagination automatically.
    channels: List[Dict] = []
    cursor = None
    while True:
        try:
            resp = _conversations_list(
                client,
                types=types,
                exclude_archived=True,
                limit=PAGINATION_LIMIT,
                cursor=cursor
            )
        except SlackApiError as e:
            logger.error(f"Error fetching channels: {_slack_error_code(e) or e}")
            raise

        channels.extend(resp.get("channels", []))
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break

    logger.info(f"Retrieved {len(channels)} channels")
    return channels

def resolve_channel_ids_from_cache(channels: List[Dict], requested: List[str]) -> List[str]:
    # Efficiently resolves channel names to IDs using a pre-fetched list of channels.
    name_to_id = {c.get("name", ""): c.get("id", "") for c in channels}
    out: List[str] = []
    for item in requested or []:
        item = item.strip()
        if not item:
            continue
        if _is_channel_id(item):
            out.append(item)
        else:
            cid = name_to_id.get(item)
            if cid:
                out.append(cid)
                logger.info(f"Resolved channel '{item}' to ID: {cid}")
            else:
                logger.warning(f"Channel not found or inaccessible: '{item}'")
    return out

def get_active_channels_only(client: WebClient, channel_ids: List[str], days: int = 30) -> List[str]:
    """Only search channels that had recent activity"""
    recent_ts = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
    active_channels = []
    
    for cid in channel_ids:
        try:
            resp = _conversations_history(client, channel=cid, limit=1)
            if resp.get("messages") and float(resp["messages"][0]["ts"]) > recent_ts:
                active_channels.append(cid)
        except SlackApiError:
            continue
    
    logger.info(f"Filtered to {len(active_channels)} active channels from {len(channel_ids)} total")
    return active_channels

def fetch_history_optimized(client: WebClient, channel_id: str, oldest_ts: float, compiled_patterns: List[re.Pattern]) -> List[Dict]:
    """Optimized version with early termination and pre-filtering"""
    # Check cache first
    cached_messages = message_cache.get_cached_messages(channel_id, oldest_ts)
    if cached_messages is not None:
        # Still need to filter cached messages
        return [msg for msg in cached_messages 
                if fast_text_matches(normalize_text(msg.get("text") or ""), compiled_patterns)]

    messages = []
    cursor = None
    all_messages = []  # Keep track of all messages for caching
    
    while True:
        try:
            resp = _conversations_history(
                client,
                channel=channel_id,
                oldest=str(oldest_ts),
                limit=PAGINATION_LIMIT,
                cursor=cursor
            )
        except SlackApiError as e:
            code = _slack_error_code(e)
            if code in {"not_in_channel", "channel_not_found", "is_archived", "access_denied"}:
                logger.warning(f"Skipping channel {channel_id}: {code}")
                return []
            logger.error(f"Unexpected error fetching history for {channel_id}: {code or e}")
            return []

        batch = resp.get("messages", [])
        all_messages.extend(batch)  # Store all messages for caching
        
        # Pre-filter messages before processing threads
        relevant_messages = [
            msg for msg in batch
            if fast_text_matches(normalize_text(msg.get("text") or ""), compiled_patterns)
        ]
        
        messages.extend(relevant_messages)
        
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    
    # Cache all messages (not just filtered ones) for future use
    message_cache.cache_messages(channel_id, oldest_ts, all_messages)
    return messages

def fetch_replies(client: WebClient, channel_id: str, thread_ts: str) -> List[Dict]:
    # Fetches all replies for a specific message thread.
    replies: List[Dict] = []
    cursor = None
    while True:
        try:
            resp = _conversations_replies(
                client,
                channel=channel_id,
                ts=thread_ts,
                limit=PAGINATION_LIMIT,
                cursor=cursor
            )
        except SlackApiError as e:
            logger.warning(f"Could not fetch replies for thread {thread_ts}: {_slack_error_code(e) or e}")
            return []

        replies.extend(resp.get("messages", []))
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return replies

def fetch_and_filter_channel(client: WebClient, channel_id: str, oldest_ts: float, 
                             identifiers: Dict[str, str], compiled_patterns: List[re.Pattern], 
                             channel_map: Dict[str, str]) -> List[Tuple[datetime, str, str, str]]:
    """Process a single channel and return relevant messages"""
    try:
        history = fetch_history_optimized(client, channel_id, oldest_ts, compiled_patterns)
        if not history:
            return []

        channel_lines = []
        parents = [m for m in history if (not m.get("thread_ts")) or (m.get("ts") == m.get("thread_ts"))]

        for m in parents:
            texts = []
            ptxt = normalize_text(m.get("text") or "")
            if ptxt:
                texts.append(("parent", m, ptxt))

            # Only fetch threads for messages that already match
            if fast_text_matches(ptxt, compiled_patterns) and m.get("reply_count", 0) > 0 and m.get("thread_ts"):
                replies = fetch_replies(client, channel_id, m["thread_ts"])
                for r in replies:
                    rtxt = normalize_text(r.get("text") or "")
                    if rtxt:
                        texts.append(("reply", r, rtxt))

            if not any(fast_text_matches(t, compiled_patterns) for _, _, t in texts):
                continue

            for _, msg, t in texts:
                ts = float(msg.get("ts", "0"))
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                ist = dt.astimezone(timezone(timedelta(hours=5, minutes=30)))

                uid = msg.get("user") or "unknown"
                cname = channel_map.get(channel_id, channel_id)
                channel_lines.append((ist, cname, uid, t))

        return channel_lines
    except Exception as e:
        logger.error(f"Error processing channel {channel_id}: {e}")
        return []

def fetch_all_channels_parallel(client: WebClient, channel_ids: List[str], oldest_ts: float, 
                                  identifiers: Dict[str, str], compiled_patterns: List[re.Pattern], 
                                  channel_map: Dict[str, str]) -> List[Tuple[datetime, str, str, str]]:
    """Process channels in parallel"""
    all_lines = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all channel processing tasks
        future_to_channel = {
            executor.submit(fetch_and_filter_channel, client, cid, oldest_ts, 
                            identifiers, compiled_patterns, channel_map): cid 
            for cid in channel_ids
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_channel):
            channel_id = future_to_channel[future]
            try:
                result = future.result(timeout=300)  # 5 minute timeout per channel
                all_lines.extend(result)
                if result:
                    logger.info(f"Channel {channel_id}: found {len(result)} relevant messages")
            except Exception as e:
                logger.error(f"Channel {channel_id} processing failed: {e}")
    
    return all_lines

# ----- Text Normalization and Matching -----
def normalize_text(text: str) -> str:
    # Cleans Slack's special formatting (links, mentions, etc.) from a message string.
    if not text:
        return ""
    t = unescape(text)
    t = re.sub(r"<http[^|>]+?\|([^>]+)>", r"\1", t)
    t = re.sub(r"<mailto:([^|>]+)\|[^>]+>", r"\1", t)
    t = re.sub(r"<#[A-Z0-9]+?\|([^>]+)>", r"#\1", t)
    t = re.sub(r"<@[A-Z0-9]+>", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n+", " ", t)
    return t.strip()

def normalize_identifiers(email: Optional[str], phone: Optional[str], app_id: Optional[str]) -> Dict[str, str]:
    # Standardizes customer identifiers (email, phone, app_id) for reliable matching.
    normalized = {
        "email": (email or "").strip().lower(),
        "phone_digits": re.sub(r"\D", "", phone or ""),
        "app_id": (app_id or "").strip().lower(),
    }
    provided = [k for k, v in normalized.items() if v]
    logger.info(f"Normalized identifiers provided: {provided}")
    return normalized

def create_search_patterns(identifiers: Dict[str, str]) -> List[re.Pattern]:
    """Pre-compile regex patterns for faster matching"""
    patterns = []
    if identifiers.get("email"):
        patterns.append(re.compile(re.escape(identifiers["email"]), re.IGNORECASE))
    if identifiers.get("app_id"):
        app_id = identifiers["app_id"]
        patterns.append(re.compile(rf"\bapp(?:lication)?\s*id(?:entifier)?\b[:=]?\s*{re.escape(app_id)}\b", re.IGNORECASE))
        patterns.append(re.compile(rf"\bappid\b[:=]?\s*{re.escape(app_id)}\b", re.IGNORECASE))
        patterns.append(re.compile(re.escape(app_id), re.IGNORECASE))
    if identifiers.get("phone_digits") and len(identifiers["phone_digits"]) >= 7:
        patterns.append(re.compile(identifiers["phone_digits"]))
    return patterns

def fast_text_matches(text: str, compiled_patterns: List[re.Pattern]) -> bool:
    """Fast text matching using pre-compiled patterns"""
    if not text or not compiled_patterns:
        return False
    
    # For phone numbers, we need to extract digits first
    text_lower = text.lower()
    text_digits = ""
    
    for pattern in compiled_patterns:
        if pattern.search(text_lower):
            return True
        # Only compute digits if a pattern seems to need it
        if not text_digits and any(char.isdigit() for char in pattern.pattern):
             text_digits = re.sub(r"\D", "", text)
        if text_digits and pattern.search(text_digits):
            return True

    return False

# ----- User Name Resolution -----
class UserCache:
    # Caches user ID-to-name mappings to reduce API calls, including failed lookups.
    def __init__(self):
        self._cache: Dict[str, str] = {}
        self._failed: Set[str] = set()

    def get(self, uid: str) -> Optional[str]:
        return self._cache.get(uid)

    def set(self, uid: str, name: str):
        self._cache[uid] = name

    def mark_failed(self, uid: str):
        self._failed.add(uid)

    def is_failed(self, uid: str) -> bool:
        return uid in self._failed

user_cache = UserCache()

def get_user_name(client: WebClient, user_id: str) -> str:
    # Resolves a Slack user ID to their display name, using a cache to improve performance.
    if not user_id or user_id == "unknown":
        return "unknown"

    cached = user_cache.get(user_id)
    if cached:
        return cached
    if user_cache.is_failed(user_id):
        return user_id

    try:
        resp = _users_info(client, user=user_id)
        user_data = resp.get("user", {}) if isinstance(resp, dict) else resp.data.get("user", {})
        name = user_data.get("real_name") or user_data.get("name") or user_id
        user_cache.set(user_id, name)
        return name
    except SlackApiError as e:
        logger.debug(f"Could not resolve user {user_id}: {_slack_error_code(e) or e}")
        user_cache.mark_failed(user_id)
        return user_id

def build_user_map(client: WebClient, user_ids: List[str]) -> Dict[str, str]:
    # Builds a dictionary mapping a list of unique user IDs to their real names.
    unique_ids = set(uid for uid in user_ids if uid and uid != "unknown")
    logger.info(f"Resolving {len(unique_ids)} unique user IDs")
    
    # Batch process user lookups for efficiency
    name_map = {}
    with ThreadPoolExecutor(max_workers=3) as executor:  # Lower concurrency for user lookups
        future_to_uid = {executor.submit(get_user_name, client, uid): uid for uid in unique_ids}
        for future in as_completed(future_to_uid):
            uid = future_to_uid[future]
            try:
                name_map[uid] = future.result()
            except Exception as e:
                logger.debug(f"Failed to resolve user {uid}: {e}")
                name_map[uid] = uid
    
    return name_map

# ----- Progressive Search Implementation -----
def progressive_search(client: WebClient, channel_ids: List[str], identifiers: Dict[str, str], 
                       channel_map: Dict[str, str]) -> Tuple[str, int, Optional[datetime], Optional[datetime]]:
    """Implement progressive search starting with recent data"""
    time_windows = [30, 90, 180, 365, 730]  # days
    compiled_patterns = create_search_patterns(identifiers)
    
    for days in time_windows:
        logger.info(f"Searching last {days} days...")
        oldest = datetime.now(timezone.utc) - timedelta(days=days)
        oldest_ts = oldest.timestamp()
        
        # Filter to active channels for recent searches
        if days <= 90:
            active_channel_ids = get_active_channels_only(client, channel_ids, days=days)
            if not active_channel_ids:
                continue
            search_channels = active_channel_ids
        else:
            search_channels = channel_ids
        
        all_lines = fetch_all_channels_parallel(
            client, search_channels, oldest_ts, identifiers, compiled_patterns, channel_map
        )
        
        if all_lines:
            logger.info(f"Found {len(all_lines)} messages in {days}-day window")
            return build_transcript_from_lines(client, all_lines)
    
    return "", 0, None, None

def build_transcript_from_lines(client: WebClient, all_lines: List[Tuple[datetime, str, str, str]]) -> Tuple[str, int, Optional[datetime], Optional[datetime]]:
    """Build transcript from collected message lines"""
    if not all_lines:
        return "", 0, None, None
    
    user_ids = [uid for _, _, uid, _ in all_lines]
    first_dt = min(dt for dt, _, _, _ in all_lines)
    last_dt = max(dt for dt, _, _, _ in all_lines)
    total_msgs = len(all_lines)
    
    name_map = build_user_map(client, user_ids)
    lines: List[str] = []
    for ist, cname, uid, text in sorted(all_lines, key=lambda x: x[0]):
        who = name_map.get(uid, uid or "unknown")
        lines.append(f"[{ist.isoformat(timespec='seconds')}] #{cname} | {who}: {text}")

    transcript = "\n".join(lines)
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        logger.warning(f"Transcript truncated from {len(transcript)} to {MAX_TRANSCRIPT_CHARS} characters")
        transcript = transcript[-MAX_TRANSCRIPT_CHARS:]

    logger.info(f"Generated transcript: {total_msgs} messages, {len(transcript)} chars")
    return transcript, total_msgs, first_dt, last_dt

# ----- Transcript Generation -----
def collect_customer_discussion(
    client: WebClient,
    channel_ids: List[str],
    email: Optional[str],
    phone: Optional[str],
    app_id: Optional[str],
    channel_map: Optional[Dict[str, str]] = None,
) -> Tuple[str, int, Optional[datetime], Optional[datetime]]:
    # Gathers all relevant messages for a customer and builds a chronological transcript.
    identifiers = normalize_identifiers(email, phone, app_id)
    
    if channel_map is None:
        channels = list_channels(client, types="public_channel,private_channel")
        channel_map = {c["id"]: c.get("name", c["id"]) for c in channels}

    logger.info(f"Starting progressive search in {len(channel_ids)} channels")
    
    # Use progressive search for better performance
    return progressive_search(client, channel_ids, identifiers, channel_map)

# ----- Utility and Summarization -----
def list_all_channels() -> List[Dict[str, str]]:
    # A utility function to list all channels the bot can see.
    is_valid, err = validate_env()
    if not is_valid:
        raise SlackAgentError(err)

    client = make_client()
    channels = list_channels(client, types="public_channel,private_channel")
    out: List[Dict[str, str]] = []
    for c in channels:
        info = {
            "name": c.get("name", ""),
            "id": c.get("id", ""),
            "private": bool(c.get("is_private", False)),
            "member": bool(c.get("is_member", False)),
            "archived": bool(c.get("is_archived", False))
        }
        out.append(info)
        print(f"Name: {info['name']} | ID: {info['id']} | Private: {info['private']} | Member: {info['member']} | Archived: {info['archived']}")
    return out

def summarize_with_gemini(transcript: str, identifiers: Dict[str, str]) -> str:
    # Sends the final transcript to Gemini for an AI-powered summary.
    if not HAS_GENAI:
        return "❌ Gemini is unavailable. Install with: pip install google-generativeai"
    if not GEMINI_API_KEY:
        return "❌ GEMINI_API_KEY environment variable is not set"

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)  # <-- updated

        id_info = ", ".join([f"{k}: {v}" for k, v in identifiers.items() if v])

        prompt = (
            "You are a customer support analyst. Summarize the following Slack discussion about a customer.\n"
            "Write a crisp, factual summary under 180 words. Focus on:\n"
            "- The core issue(s) the customer faced\n"
            "- Key actions taken by the team and who was involved\n"
            "- The final resolution or current status and next steps\n"
            "Do not quote messages directly. Provide a professional, narrative summary.\n\n"
            f"Customer Identifiers: {id_info}\n\n"
            f"Transcript:\n{transcript}"
        )

        resp = model.generate_content(prompt)
        text = getattr(resp, "text", None)
        if not text and getattr(resp, "candidates", None):
            text = resp.candidates[0].content.parts[0].text
        return text.strip() if text else "❌ No summary was generated by Gemini"

    except Exception as e:
        logger.error(f"Gemini summarization failed: {e}")
        return f"❌ Error during Gemini summarization: {str(e)}"

# ----- Main Orchestrator and CLI -----
def run_slack_summary(email: Optional[str], phone: Optional[str], app_id: Optional[str]):
    # Orchestrates the entire process: search, collection, and summarization.
    is_valid, err = validate_env()
    if not is_valid:
        print(f"❌ Error: {err}")
        return

    client = make_client()

    all_channels = list_channels(client, types="public_channel,private_channel")
    channel_map = {c["id"]: c.get("name", c["id"]) for c in all_channels}

    print(f"🔍 Starting Tier 1 search in priority channels: {SLACK_CHANNELS}")
    priority_channel_ids = resolve_channel_ids_from_cache(all_channels, SLACK_CHANNELS)

    transcript, count, first, last = collect_customer_discussion(
        client, priority_channel_ids, email=email, phone=phone, app_id=app_id, channel_map=channel_map
    )

    if not transcript:
        print("\n❌ No relevant discussion found in priority channels.")
        choice = input("🔍 Perform deep search across all channels the bot is a member of? [y/N]: ").strip().lower()
        if choice == "y":
            member_channel_ids = [c["id"] for c in all_channels if c.get("is_member") and not c.get("is_archived")]
            print(f"📊 Searching {len(member_channel_ids)} member-only channels...")
            transcript, count, first, last = collect_customer_discussion(
                client, member_channel_ids, email=email, phone=phone, app_id=app_id, channel_map=channel_map
            )

    if not transcript:
        print("\n❌ No relevant Slack discussion found for the given identifiers.")
        return

    print(f"\n✅ Found {count} matching messages from {first.date()} to {last.date()}")
    print("🤖 Generating AI summary...")
    ids_for_prompt = {"App ID": app_id, "Phone": phone, "Email": email}
    summary = summarize_with_gemini(transcript, ids_for_prompt)

    print("\n" + "=" * 25 + " 🤖 AI Summary " + "=" * 25)
    print(summary)
    print("=" * 70 + "\n")

    print("--- 📝 Transcript Preview (First 5 Messages) ---")
    lines = transcript.splitlines()
    for line in lines[:5]:
        print(line)
    if len(lines) > 5:
        print(f"... and {len(lines) - 5} more messages")
    print("-" * 50)

def main():
    # Handles the command-line interface and user input for the agent.
    try:
        print("🤖 Slack Customer Discussion Summarizer")
        print("-" * 40)
        
        print("\n🔍 Enter at least one customer identifier:")
        app_id = input("  📱 Application ID: ").strip() or None
        phone = input("  📞 Phone Number: ").strip() or None
        email = input("  📧 Email Address: ").strip() or None

        if not any([app_id, phone, email]):
            print("\n❌ Please provide at least one identifier (App ID, Phone, or Email)")
            return

        print(f"\n🚀 Starting progressive search (up to 2 years)...")
        run_slack_summary(email=email, phone=phone, app_id=app_id)

    except KeyboardInterrupt:
        print("\n\n🛑 Process cancelled by user. Exiting.")
    except EOFError:
        print("\n\n🛑 Input terminated. Exiting.")
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        print(f"\n💥 An unexpected error occurred: {e}")
        print("📋 Check the logs for detailed error information.")

        # add this near the bottom, above the main-guard
def summary_customer_conversation(app_id=None, phone=None, email=None):
    # thin alias; no behavior change
    return run_slack_summary(email=email, phone=phone, app_id=app_id)

# === BEGIN NON-DESTRUCTIVE NORMALIZATION LAYER (append-only) ===============
from typing import Any, Dict, List, Tuple, Set
from datetime import datetime, timezone
import logging

LOG = logging.getLogger("slack_agent.normalize")

def _to_iso_from_slack_ts(ts_raw: str) -> str:
    if not ts_raw:
        return ""
    try:
        # Slack ts like "1726927800.12345"
        epoch = float(str(ts_raw).split(".")[0])
        return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()
    except Exception:
        return ""

def _normalize_one(msg: Dict[str, Any], channel_name: str = None, user_display: str = None) -> Dict[str, Any]:
    """
    Convert a single Slack message dict into the canonical event shape.
    We are defensive: accept multiple field names your code may already produce.
    """
    text = msg.get("text") or msg.get("message") or msg.get("body") or ""
    ts_raw = msg.get("ts") or msg.get("timestamp") or msg.get("time") or ""
    ch = channel_name or msg.get("channel_name") or msg.get("channel") or msg.get("channel_id") or ""
    user = user_display or msg.get("user_display") or msg.get("user_name") or msg.get("user") or "user"

    return {
        "ts": _to_iso_from_slack_ts(str(ts_raw)) or str(ts_raw),
        "ts_raw": str(ts_raw),
        "channel_name": ch,
        "user_display": user,
        "text": str(text),
        "thread_ts": msg.get("thread_ts"),
        "permalink": msg.get("permalink"),
        "source": "slack",
    }

def _infer_channel_name(msg: Dict[str, Any], fallback: str = "") -> str:
    return msg.get("channel_name") or msg.get("channel") or fallback

def _infer_user_display(msg: Dict[str, Any]) -> str:
    return msg.get("user_display") or msg.get("user_name") or msg.get("username") or msg.get("user") or "user"

def _normalize_events_from_container(obj: Any) -> List[Dict[str, Any]]:
    """
    Accepts many shapes and emits a list of normalized events.
    Supported inputs:
      - list[dict] of slack messages
      - {"events": [...]}
      - {"messages": [...]}
      - {"results": {"messages": [...]}}    # some codebases wrap results
      - {"data": {"messages": [...]}}
      - {"transcript": {"messages": [...]}} # if you stored both transcript & messages
    """
    # Direct list of msgs?
    if isinstance(obj, list):
        out: List[Dict[str, Any]] = []
        for m in obj:
            if isinstance(m, dict) and ("text" in m or "ts" in m):
                out.append(_normalize_one(m, _infer_channel_name(m), _infer_user_display(m)))
        return out

    if not isinstance(obj, dict):
        return []

    # Common dict shapes
    if "events" in obj and isinstance(obj["events"], list):
        # Already normalized by your own code
        events = []
        for m in obj["events"]:
            if isinstance(m, dict) and ("text" in m or "ts" in m):
                events.append(_normalize_one(m, _infer_channel_name(m), _infer_user_display(m)))
        return events

    for key in ("messages", "msgs", "items"):
        if key in obj and isinstance(obj[key], list):
            events = []
            for m in obj[key]:
                if isinstance(m, dict) and ("text" in m or "ts" in m):
                    events.append(_normalize_one(m, _infer_channel_name(m), _infer_user_display(m)))
            return events

    # Nested
    for outer in ("results", "data", "transcript", "payload"):
        inner = obj.get(outer)
        if isinstance(inner, dict):
            ev = _normalize_events_from_container(inner)
            if ev:
                return ev

    return []

def _safe_meta_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    for k, v in (extra or {}).items():
        if k not in out:
            out[k] = v
    return out

def _build_meta(raw: Dict[str, Any], events: List[Dict[str, Any]], lookback_days: int, tier1_channels: List[str]) -> Dict[str, Any]:
    # Try to preserve any existing meta from your module
    meta_candidates = []
    if isinstance(raw, dict):
        for k in ("meta", "metadata", "summary", "info"):
            if isinstance(raw.get(k), dict):
                meta_candidates.append(raw.get(k))
    meta = {}
    for m in meta_candidates:
        meta = _safe_meta_merge(meta, m)
    # Add normalized fields
    meta = _safe_meta_merge(meta, {
        "count": len(events),
        "lookback_days": lookback_days,
        "channels_considered": tier1_channels,
    })
    # Optional: collect channel hits
    channels_hit = sorted({e.get("channel_name","") for e in events if e.get("channel_name")})
    meta["channels_hit"] = channels_hit
    # Optional: time window if present in your raw meta
    return meta

def _call_existing_search(identifiers: Dict[str, Any], days: int, tier1_channels: List[str]):
    """
    Retained for backward-compatibility, but the adapter `search(...)`
    below uses `collect_customer_discussion(...)` directly to avoid
    signature mismatches. This function is not used in the adapter.
    """
    raise RuntimeError("Adapter now routes via collect_customer_discussion(); _call_existing_search is unused.")

def search(identifiers: Dict[str, Any], days: int = 30, tier1_channels: List[str] = None) -> Dict[str, Any]:
    """
    ADAPTER LAYER (safe): calls your existing collection pipeline and ALWAYS returns:
      { "events": [ ...normalized... ], "meta": { ... } }

    It:
      - Creates a Slack client
      - Resolves the Tier-1 channels (from `tier1_channels` arg or SLACK_CHANNELS env)
      - Calls collect_customer_discussion(...)
      - Parses the transcript lines back into normalized events
    """
    tier1_channels = tier1_channels or SLACK_CHANNELS or []
    is_valid, err = validate_env()
    if not is_valid:
        return {"events": [], "meta": {"error": err, "lookback_days": days, "channels_considered": tier1_channels}}

    client = make_client()

    # Get channels + resolve IDs for requested tier-1 list
    channels = list_channels(client, types="public_channel,private_channel")
    channel_map = {c["id"]: c.get("name", c["id"]) for c in channels}

    # Allow caller to pass channel names; resolve to IDs
    tier1_ids = resolve_channel_ids_from_cache(channels, tier1_channels)

    # Pull transcript using your established pipeline
    email = identifiers.get("email")
    phone = identifiers.get("phone_digits") or identifiers.get("phone")
    app_id = identifiers.get("app_id")
    transcript, count, first_dt, last_dt = collect_customer_discussion(
        client, tier1_ids, email=email, phone=phone, app_id=app_id, channel_map=channel_map
    )

    # If nothing in tier-1, do NOT auto-deep-search here (leave that to CLI flow)
    if not transcript:
        meta = {
            "count": 0,
            "lookback_days": days,
            "channels_considered": tier1_channels,
            "channels_hit": [],
        }
        # Provide window if available
        if first_dt and last_dt:
            meta["window"] = {"oldest_iso": first_dt.isoformat(), "latest_iso": last_dt.isoformat()}
        return {"events": [], "meta": meta}

    # Parse transcript lines back into normalized events
    # Format your builder produced:
    #   "[2025-09-22T18:50:38+05:30] #operations-archived | Kajal Sharma: text..."
    line_re = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+#(?P<chan>[^|]+)\s+\|\s+(?P<user>[^:]+):\s*(?P<text>.*)$")
    events: List[Dict[str, Any]] = []
    for line in transcript.splitlines():
        m = line_re.match(line)
        if not m:
            continue
        ts_iso = m.group("ts").strip()
        chan = m.group("chan").strip()
        user = m.group("user").strip()
        text = m.group("text").strip()
        events.append({
            "ts": ts_iso,
            "ts_raw": ts_iso,
            "channel_name": chan,
            "user_display": user,
            "text": text,
            "thread_ts": None,
            "permalink": None,
            "source": "slack",
        })

    # Build meta
    meta = {
        "count": len(events),
        "lookback_days": days,
        "channels_considered": tier1_channels,
        "channels_hit": sorted({e["channel_name"] for e in events}),
    }
    if first_dt and last_dt:
        meta["window"] = {"oldest_iso": first_dt.isoformat(), "latest_iso": last_dt.isoformat()}

    LOG.info("Generated events: %d", len(events))
    return {"events": events, "meta": meta}

# === END NON-DESTRUCTIVE NORMALIZATION LAYER ================================

if __name__ == "__main__":
    # Main execution block that handles the --list-channels utility flag.
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--list-channels":
        try:
            chans = list_all_channels()
            print(f"\n📊 Total accessible channels: {len(chans)}")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        main()
