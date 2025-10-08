# Fetches, cleans, and analyzes customer support emails from Gmail.
# Uses .env for credentials. Provides a searchable support conversation history and, when a customer email is entered, a Gemini-powered summary of the conversation. Includes locale-safe Sent mailbox selection and HTML-to-text cleanup for better summaries.

import os
import imaplib
import email
from email.header import decode_header
import pandas as pd
import re
import logging
from datetime import datetime, timedelta
from dateutil import parser
from typing import List, Tuple, Optional, Dict, Any
from email.message import Message
from dotenv import load_dotenv, find_dotenv
from html import unescape

load_dotenv(find_dotenv(filename=".env"))

EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

try:
    import google.generativeai as genai
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('email_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

IMAP_SERVER: str = "imap.gmail.com"
SENT_MAILBOX: str = "[Gmail]/Sent Mail"
SENT_MAILBOX_CANDIDATES = ("[Gmail]/Sent Mail", "[Gmail]/Sent Messages", "Sent", "Sent Items", "Sent Mail")
FETCH_LIMIT: int = 250
SUPPORT_DOMAIN: str = "epaylater.in"
PHONE_PATTERN: str = r'(\b[789]\d{9}\b)'
DEFAULT_DAYS: int = 7
MAX_BODY_LENGTH: int = 10000
MAX_TRANSCRIPT_CHARS: int = 18000


def select_mailbox_safe(mail, mailbox: str) -> str:
    """Selects an IMAP mailbox, trying alternate names for Sent folders, using readonly mode."""
    try:
        typ, _ = mail.select(mailbox, readonly=True)
        if typ == 'OK':
            return mailbox
    except Exception:
        pass
    if mailbox in SENT_MAILBOX_CANDIDATES:
        for alt in SENT_MAILBOX_CANDIDATES:
            try:
                typ, _ = mail.select(alt, readonly=True)
                if typ == 'OK':
                    return alt
            except Exception:
                continue
    raise imaplib.IMAP4.error(f"Failed to select mailbox: {mailbox}")


def _html_to_text(html: str) -> str:
    """Strips HTML tags and cleans up the resulting text."""
    text = re.sub(r"(?s)<(script|style).*?>.*?</\1>", " ", html)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(data: Any) -> str:
    """Safely decodes bytes to a string or returns an empty string for None."""
    if data is None:
        return ""
    if isinstance(data, bytes):
        try:
            return data.decode('utf-8')
        except UnicodeDecodeError:
            logger.warning("Failed to decode as UTF-8, falling back to latin-1")
            return data.decode('latin-1', errors='ignore')
    return str(data)


def decode_email_header(header_value: Optional[str]) -> str:
    """Decodes an email header, handling various character encodings."""
    if not header_value:
        return ""
    try:
        parts = decode_header(header_value)
        out = []
        for val, enc in parts:
            if isinstance(val, bytes):
                try:
                    out.append(val.decode(enc or 'utf-8', errors='ignore'))
                except Exception:
                    out.append(val.decode('utf-8', errors='ignore'))
            else:
                out.append(val or "")
        return "".join(out)
    except Exception as e:
        logger.warning(f"Failed to decode header '{header_value}': {e}")
        return ""


def extract_email_body(msg: Message) -> str:
    """Extracts the plain text body from an email message, falling back to HTML."""
    body = ""
    try:
        if msg.is_multipart():
            for part in msg.walk():
                ct = part.get_content_type()
                cd = part.get('Content-Disposition')
                if ct == "text/plain" and (not cd or "attachment" not in (cd or "").lower()):
                    payload = part.get_payload(decode=True)
                    if payload:
                        body = payload.decode('utf-8', errors='ignore')
                        break
            if not body:
                for part in msg.walk():
                    if part.get_content_type() == "text/html":
                        payload = part.get_payload(decode=True)
                        if payload:
                            html = payload.decode('utf-8', errors='ignore')
                            body = _html_to_text(html)
                            break
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                raw = payload.decode('utf-8', errors='ignore')
                if (msg.get_content_type() or "").lower() == "text/html":
                    body = _html_to_text(raw)
                else:
                    body = raw
    except Exception as e:
        logger.error(f"Error extracting email body: {e}")

    if len(body) > MAX_BODY_LENGTH:
        logger.warning(f"Truncating email body from {len(body)} to {MAX_BODY_LENGTH} characters")
        body = body[:MAX_BODY_LENGTH] + "... [TRUNCATED]"
    return body


def parse_email_date(date_string: Optional[str]) -> Optional[datetime]:
    """Parses a date string from an email into a datetime object."""
    if not date_string:
        return None
    try:
        return parser.parse(date_string)
    except (ValueError, TypeError, parser.ParserError) as e:
        logger.warning(f"Failed to parse date '{date_string}': {e}")
        return None


def fetch_emails_imap(email_address: str, app_password: str, mailbox: str, days_ago: int) -> pd.DataFrame:
    """Connects to Gmail via IMAP and fetches emails from a specific mailbox."""
    email_data: List[Dict[str, Any]] = []
    mail = None
    try:
        logger.info(f"Connecting to {IMAP_SERVER} for mailbox: {mailbox}")
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        try:
            mail.login(email_address, app_password)
            logger.info("Successfully logged in to Gmail")
        except imaplib.IMAP4.error as e:
            logger.error(f"Login failed: {e}")
            return pd.DataFrame()
        try:
            chosen_box = select_mailbox_safe(mail, mailbox)
            logger.info(f"Selected mailbox: {chosen_box}")
        except imaplib.IMAP4.error as e:
            logger.error(f"Failed to select mailbox {mailbox}: {e}")
            return pd.DataFrame()

        date_since = (datetime.now() - timedelta(days=days_ago)).strftime("%d-%b-%Y")
        logger.info(f"Fetching emails since: {date_since}")

        try:
            typ, message_ids = mail.search(None, 'SINCE', date_since)
            if typ != 'OK':
                logger.error("IMAP search failed")
                return pd.DataFrame()
            email_id_list = message_ids[0].split() if message_ids and message_ids[0] else []
        except imaplib.IMAP4.error as e:
            logger.error(f"Search failed: {e}")
            return pd.DataFrame()

        if not email_id_list:
            logger.info(f"No emails found in {mailbox} since {date_since}")
            return pd.DataFrame()

        emails_to_process = email_id_list[-FETCH_LIMIT:] if len(email_id_list) > FETCH_LIMIT else email_id_list
        logger.info(f"Processing {len(emails_to_process)} emails from {mailbox}")

        for i, email_id in enumerate(emails_to_process, start=1):
            try:
                if i % 50 == 0:
                    logger.info(f"Processing email {i}/{len(emails_to_process)}")
                _, msg_data = mail.fetch(email_id, '(RFC822)')
                for response_part in msg_data:
                    if isinstance(response_part, tuple) and len(response_part) > 1:
                        try:
                            msg = email.message_from_bytes(response_part[1])
                            email_record = {
                                'Sender': decode_email_header(msg.get("From")),
                                'To': decode_email_header(msg.get("To")),
                                'Cc': decode_email_header(msg.get("Cc")),
                                'Date': parse_email_date(msg.get("Date")),
                                'Subject': decode_email_header(msg.get("Subject")),
                                'Body': extract_email_body(msg)
                            }
                            email_data.append(email_record)
                        except Exception as e:
                            logger.warning(f"Failed to process email {email_id}: {e}")
                            continue
            except imaplib.IMAP4.error as e:
                logger.warning(f"Failed to fetch email {email_id}: {e}")
                continue
    except imaplib.IMAP4.error as e:
        logger.error(f"IMAP error while fetching {mailbox}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error while fetching {mailbox}: {e}")
    finally:
        if mail:
            try:
                mail.close()
                mail.logout()
                logger.info("Successfully closed IMAP connection")
            except Exception as e:
                logger.warning(f"Error closing IMAP connection: {e}")

    logger.info(f"Successfully fetched {len(email_data)} emails from {mailbox}")
    return pd.DataFrame(email_data)


def is_customer_email(email_address: str) -> bool:
    """Checks if an email address belongs to a customer, not the support domain."""
    if not email_address or not isinstance(email_address, str):
        return False
    email_lower = email_address.lower()
    return SUPPORT_DOMAIN not in email_lower and '@' in email_lower


def extract_customer_email_from_recipients(recipients_string: str) -> str:
    """Finds the first customer email address from a comma-separated list of recipients."""
    if not recipients_string:
        return ""
    recipients = [addr.strip() for addr in str(recipients_string).split(',')]
    for addr in recipients:
        if is_customer_email(addr):
            return addr
    return ""


def extract_name_from_email_field(email_string: str) -> str:
    """Extracts the display name from an email field (e.g., 'John Doe <j.doe@...>' -> 'John Doe')."""
    if not email_string or not isinstance(email_string, str):
        return ""
    name_match = re.match(r'^(.*?)\s*<', str(email_string))
    if name_match:
        name = name_match.group(1).strip()
        return name.strip('\'"')
    return ""


def extract_email_from_field(email_string: str) -> str:
    """Extracts the email address from an email field (e.g., 'John Doe <j.doe@...>' -> 'j.doe@...')."""
    if not email_string or not isinstance(email_string, str):
        return ""
    bracket_match = re.search(r'<(.*?)>', str(email_string))
    if bracket_match:
        return bracket_match.group(1).strip()
    return str(email_string).strip()


def process_and_clean_emails(inbound_df: pd.DataFrame, outbound_df: pd.DataFrame) -> pd.DataFrame:
    """Combines, cleans, and standardizes inbound and outbound email DataFrames."""
    logger.info("Processing and cleaning email data")

    if not inbound_df.empty:
        inbound_df = inbound_df.copy()
        customer_mask = ~inbound_df['Sender'].str.lower().str.contains(SUPPORT_DOMAIN, na=False)
        inbound_df = inbound_df[customer_mask].copy()
        inbound_df['Direction'] = 'inbound'
        inbound_df['ParsedCustomerEmail'] = inbound_df['Sender']
        logger.info(f"Processed {len(inbound_df)} inbound customer emails")

    if not outbound_df.empty:
        outbound_df = outbound_df.copy()

        def has_customer_recipient(row) -> bool:
            recipients = f"{row.get('To') or ''},{row.get('Cc') or ''}"
            return extract_customer_email_from_recipients(recipients) != ""

        customer_mask = outbound_df.apply(has_customer_recipient, axis=1)
        outbound_df = outbound_df[customer_mask].copy()
        outbound_df['Direction'] = 'outbound'

        def get_customer_email(row) -> str:
            recipients = f"{row.get('To') or ''},{row.get('Cc') or ''}"
            return extract_customer_email_from_recipients(recipients)

        outbound_df['ParsedCustomerEmail'] = outbound_df.apply(get_customer_email, axis=1)
        logger.info(f"Processed {len(outbound_df)} outbound customer emails")

    df_list = []
    if not inbound_df.empty:
        df_list.append(inbound_df)
    if not outbound_df.empty:
        df_list.append(outbound_df)

    if not df_list:
        logger.warning("No customer emails found to process")
        return pd.DataFrame()

    df = pd.concat(df_list, ignore_index=True)
    df = df.sort_values('Date', na_position='last').reset_index(drop=True)
    df['ParsedCustomerName'] = df['ParsedCustomerEmail'].apply(extract_name_from_email_field)
    df['ParsedCustomerEmail'] = df['ParsedCustomerEmail'].apply(extract_email_from_field)
    df['PhoneExtract'] = df['Body'].str.extract(PHONE_PATTERN, expand=False).fillna('')

    logger.info(f"Final processed dataset contains {len(df)} customer conversations")
    return df


def search_conversations(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Searches the DataFrame for conversations matching a query (name, email, or phone)."""
    if not query or df.empty:
        return pd.DataFrame()
    query = query.strip()
    if not query:
        return pd.DataFrame()

    logger.info(f"Searching conversations for: '{query}'")
    normalized_query = re.sub(r'\D', '', query)
    name_match = df['ParsedCustomerName'].str.contains(query, case=False, na=False)
    email_match = df['ParsedCustomerEmail'].str.contains(query, case=False, na=False)
    if normalized_query:
        phone_match = df['PhoneExtract'].str.contains(normalized_query, na=False)
    else:
        phone_match = pd.Series([False] * len(df), index=df.index)
    combined_match = name_match | email_match | phone_match
    results = df[combined_match]
    logger.info(f"Found {len(results)} matching conversations")
    return results


def validate_credentials() -> Tuple[bool, str]:
    """Checks that necessary email credentials exist in environment variables."""
    if not EMAIL:
        return False, "EMAIL not found in environment variables"
    if not PASSWORD:
        return False, "PASSWORD not found in environment variables"
    if '@' not in EMAIL:
        return False, "EMAIL format appears invalid"
    return True, ""


def build_transcript_for_email(df: pd.DataFrame, customer_email: str) -> Tuple[str, int, Optional[datetime], Optional[datetime]]:
    """Builds a chronological plain-text transcript for a single customer's conversation."""
    cemail = customer_email.strip().lower()
    rows = df[df['ParsedCustomerEmail'].str.lower() == cemail].copy()
    if rows.empty:
        return "", 0, None, None

    rows = rows.sort_values('Date', na_position='last')
    parts = []
    first_dt = None
    last_dt = None
    for _, r in rows.iterrows():
        dt = r.get('Date')
        if isinstance(dt, datetime):
            if not first_dt:
                first_dt = dt
            last_dt = dt
            dt_str = dt.isoformat(timespec='seconds')
        else:
            dt_str = "UnknownDate"
        direction = r.get('Direction', '')
        subject = r.get('Subject', '')
        body = r.get('Body', '')
        body = body.replace('\r', ' ').replace('\n', ' ').strip()
        if len(body) > 2000:
            body = body[:2000] + " ... [TRUNCATED]"
        parts.append(f"[{dt_str}] {direction.upper()} | Subject: {subject}\n{body}\n")
    transcript = "\n".join(parts)
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        transcript = transcript[-MAX_TRANSCRIPT_CHARS:]
    return transcript, len(rows), first_dt, last_dt


def summarize_with_gemini(transcript: str, customer_email: str) -> str:
    """Summarizes a conversation transcript using the Gemini API."""
    if not HAS_GENAI:
        return "Gemini Python SDK not installed. Install with: pip install google-generativeai"
    if not GEMINI_API_KEY:
        return "GEMINI_API_KEY is missing in .env. Add it to enable AI summaries."

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "You are a customer support analyst. Summarize the full conversation between a customer and a support agent.\n"
            "Write a crisp, factual summary ≤ 180 words. Include:\n"
            "- Customer's main issue(s)\n"
            "- Key actions taken by support\n"
            "- Current status and any pending items\n"
            "- If present, promised timelines or next steps\n"
            "Avoid quoting long text. Do not include any sensitive credentials. "
            f"Customer: {customer_email}\n\n"
            f"Transcript:\n{transcript}"
        )
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", None) or (resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else "")
        return text.strip() if text else "No summary generated."
    except Exception as e:
        logger.error(f"Gemini summarization failed: {e}")
        return f"Gemini summarization failed: {e}"


def get_days_input() -> int:
    """Prompts the user to enter the number of days of email history to fetch."""
    while True:
        try:
            user_input = input(f"Enter days of email history to fetch (default {DEFAULT_DAYS}): ").strip()
            if not user_input:
                return DEFAULT_DAYS
            days = int(user_input)
            if days <= 0:
                print("Please enter a positive number of days.")
                continue
            if days > 365:
                print("Warning: Fetching more than 365 days may take a very long time.")
                confirm = input("Continue? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue
            return days
        except ValueError:
            print("Please enter a valid number.")
        except (KeyboardInterrupt, EOFError):
            logger.info("User cancelled input")
            return DEFAULT_DAYS


def main() -> None:
    """Main function to run the email analyzer CLI application."""
    logger.info("Starting Customer Support Email Analyzer")

    is_valid, error_msg = validate_credentials()
    if not is_valid:
        logger.error(f"Credential validation failed: {error_msg}")
        print(f"Error: {error_msg}. Please create a .env file with your credentials.")
        return

    try:
        days_to_fetch = get_days_input()
    except (KeyboardInterrupt, EOFError):
        logger.info("Process cancelled by user")
        print("\nProcess cancelled.")
        return

    logger.info(f"Fetching emails from last {days_to_fetch} days")
    try:
        logger.info("Fetching inbound emails...")
        inbound_data = fetch_emails_imap(EMAIL, PASSWORD, mailbox='INBOX', days_ago=days_to_fetch)
        logger.info("Fetching outbound emails...")
        outbound_data = fetch_emails_imap(EMAIL, PASSWORD, mailbox=SENT_MAILBOX, days_ago=days_to_fetch)
    except Exception as e:
        logger.error(f"Failed to fetch emails: {e}")
        print(f"Error fetching emails: {e}")
        return

    try:
        all_conversations_df = process_and_clean_emails(inbound_data, outbound_data)
    except Exception as e:
        logger.error(f"Failed to process emails: {e}")
        print(f"Error processing emails: {e}")
        return

    if all_conversations_df.empty:
        print(f"\nNo relevant customer conversations found in the last {days_to_fetch} days.")
        logger.info("Customer Support Email Analyzer completed")
        return

    print(f"\n✅ Found and processed {len(all_conversations_df)} emails from the last {days_to_fetch} days.")
    preview_columns = ['Date', 'Direction', 'ParsedCustomerName', 'ParsedCustomerEmail', 'PhoneExtract', 'Subject']
    available_columns = [c for c in preview_columns if c in all_conversations_df.columns]
    try:
        preview_data = all_conversations_df[available_columns].head(10)
        print("\nPreview of conversations:")
        print(preview_data.to_string(index=False))
    except Exception as e:
        logger.warning(f"Failed to display preview: {e}")

    while True:
        try:
            search_input = input("\nEnter a customer EMAIL to summarize, or any text to search (or 'exit' to quit): ").strip()

            if not search_input or search_input.lower() == 'exit':
                logger.info("User finished searching.")
                break

            if '@' in search_input:
                transcript, count, first_dt, last_dt = build_transcript_for_email(all_conversations_df, search_input)
                if not transcript:
                    print(f"\nNo conversation found for {search_input}.")
                    continue
                print(f"\nBuilding AI summary for {search_input} (messages: {count}, range: {first_dt} → {last_dt}) ...")
                summary = summarize_with_gemini(transcript, search_input)
                print("\n===== Conversation Summary (Gemini) =====\n")
                print(summary)
                print("\n=========================================\n")
            else:
                results_df = search_conversations(all_conversations_df, search_input)
                if results_df.empty:
                    print(f"\nNo conversations found for '{search_input}'.")
                else:
                    print(f"\nFound {len(results_df)} matching conversations:")
                    display_columns = ['Date', 'Direction', 'ParsedCustomerName', 'ParsedCustomerEmail', 'Subject']
                    available_display_columns = [c for c in display_columns if c in results_df.columns]
                    try:
                        results_preview = results_df[available_display_columns].head(20)
                        print(results_preview.to_string(index=False))
                    except Exception as e:
                        logger.warning(f"Failed to display search results: {e}")
                        print("Search completed but failed to display results.")
        except (KeyboardInterrupt, EOFError):
            logger.info("Search cancelled by user")
            print("\nSearch cancelled.")
            break

    logger.info("Customer Support Email Analyzer completed")


if __name__ == "__main__":
    main()

