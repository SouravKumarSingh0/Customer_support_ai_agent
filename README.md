Slack Customer Discussion Summarizer

A Python-based agent that searches Slack discussions related to a customer (by Application ID, Phone Number, or Email) and generates a factual, concise summary using Google Gemini.

🚀 Features

Multi-identifier search: Find customer discussions using App ID, phone, or email.

Priority + deep search:

First searches only the channels listed in .env (SLACK_CHANNELS).

If nothing is found, prompts to search across all accessible channels.

Transcript builder: Fetches channel history and threaded replies, normalizes text, and builds a chronological transcript in IST.

AI summarization: Sends the transcript to Gemini (gemini-1.5-flash) and returns a professional, factual summary under 180 words.

Debug utilities:

list_all_channels() → list all channels the bot can access.

join_specific_channel.py → utility script to manually join a public channel by name.

📂 Project Structure
project-root/
│
├── Slack.py                  # Main summarizer script
├── join_specific_channel.py   # Utility to join a public channel
├── .env                       # Environment variables (not committed)
└── README.md                  # Project notes and usage

⚙️ Setup

Clone repo

git clone <your-gitlab-url>
cd project-root


Create virtual environment

python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows


Install requirements

pip install slack_sdk python-dotenv google-generativeai


Environment variables (.env)

SLACK_BOT_TOKEN=xoxb-...
GEMINI_API_KEY=your-gemini-api-key
SLACK_CHANNELS=sales-product-support


SLACK_BOT_TOKEN: Slack bot token with scopes:

channels:read, groups:read, channels:join, conversations.history

SLACK_CHANNELS: Comma-separated list of channel names or IDs to search first.

GEMINI_API_KEY: Google Generative AI API key.

▶️ Usage
Run the summarizer
python Slack.py


You’ll be prompted to enter:

Days of history to search

Application ID, Phone Number, or Email

If discussions are found, Gemini generates a crisp summary.

Join a channel (if bot not yet a member)
python join_specific_channel.py


Edit CHANNEL_TO_JOIN inside the script before running.

📌 Notes & Learnings (so far)

Channel membership matters: The bot must be a member of any channel to fetch history.

Public → can be joined with channels:join scope.

Private → must be invited manually.

Archived → cannot fetch history until unarchived.

System messages: Joining a public channel posts a “Bot has joined the channel” message visible to all.

Transcript limits: Long transcripts are truncated to the last ~18k characters to fit Gemini’s prompt size.

Error handling:

Handles Slack rate limits (429) with backoff.

Skips channels not found or inaccessible instead of crashing.

Current workflow:

Ensure bot is in relevant channels (use join_specific_channel.py).

Run Slack.py and provide App ID/Phone/Email.

Get a factual Gemini summary + transcript preview.