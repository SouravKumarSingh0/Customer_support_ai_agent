# 🤖 Customer Support Agent (CSA)

> An AI-powered orchestration tool that unifies customer data from multiple sources — Slack, Tele-Calling Databricks, Partner Dashboard, Freshdesk, and Onboarding — into clean summaries, CSV reports, and concise Gemini-based narratives.

---

## 🚀 Installation Guide (Step-by-Step)

> 🕐 Estimated setup time: ~5 minutes  
> Works on macOS or Linux (Intel or Apple Silicon)

---

### 1️⃣ Prerequisites

Make sure these are installed:

```bash
python3 --version   # Python 3.11 or later
pip3 --version
git --version
```

You’ll also need access to:
- Databricks SQL Warehouse  
- Google Gemini API key  
- Slack bot token (optional)  
- Internet connection  

---

### 2️⃣ Clone the repository

```bash
git clone <your-gitlab-repo-url> support-automation
cd support-automation
```

---

### 3️⃣ Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
```

Check:
```bash
python -V
pip -V
```

---

### 4️⃣ Install dependencies

If you already have a `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Otherwise, install manually:
```bash
pip install --upgrade pip
pip install python-dotenv databricks-sql-connector pandas google-generativeai slack_sdk rich tenacity
```

---

### 5️⃣ Create a `.env` file

Create a new file named `.env` in your project root and paste:

```bash
# --- Gemini (required) ---
GEMINI_API_KEY=your_google_generative_ai_key
GEMINI_MODEL=gemini-2.5-flash

# --- Slack (optional) ---
SLACK_BOT_TOKEN=xoxb-...
SLACK_CHANNELS=support,critical-incidents

# --- Databricks (required) ---
DATABRICKS_SERVER_HOSTNAME=adb-xxxxxxxx.azuredatabricks.net
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/xxxxxxxxxxxxxxxx
DATABRICKS_TOKEN=pat-xxxxxxxxxxxxxxxx

# --- Email (optional) ---
GMAIL_ENABLED=false

# --- General ---
LOG_LEVEL=INFO
TZ=Asia/Kolkata
```

⚠️ **Do not commit this file.** It contains credentials.

---

### 6️⃣ Test your setup

#### ✅ Databricks connection
```bash
python - <<'PY'
from databricks import sql
import os
with sql.connect(
    server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
    http_path=os.getenv("DATABRICKS_HTTP_PATH"),
    access_token=os.getenv("DATABRICKS_TOKEN")
) as c:
    with c.cursor() as cur:
        cur.execute("SELECT 1")
        print("Databricks OK:", cur.fetchone())
PY
```

#### ✅ Gemini
```bash
python - <<'PY'
import os, google.generativeai as genai
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
m = os.getenv("GEMINI_MODEL","gemini-2.5-flash")
r = genai.GenerativeModel(m).generate_content("ping")
print("Gemini OK" if r else "Gemini failed")
PY
```

#### ✅ Slack
Invite your bot to channels:
```
/invite @YourBot
```

---

### 7️⃣ Run the Agent

#### ▶️ Full orchestrator
```bash
python customer_journey_orchestrator.py
```
You can enter:
- Application ID (press Enter if unknown)
- Phone Number (press Enter if unknown)

#### 💬 NLP CLI
```bash
python NLP.py
```
or (if executable)
```bash
python Customer_Support_nlp
```

#### 🎟 Freshdesk module
```bash
python FreshDesk.py
```

**Generated files:**
```
journey_summary.csv
journey_key_events.csv
journey_summary.txt
journey_bundle.json
```

---

## 🧱 Repository Overview

| File | Purpose |
|------|----------|
| `customer_journey_orchestrator.py` | Core orchestrator – merges data from all sources, calls Gemini, and generates outputs |
| `Slack.py` / `slack_agent.py` | Reads messages from Slack support channels |
| `TeleCalling_Databricks.py` | Fetches tele-calling remarks & dispositions |
| `PartnerDB_Databrics.py` | Retrieves Partner Dashboard `application_comment` |
| `FreshDesk.py` | Gets Freshdesk ticket info from Databricks |
| `OnboardingTime.py` | Computes onboarding statuses and delays |
| `Email.py` *(optional)* | Parses Gmail support threads |
| `NLP.py` / `Customer_Support_nlp` | CLI for interactive customer queries |

---

## 🧠 System Architecture

```
[SOURCES]
  ├── Slack
  ├── Tele-Calling DB (Databricks)
  ├── Partner Dashboard (Databricks)
  ├── Freshdesk (Databricks)
  ├── Onboarding Timeline (Databricks)
  └── Email (optional)

[ADAPTERS]
  ├── Slack.py
  ├── TeleCalling_Databricks.py
  ├── PartnerDB_Databrics.py
  ├── FreshDesk.py
  ├── OnboardingTime.py
  └── Email.py

[ORCHESTRATOR]
  └── customer_journey_orchestrator.py

[INTELLIGENCE]
  └── Gemini 2.5-flash (google-generativeai)

[OUTPUTS]
  ├── journey_summary.csv
  ├── journey_key_events.csv
  ├── journey_summary.txt
  └── journey_bundle.json
```

---

## ⚙️ How It Works

1. You provide **phone number**, **application ID**, or **email**.  
2. The orchestrator resolves identifiers across all data sources.  
3. Each adapter fetches data concurrently from its system.  
4. All timestamps are converted to **IST (+05:30)**.  
5. Gemini 2.5-flash summarizes the merged data into insights.  
6. Results are written to CSV, TXT, and JSON outputs.

---

## 📦 Logs & Ignored Files

Add these entries to `.gitignore`:

```
# Generated reports
journey_*.csv
journey_summary.txt
journey_bundle.json

# Caches
*.pkl
slack_cache.pkl

# Virtual environment
venv/
.env
```

---

## 🛠 Troubleshooting

| Problem | Cause / Fix |
|----------|-------------|
| `python: command not found` | Use `python3` or activate your virtualenv: `source venv/bin/activate` |
| Gemini model error | Verify `GEMINI_API_KEY` and model name in `.env` |
| Databricks connection failed | Check hostname, HTTP path, and token in `.env` |
| Slack data empty | Invite bot to channels using `/invite @YourBot` |
| File won’t run | Use `python ./filename.py` or make it executable with `chmod +x` |

---

## 🔒 Security Guidelines

- Never commit `.env` or any file containing secrets.  
- Mask sensitive PII (e.g., show `xxxxxx6497` instead of full number).  
- Keep Databricks and Slack scopes minimal.  

---

## 🧭 Roadmap

- ✅ Multi-source orchestration  
- ✅ Gemini summarization  
- ✅ Interactive NLP CLI  
- 🔜 Automated weekly “Top Issues” CSV  
- 🔜 Slack `/journey <phone>` command  

---

## ✅ Acceptance Criteria

- Works with **only phone number** as input.  
- Generates:
  - `journey_summary.csv`
  - `journey_key_events.csv`
  - `journey_summary.txt`
  - `journey_bundle.json`
- All timestamps normalized to IST.  
- Partial adapter failures don’t break execution.  
- Cached runs execute faster.

---

**Maintainer:** _Internal AI & Automation Team_  
**Last updated:** October 2025  
**License:** Internal Use Only
