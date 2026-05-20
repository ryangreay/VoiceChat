# VoiceChat (Step 1 Bootstrap)

This project starts a phone-call conversational assistant using:

- Twilio Programmable Voice + Media Streams
- OpenAI Realtime API (audio in/out)
- FastAPI backend

The initial goal is to call a Twilio phone number and have a live voice conversation with an LLM.

## 1) Local setup (PowerShell)

1. Create a virtual environment:

```powershell
python -m venv .venv
```

2. Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies inside the venv:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Copy `.env.example` to `.env` and fill in values.

Minimum required env vars:

- `OPENAI_API_KEY`
- `PUBLIC_BASE_URL` (your externally reachable HTTPS URL, e.g. ngrok)
- `OPENAI_REALTIME_MODEL` (default `gpt-realtime`; also try `gpt-realtime-2`)

The app uses the **GA** Realtime API (not the removed beta `OpenAI-Beta: realtime=v1` interface).
Deprecated preview models like `gpt-4o-realtime-preview` will not work.

## 2) Run the app

```powershell
uvicorn app.main:app --reload --port 8000
```

Health check:

- `GET http://localhost:8000/health`

## 3) Expose locally to Twilio

Use ngrok (or similar):

```powershell
ngrok http 8000
```

Set `PUBLIC_BASE_URL` to your ngrok HTTPS URL.

## 4) Configure Twilio number

In Twilio Console for your phone number:

- Voice webhook URL (A call comes in):  
  `https://<your-public-host>/twilio/voice`
- HTTP method: `POST`

When you call the number, Twilio requests TwiML from `/twilio/voice`, then opens a media stream to:

- `wss://<your-public-host>/twilio/media-stream`

The server bridges that stream to OpenAI Realtime.

## 5) Phase 1 memory with Postgres

Memory tools are available when `MEMORY_ENABLED=true` and `DATABASE_URL` is set.

- Uses a simple text-memory table in Postgres (Neon compatible)
- **`caller_profiles`** stores each caller’s preferred name (keyed by phone number). First calls ask for a name and use the **`save_caller_name`** tool; later calls greet by that name.
- Scopes memories by caller phone number
- Exposes tools to the model:
  - `save_memory`
  - `search_memory`
  - `get_recent_memories`
  - `save_caller_name`

Example env values:

- `MEMORY_ENABLED=true`
- `DATABASE_URL=postgresql://...` (include `sslmode=require` for Neon)
- `MEMORY_RECENT_LIMIT=5`

## Notes and next steps

- This is a practical starter for Step 1 in your plan.
- Current code is intentionally minimal and does not yet include:
  - Twilio request signature validation
  - rich logging/observability
  - production-grade reconnection strategy
  - persistent memory/tool calling

Planned roadmap:

1. Stabilize call quality + error handling
2. Add custom voice cloning/voice pipeline for your own voice
3. Add Neon + pgvector memory summaries/retrieval
4. Add tool calls (calendar/email/text workflows)
