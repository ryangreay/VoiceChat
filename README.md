# VoiceChat

A phone-call voice assistant that bridges **Twilio Media Streams** to the **OpenAI Realtime API (GA)**. Call a Twilio number and talk to an LLM in real time, with optional caller memory, web search, and post-call summaries.

## What it does

```
Caller → Twilio Phone → POST /twilio/voice (TwiML)
                      → WebSocket /twilio/media-stream (μ-law audio)
                      → OpenAI Realtime (gpt-realtime / gpt-realtime-2)
                      → audio back to caller
```

| Feature | Status |
| --- | --- |
| Live speech-to-speech on a phone call | ✅ |
| OpenAI GA Realtime (`audio/pcmu` ↔ Twilio G.711 μ-law) | ✅ |
| Server-side VAD and tool calling | ✅ |
| Web search tool (DDGS + page extract + optional LLM summary) | ✅ |
| Postgres memory (Neon-compatible) | ✅ optional |
| Caller profiles + `save_caller_name` | ✅ with memory |
| Auto call summary saved after each call | ✅ with memory |
| pgvector embeddings (background worker) | ✅ stored; semantic search not wired yet |
| Twilio webhook signature validation | ✅ when `TWILIO_AUTH_TOKEN` is set |

## Requirements

- Python 3.12+
- [Twilio](https://www.twilio.com/) account with a voice-capable phone number
- [OpenAI API key](https://platform.openai.com/api-keys) with Realtime access
- Public HTTPS URL for webhooks (e.g. [ngrok](https://ngrok.com/) for local dev)
- Optional: [Neon](https://neon.tech/) or other Postgres for memory

## Quick start (local)

### 1. Clone and install

```powershell
git clone https://github.com/<your-user>/VoiceChat.git
cd VoiceChat
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Configure environment

```powershell
copy .env.example .env
```

Edit `.env`. Minimum for a working call:

| Variable | Description |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI secret key (server-side only) |
| `PUBLIC_BASE_URL` | Public HTTPS base URL, no trailing slash (e.g. ngrok URL) |
| `OPENAI_REALTIME_MODEL` | `gpt-realtime` (default) or `gpt-realtime-2` |

For production webhooks, also set:

| Variable | Description |
| --- | --- |
| `TWILIO_AUTH_TOKEN` | From Twilio Console → Account → API keys & tokens. Enables signature checks on `/twilio/voice`. |

See `.env.example` for memory, web search, voice, and VAD tuning.

### 3. Run

```powershell
uvicorn app.main:app --reload --port 8000
```

Health check: `GET http://localhost:8000/health`

### 4. Expose to the internet

```powershell
ngrok http 8000
```

Set `PUBLIC_BASE_URL` to the ngrok **HTTPS** URL (e.g. `https://abc123.ngrok-free.app`).

### 5. Configure Twilio

In the Twilio Console, for your phone number → **Voice configuration**:

- **A call comes in:** Webhook `https://<PUBLIC_BASE_URL>/twilio/voice`, method **POST**

On incoming calls, Twilio fetches TwiML, then opens a media stream to `wss://<host>/twilio/media-stream`.

## Memory (optional)

Enable Postgres-backed tools and auto summaries:

```env
MEMORY_ENABLED=true
DATABASE_URL=postgresql://user:pass@host/dbname?sslmode=require
```

On startup the app creates `memory_entries` and `caller_profiles` tables (and attempts `vector` extension for embeddings).

**Tools exposed to the model:**

- `save_memory` — durable notes about the caller
- `search_memory` — text search over past notes
- `get_recent_memories` — latest notes for session context
- `save_caller_name` — store preferred name by phone number

After each call, an LLM-generated summary is saved automatically (tag `auto-call-summary`).

## Deployment (Fly.io)

This repo includes a `Dockerfile` and example `fly.toml`. Before deploying:

1. Create your own Fly app: `fly apps create <your-app-name>` and set `app` in `fly.toml`.
2. Set secrets on Fly (never in git):

   ```powershell
   fly secrets set OPENAI_API_KEY=sk-... PUBLIC_BASE_URL=https://<your-app>.fly.dev TWILIO_AUTH_TOKEN=...
   ```

3. Optional: `fly secrets set DATABASE_URL=...` and `MEMORY_ENABLED=true`.

GitHub Actions deploy (`.github/workflows/fly-deploy.yml`) expects `FLY_API_TOKEN` in repository secrets.

## Realtime API notes

- Uses the **GA** interface (no `OpenAI-Beta: realtime=v1` header).
- Deprecated preview models (e.g. `gpt-4o-realtime-preview`) will not work.
- Output modality is `["audio"]` only; transcripts use separate Realtime events.
- Twilio μ-law is configured as `audio/pcmu` in session audio settings.

## Security

### Twilio webhook validation

When `TWILIO_AUTH_TOKEN` is set, `POST /twilio/voice` validates the `X-Twilio-Signature` header against `PUBLIC_BASE_URL/twilio/voice`. Leave the token unset only for local experiments.

### Logging and PII

Server logs may include phone numbers (`Caller id`), transcripts, and tool arguments. Treat logs as sensitive. Avoid shipping logs to public systems without redaction.

### Web search

The `web_search` tool fetches arbitrary URLs from the public internet. Run with `WEB_SEARCH_ENABLED=false` if you do not want outbound HTTP from your server.

## Project layout

```
app/
  main.py          # FastAPI app, Twilio bridge, Realtime session, tools
  memory_store.py  # Postgres memory + caller profiles
  embeddings.py    # Background OpenAI embeddings for memory rows
  web_search.py    # DDGS + trafilatura + optional summarization
```

## Roadmap ideas

- Semantic memory search over embeddings
- Similar functionality over sms/rcs. Requires A2P 10DLC approval on twilio
- Stronger web search / deep research. For example, checking the weather, checking and booking movie times, etc

## License

MIT — see [LICENSE](LICENSE).
