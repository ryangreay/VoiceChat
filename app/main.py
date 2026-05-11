import asyncio
import json
import os
from typing import Optional
from urllib.parse import parse_qs

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
import websockets
from websockets.exceptions import ConnectionClosed
from app.memory_store import MemoryStore
from app.web_search import run_web_search

load_dotenv()

app = FastAPI(title="VoiceChat", version="0.1.0")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")
ASSISTANT_INSTRUCTIONS = os.getenv(
    "ASSISTANT_INSTRUCTIONS",
    "You are a friendly peer-like assistant on a live phone call. Be warm, concise, and practical.",
)
ASSISTANT_RESPONSE_POLICY = (
    "Non-negotiable rules: "
    "1) Speak only in English unless the caller explicitly asks for another language. "
    "2) Keep responses short and realistic for a phone call (1-2 short sentences). "
    "3) Avoid long lists, long explanations, and rambling. "
    "4) Use a natural, conversational tone."
)
OPENAI_VOICE = os.getenv("OPENAI_VOICE", "alloy")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")
VAD_THRESHOLD = float(os.getenv("OPENAI_VAD_THRESHOLD", "0.6"))
VAD_PREFIX_PADDING_MS = int(os.getenv("OPENAI_VAD_PREFIX_PADDING_MS", "300"))
VAD_SILENCE_DURATION_MS = int(os.getenv("OPENAI_VAD_SILENCE_DURATION_MS", "900"))
WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "true").lower() == "true"
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "false").lower() == "true"
DATABASE_URL = os.getenv("DATABASE_URL", "")
MEMORY_RECENT_LIMIT = int(os.getenv("MEMORY_RECENT_LIMIT", "5"))
# User-speech ASR for call summaries / logs; Realtime defaults transcription OFF without this.
INPUT_AUDIO_TRANSCRIPTION_MODEL = os.getenv("INPUT_AUDIO_TRANSCRIPTION_MODEL", "whisper-1")
INPUT_AUDIO_TRANSCRIPTION_LANGUAGE = os.getenv("INPUT_AUDIO_TRANSCRIPTION_LANGUAGE", "en").strip()

memory_store: Optional[MemoryStore] = None
if MEMORY_ENABLED and DATABASE_URL:
    memory_store = MemoryStore(DATABASE_URL)


def build_input_audio_transcription() -> dict:
    cfg: dict = {"model": INPUT_AUDIO_TRANSCRIPTION_MODEL}
    if INPUT_AUDIO_TRANSCRIPTION_LANGUAGE:
        cfg["language"] = INPUT_AUDIO_TRANSCRIPTION_LANGUAGE
    return cfg


def build_session_instructions(
    caller_profile: dict | None = None,
    *,
    include_caller_identity: bool = True,
) -> str:
    parts = [
        ASSISTANT_INSTRUCTIONS,
        ASSISTANT_RESPONSE_POLICY,
    ]
    if memory_store:
        parts.append(
            "When memory tools are available, use them for durable preferences and "
            "to look up relevant prior notes from past calls."
        )
        if include_caller_identity:
            if caller_profile and caller_profile.get("display_name"):
                name = caller_profile["display_name"]
                parts.append(
                    f"Caller identity: this phone number is associated with the saved name '{name}'. "
                    "Greet them using that name when the call opens. Do not ask for their name unless "
                    "they say it is wrong or they want to change it. If they give a new preferred name, "
                    "call save_caller_name immediately with the new name."
                )
            else:
                parts.append(
                    "Caller identity: there is no saved name yet for this phone number. "
                    "After your opening greeting, politely ask what name they would like you to use. "
                    "When they clearly state their preferred name, call save_caller_name once with "
                    "that name (use the name they gave, not a nickname you invent). "
                    "Then briefly acknowledge you will remember it."
                )
    return "\n\n".join(parts)


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.on_event("startup")
async def startup_event() -> None:
    if memory_store:
        await asyncio.to_thread(memory_store.init_schema)
        print("Memory store ready.")


@app.post("/twilio/voice")
async def twilio_voice(request: Request) -> PlainTextResponse:
    if not PUBLIC_BASE_URL:
        return PlainTextResponse(
            "Set PUBLIC_BASE_URL in environment before handling live calls.",
            status_code=500,
        )

    stream_url = PUBLIC_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
    body = (await request.body()).decode("utf-8")
    form_data = parse_qs(body)
    from_number = form_data.get("From", ["unknown"])[0]
    call_sid = form_data.get("CallSid", [""])[0]

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>Connecting you now.</Say>
  <Connect>
    <Stream url="{stream_url}/twilio/media-stream">
      <Parameter name="from_number" value="{from_number}" />
      <Parameter name="call_sid" value="{call_sid}" />
    </Stream>
  </Connect>
</Response>"""
    return PlainTextResponse(twiml, media_type="application/xml")


async def send_openai_session_update(openai_ws) -> None:
    tools = []
    if WEB_SEARCH_ENABLED:
        tools.append(
            {
                "type": "function",
                "name": "web_search",
                "description": (
                    "Search the web for recent or factual information. Returns a synthesized summary "
                    "from fetched pages plus source titles and URLs."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user query to search for on the web.",
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            }
        )
    if memory_store:
        tools.extend(
            [
                {
                    "type": "function",
                    "name": "save_memory",
                    "description": "Save a durable memory note about the caller for future calls.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "note": {
                                "type": "string",
                                "description": "Short durable memory about the caller.",
                            },
                            "tags": {
                                "type": "string",
                                "description": "Optional comma-separated tags for retrieval.",
                            },
                        },
                        "required": ["note"],
                        "additionalProperties": False,
                    },
                },
                {
                    "type": "function",
                    "name": "search_memory",
                    "description": "Search past memories for this caller with text matching.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "What to look up in caller memory.",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Optional max memory rows to return.",
                            },
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
                {
                    "type": "function",
                    "name": "get_recent_memories",
                    "description": "Get most recent saved memories for this caller.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Optional max memory rows to return.",
                            }
                        },
                        "additionalProperties": False,
                    },
                },
                {
                    "type": "function",
                    "name": "save_caller_name",
                    "description": (
                        "Save this caller's preferred name for future calls from this phone number. "
                        "Call once when they clearly state how they want to be addressed."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Preferred name exactly as the caller gave it (first name or full name).",
                            },
                        },
                        "required": ["name"],
                        "additionalProperties": False,
                    },
                },
            ]
        )

    session_update = {
        "type": "session.update",
        "session": {
            "modalities": ["audio", "text"],
            "instructions": build_session_instructions(None, include_caller_identity=False),
            "voice": OPENAI_VOICE,
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "input_audio_transcription": build_input_audio_transcription(),
            "turn_detection": {
                "type": "server_vad",
                "threshold": VAD_THRESHOLD,
                "prefix_padding_ms": VAD_PREFIX_PADDING_MS,
                "silence_duration_ms": VAD_SILENCE_DURATION_MS,
                "create_response": True,
            },
            "tools": tools,
        },
    }
    await openai_ws.send(json.dumps(session_update))


async def patch_session_for_caller(openai_ws, caller_profile: dict | None) -> None:
    await openai_ws.send(
        json.dumps(
            {
                "type": "session.update",
                "session": {
                    "instructions": build_session_instructions(
                        caller_profile, include_caller_identity=True
                    )
                },
            }
        )
    )


async def send_openai_response_create(openai_ws) -> None:
    await openai_ws.send(
        json.dumps(
            {
                "type": "response.create",
                "response": {"modalities": ["audio", "text"]},
            }
        )
    )


async def send_call_opening(openai_ws, caller_profile: dict | None) -> None:
    if not memory_store:
        prompt = (
            "In English: greet the caller warmly in one short sentence, "
            "then ask how you can help in one short sentence."
        )
    elif caller_profile and caller_profile.get("display_name"):
        name = caller_profile["display_name"]
        prompt = (
            f"In English: greet {name} warmly by name in one short sentence, "
            "then ask how you can help in one short sentence."
        )
    else:
        prompt = (
            "In English: give one warm, welcoming sentence. "
            "Then ask what name they would like you to use for them, in one short sentence."
        )
    await openai_ws.send(
        json.dumps(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            }
        )
    )
    await send_openai_response_create(openai_ws)


def summarize_call_memory(
    caller_id: str,
    call_sid: str,
    user_utterances: list[str],
    assistant_utterances: list[str],
) -> str:
    user_preview = " | ".join(user_utterances[-3:]) if user_utterances else "No transcription captured."
    assistant_preview = (
        " | ".join(assistant_utterances[-3:]) if assistant_utterances else "No assistant transcript captured."
    )
    return (
        f"Call summary for {caller_id}. "
        f"CallSid: {call_sid or 'unknown'}. "
        f"User said: {user_preview}. "
        f"Assistant replied: {assistant_preview}."
    )


def clamp_limit(value: object, default: int = MEMORY_RECENT_LIMIT) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    return max(1, min(10, parsed))


async def handle_tool_call(
    openai_ws,
    call_id: str,
    tool_name: str,
    arguments_raw: str,
    caller_id: str,
) -> None:
    try:
        args = json.loads(arguments_raw or "{}")
    except json.JSONDecodeError:
        args = {}

    if tool_name == "web_search":
        query = (args.get("query") or "").strip()
        if not query:
            output = {"error": "Missing required argument: query"}
        else:
            print(f"Tool call: web_search('{query}')")
            output = await run_web_search(query)
    elif tool_name == "save_memory":
        if not memory_store:
            output = {"error": "Memory is disabled."}
        else:
            note = (args.get("note") or "").strip()
            tags = (args.get("tags") or "").strip() or None
            if not note:
                output = {"error": "Missing required argument: note"}
            else:
                print(f"Tool call: save_memory for caller '{caller_id}'")
                saved = await asyncio.to_thread(memory_store.save_memory, caller_id, note, tags)
                output = {"ok": True, "saved": saved}
    elif tool_name == "search_memory":
        if not memory_store:
            output = {"error": "Memory is disabled."}
        else:
            query = (args.get("query") or "").strip()
            limit = clamp_limit(args.get("limit"), MEMORY_RECENT_LIMIT)
            if not query:
                output = {"error": "Missing required argument: query"}
            else:
                print(f"Tool call: search_memory('{query}') for caller '{caller_id}'")
                rows = await asyncio.to_thread(memory_store.search_memory, caller_id, query, limit)
                output = {"query": query, "results": rows}
    elif tool_name == "get_recent_memories":
        if not memory_store:
            output = {"error": "Memory is disabled."}
        else:
            limit = clamp_limit(args.get("limit"), MEMORY_RECENT_LIMIT)
            print(f"Tool call: get_recent_memories for caller '{caller_id}'")
            rows = await asyncio.to_thread(memory_store.get_recent_memories, caller_id, limit)
            output = {"results": rows}
    elif tool_name == "save_caller_name":
        if not memory_store:
            output = {"error": "Memory is disabled."}
        else:
            raw_name = (args.get("name") or "").strip()
            if not raw_name:
                output = {"error": "Missing required argument: name"}
            else:
                print(f"Tool call: save_caller_name('{raw_name}') for caller '{caller_id}'")
                try:
                    saved = await asyncio.to_thread(
                        memory_store.upsert_caller_name, caller_id, raw_name
                    )
                    output = {"ok": True, "profile": saved}
                except ValueError as exc:
                    output = {"error": str(exc)}
    else:
        output = {"error": f"Unknown tool: {tool_name}"}

    await openai_ws.send(
        json.dumps(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(output),
                },
            }
        )
    )
    await send_openai_response_create(openai_ws)


@app.websocket("/twilio/media-stream")
async def twilio_media_stream(ws: WebSocket) -> None:
    await ws.accept()

    if not OPENAI_API_KEY:
        await ws.close(code=1011)
        return

    stream_sid: Optional[str] = None
    caller_id = "unknown"
    call_sid = ""
    media_packets = 0
    user_utterances: list[str] = []
    assistant_utterances: list[str] = []
    call_memory_saved = False

    realtime_url = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    try:
        async with websockets.connect(realtime_url, additional_headers=headers) as openai_ws:
            await send_openai_session_update(openai_ws)

            async def forward_twilio_to_openai() -> None:
                nonlocal stream_sid, caller_id, call_sid, media_packets
                try:
                    while True:
                        message = await ws.receive_text()
                        data = json.loads(message)
                        event = data.get("event")

                        if event == "start":
                            stream_sid = data.get("start", {}).get("streamSid")
                            custom_params = data.get("start", {}).get("customParameters", {})
                            caller_id = custom_params.get("from_number", "unknown")
                            call_sid = custom_params.get("call_sid", "")
                            print(f"Twilio stream started: {stream_sid}")
                            print(f"Caller id: {caller_id}")
                            caller_profile = None
                            if memory_store and caller_id != "unknown":
                                caller_profile = await asyncio.to_thread(
                                    memory_store.get_caller_profile, caller_id
                                )
                                if caller_profile:
                                    print(
                                        f"Caller profile loaded: name={caller_profile.get('display_name')}"
                                    )
                                else:
                                    print("No caller profile yet; first-time greeting flow.")
                            if memory_store:
                                await patch_session_for_caller(openai_ws, caller_profile)
                            await send_call_opening(openai_ws, caller_profile)
                        elif event == "media":
                            payload = data.get("media", {}).get("payload")
                            if payload:
                                media_packets += 1
                                if media_packets % 50 == 0:
                                    print(f"Twilio media packets received: {media_packets}")
                                await openai_ws.send(
                                    json.dumps(
                                        {
                                            "type": "input_audio_buffer.append",
                                            "audio": payload,
                                        }
                                    )
                                )
                        elif event == "stop":
                            print("Twilio stream stop received.")
                            try:
                                await openai_ws.close()
                            except Exception:
                                pass
                            break
                except WebSocketDisconnect:
                    print("Twilio websocket disconnected.")
                except ConnectionClosed as exc:
                    print(f"OpenAI websocket closed while sending audio: {exc}")
                except Exception as exc:
                    print(f"Twilio->OpenAI forwarding error: {exc}")

            async def forward_openai_to_twilio() -> None:
                try:
                    while True:
                        raw = await openai_ws.recv()
                        event = json.loads(raw)
                        event_type = event.get("type")

                        if event_type in {
                            "session.created",
                            "session.updated",
                            "input_audio_buffer.speech_started",
                            "input_audio_buffer.speech_stopped",
                            "response.created",
                            "response.done",
                        }:
                            print(f"OpenAI event: {event_type}")

                        if event_type in {"response.audio.delta", "response.output_audio.delta"}:
                            audio_chunk = event.get("delta")
                            if audio_chunk and stream_sid:
                                await ws.send_text(
                                    json.dumps(
                                        {
                                            "event": "media",
                                            "streamSid": stream_sid,
                                            "media": {"payload": audio_chunk},
                                        }
                                    )
                                )
                        elif event_type == "error":
                            # Keep the call alive but surface errors in server logs.
                            print("OpenAI realtime error:", event)
                        elif event_type == "response.function_call_arguments.done":
                            await handle_tool_call(
                                openai_ws=openai_ws,
                                call_id=event.get("call_id", ""),
                                tool_name=event.get("name", ""),
                                arguments_raw=event.get("arguments", "{}"),
                                caller_id=caller_id,
                            )
                        elif event_type == "conversation.item.input_audio_transcription.completed":
                            transcript = (event.get("transcript") or "").strip()
                            if transcript:
                                user_utterances.append(transcript)
                                print(f"OpenAI user transcript: {transcript[:200]!r}")
                        elif event_type == "conversation.item.input_audio_transcription.failed":
                            print(
                                "OpenAI input_audio_transcription failed:",
                                event.get("error") or event,
                            )
                        elif event_type in {"response.audio_transcript.done", "response.output_audio_transcript.done"}:
                            transcript = (event.get("transcript") or "").strip()
                            if transcript:
                                assistant_utterances.append(transcript)
                except ConnectionClosed as exc:
                    print(f"OpenAI websocket closed: {exc}")
                except Exception as exc:
                    print(f"OpenAI->Twilio forwarding error: {exc}")

            twilio_task = asyncio.create_task(forward_twilio_to_openai())
            openai_task = asyncio.create_task(forward_openai_to_twilio())
            done, pending = await asyncio.wait(
                {twilio_task, openai_task}, return_when=asyncio.FIRST_COMPLETED
            )

            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                _ = task.exception()
    except Exception as exc:
        err_msg = str(exc)
        if "invalid_model" in err_msg:
            print(
                "Bridge runtime error: invalid realtime model. "
                f"Current OPENAI_REALTIME_MODEL='{OPENAI_REALTIME_MODEL}'. "
                "Set a valid realtime-capable model in your .env file."
            )
        else:
            print("Bridge runtime error:", exc)
    finally:
        if memory_store and caller_id != "unknown" and not call_memory_saved:
            note = summarize_call_memory(
                caller_id=caller_id,
                call_sid=call_sid,
                user_utterances=user_utterances,
                assistant_utterances=assistant_utterances,
            )
            try:
                await asyncio.to_thread(memory_store.save_memory, caller_id, note, "auto-call-summary")
                call_memory_saved = True
                print(f"Auto-saved call summary for caller '{caller_id}'.")
            except Exception as exc:
                print(f"Failed to auto-save call summary: {exc}")
        if ws.client_state.name != "DISCONNECTED":
            try:
                await ws.close()
            except Exception:
                pass
