import asyncio
import json
import os
from typing import Optional

from ddgs import DDGS  # type: ignore[reportMissingImports]
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
import websockets
from websockets.exceptions import ConnectionClosed

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
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "3"))


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.post("/twilio/voice")
async def twilio_voice(_: Request) -> PlainTextResponse:
    if not PUBLIC_BASE_URL:
        return PlainTextResponse(
            "Set PUBLIC_BASE_URL in environment before handling live calls.",
            status_code=500,
        )

    stream_url = PUBLIC_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>Connecting you now.</Say>
  <Connect>
    <Stream url="{stream_url}/twilio/media-stream" />
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
                    "Search the web for recent or factual information and return concise results."
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

    session_update = {
        "type": "session.update",
        "session": {
            "modalities": ["audio", "text"],
            "instructions": f"{ASSISTANT_INSTRUCTIONS}\n\n{ASSISTANT_RESPONSE_POLICY}",
            "voice": OPENAI_VOICE,
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
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


async def send_openai_response_create(openai_ws) -> None:
    await openai_ws.send(
        json.dumps(
            {
                "type": "response.create",
                "response": {"modalities": ["audio", "text"]},
            }
        )
    )


async def send_openai_initial_greeting(openai_ws) -> None:
    await openai_ws.send(
        json.dumps(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "In English, greet the caller warmly in one short sentence. His name is Ryan. "
                            ),
                        }
                    ],
                },
            }
        )
    )
    await send_openai_response_create(openai_ws)


async def run_web_search(query: str) -> dict:
    def _search() -> dict:
        with DDGS() as ddgs:
            rows = list(ddgs.text(query, max_results=WEB_SEARCH_MAX_RESULTS))
        items = []
        for row in rows[:WEB_SEARCH_MAX_RESULTS]:
            items.append(
                {
                    "title": row.get("title", ""),
                    "snippet": row.get("body", ""),
                    "url": row.get("href", ""),
                }
            )
        return {"query": query, "results": items}

    try:
        return await asyncio.wait_for(asyncio.to_thread(_search), timeout=8)
    except Exception as exc:
        return {"query": query, "error": str(exc), "results": []}


async def handle_tool_call(openai_ws, call_id: str, tool_name: str, arguments_raw: str) -> None:
    if tool_name != "web_search":
        output = {"error": f"Unknown tool: {tool_name}"}
    else:
        try:
            args = json.loads(arguments_raw or "{}")
        except json.JSONDecodeError:
            args = {}
        query = (args.get("query") or "").strip()
        if not query:
            output = {"error": "Missing required argument: query"}
        else:
            print(f"Tool call: web_search('{query}')")
            output = await run_web_search(query)

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
    media_packets = 0

    realtime_url = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    try:
        async with websockets.connect(realtime_url, additional_headers=headers) as openai_ws:
            await send_openai_session_update(openai_ws)

            async def forward_twilio_to_openai() -> None:
                nonlocal stream_sid, media_packets
                try:
                    while True:
                        message = await ws.receive_text()
                        data = json.loads(message)
                        event = data.get("event")

                        if event == "start":
                            stream_sid = data.get("start", {}).get("streamSid")
                            print(f"Twilio stream started: {stream_sid}")
                            await send_openai_initial_greeting(openai_ws)
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
                            )
                except ConnectionClosed as exc:
                    print(f"OpenAI websocket closed: {exc}")
                except Exception as exc:
                    print(f"OpenAI->Twilio forwarding error: {exc}")

            await asyncio.gather(forward_twilio_to_openai(), forward_openai_to_twilio())
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
        if ws.client_state.name != "DISCONNECTED":
            await ws.close()
