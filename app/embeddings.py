"""Background embedding jobs for memory_entries (OpenAI + pgvector)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from app.memory_store import MemoryStore


def _fetch_openai_embedding(api_key: str, model: str, text: str) -> list[float]:
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "input": text}
    with httpx.Client(timeout=60.0) as client:
        r = client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    items = data.get("data") or []
    if not items:
        raise ValueError("embeddings response missing data")
    emb = items[0].get("embedding")
    if not isinstance(emb, list):
        raise ValueError("invalid embedding payload")
    return [float(x) for x in emb]


def embed_memory_entry_background(
    store: MemoryStore,
    row_id: int,
    api_key: str,
    model: str,
    *,
    expected_dims: int = 1536,
) -> None:
    """
    Sync worker run via asyncio.to_thread: load note by id, embed, write pgvector column.
    """
    if not api_key.strip():
        print(f"Memory embedding skipped for id={row_id}: missing OPENAI_API_KEY.")
        return
    try:
        note = store.get_note_by_id(row_id)
        if note is None:
            print(f"Memory embedding skipped for id={row_id}: row not found.")
            return
        text = note.strip()
        if not text:
            print(f"Memory embedding skipped for id={row_id}: empty note.")
            return
        vector = _fetch_openai_embedding(api_key, model, text)
        if len(vector) != expected_dims:
            print(
                f"Memory embedding warning for id={row_id}: "
                f"got dim {len(vector)}, expected {expected_dims}."
            )
        store.update_embedding(row_id, vector)
        print(f"Memory embedding stored for memory_entries.id={row_id} ({model}).")
    except httpx.HTTPStatusError as exc:
        body = ""
        try:
            body = exc.response.text[:500]
        except Exception:
            pass
        print(
            f"Memory embedding HTTP error for id={row_id}: {exc.response.status_code} {body!r}"
        )
    except Exception as exc:
        print(f"Memory embedding failed for id={row_id}: {exc!r}")
