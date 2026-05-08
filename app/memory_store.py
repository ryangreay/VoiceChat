from typing import Any

import psycopg


class MemoryStore:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def init_schema(self) -> None:
        with psycopg.connect(self.database_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_entries (
                        id BIGSERIAL PRIMARY KEY,
                        caller_id TEXT NOT NULL,
                        note TEXT NOT NULL,
                        tags TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_memory_entries_caller_created
                    ON memory_entries (caller_id, created_at DESC);
                    """
                )

    def save_memory(self, caller_id: str, note: str, tags: str | None = None) -> dict[str, Any]:
        with psycopg.connect(self.database_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memory_entries (caller_id, note, tags)
                    VALUES (%s, %s, %s)
                    RETURNING id, created_at;
                    """,
                    (caller_id, note, tags),
                )
                row = cur.fetchone()
        return {
            "ok": True,
            "id": row[0] if row else None,
            "created_at": row[1].isoformat() if row and row[1] else None,
        }

    def search_memory(self, caller_id: str, query: str, limit: int) -> list[dict[str, Any]]:
        with psycopg.connect(self.database_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, note, tags, created_at
                    FROM memory_entries
                    WHERE caller_id = %s
                      AND (note ILIKE %s OR COALESCE(tags, '') ILIKE %s)
                    ORDER BY created_at DESC
                    LIMIT %s;
                    """,
                    (caller_id, f"%{query}%", f"%{query}%", limit),
                )
                rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "note": row[1],
                "tags": row[2],
                "created_at": row[3].isoformat() if row[3] else None,
            }
            for row in rows
        ]

    def get_recent_memories(self, caller_id: str, limit: int) -> list[dict[str, Any]]:
        with psycopg.connect(self.database_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, note, tags, created_at
                    FROM memory_entries
                    WHERE caller_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s;
                    """,
                    (caller_id, limit),
                )
                rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "note": row[1],
                "tags": row[2],
                "created_at": row[3].isoformat() if row[3] else None,
            }
            for row in rows
        ]
