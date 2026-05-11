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
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                except psycopg.Error:
                    pass
                try:
                    cur.execute(
                        """
                        ALTER TABLE memory_entries
                        ADD COLUMN IF NOT EXISTS embedding vector(1536);
                        """
                    )
                except psycopg.Error:
                    pass
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS caller_profiles (
                        caller_id TEXT PRIMARY KEY,
                        display_name TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )

    def get_caller_profile(self, caller_id: str) -> dict[str, Any] | None:
        with psycopg.connect(self.database_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT caller_id, display_name, created_at, updated_at
                    FROM caller_profiles
                    WHERE caller_id = %s;
                    """,
                    (caller_id,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return {
            "caller_id": row[0],
            "display_name": row[1],
            "created_at": row[2].isoformat() if row[2] else None,
            "updated_at": row[3].isoformat() if row[3] else None,
        }

    def upsert_caller_name(self, caller_id: str, display_name: str) -> dict[str, Any]:
        name = display_name.strip()
        if not name:
            raise ValueError("display_name must be non-empty")
        with psycopg.connect(self.database_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO caller_profiles (caller_id, display_name)
                    VALUES (%s, %s)
                    ON CONFLICT (caller_id) DO UPDATE SET
                        display_name = EXCLUDED.display_name,
                        updated_at = NOW()
                    RETURNING caller_id, display_name, created_at, updated_at;
                    """,
                    (caller_id, name),
                )
                row = cur.fetchone()
        return {
            "caller_id": row[0],
            "display_name": row[1],
            "created_at": row[2].isoformat() if row[2] else None,
            "updated_at": row[3].isoformat() if row[3] else None,
        }

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

    def get_note_by_id(self, row_id: int) -> str | None:
        with psycopg.connect(self.database_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT note FROM memory_entries WHERE id = %s;
                    """,
                    (row_id,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return row[0]

    def update_embedding(self, row_id: int, embedding: list[float]) -> None:
        literal = "[" + ",".join(str(float(x)) for x in embedding) + "]"
        with psycopg.connect(self.database_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE memory_entries
                    SET embedding = %s::vector
                    WHERE id = %s;
                    """,
                    (literal, row_id),
                )

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
