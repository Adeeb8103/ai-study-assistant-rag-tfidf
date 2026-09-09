"""
SQLite persistence layer.

Plain `sqlite3` from the standard library is enough here - the schema is small
and keeping it dependency-free makes the project easy to run on any machine.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterable

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    email         TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    salt          TEXT    NOT NULL,
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title       TEXT    NOT NULL,
    filename    TEXT    NOT NULL,
    filepath    TEXT    NOT NULL,
    filetype    TEXT    NOT NULL,
    num_pages   INTEGER DEFAULT 0,
    num_chars   INTEGER DEFAULT 0,
    num_chunks  INTEGER DEFAULT 0,
    index_dir   TEXT,
    uploaded_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    document_id INTEGER REFERENCES documents(id) ON DELETE SET NULL,
    title       TEXT    NOT NULL,
    summary     TEXT    DEFAULT '',
    created_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role         TEXT    NOT NULL,          -- 'user' | 'assistant'
    content      TEXT    NOT NULL,
    approach     TEXT    DEFAULT '',        -- 'modern' | 'traditional'
    latency      REAL    DEFAULT 0,
    tokens_in    INTEGER DEFAULT 0,
    tokens_out   INTEGER DEFAULT 0,
    cost         REAL    DEFAULT 0,
    sources      TEXT    DEFAULT '[]',      -- JSON list of retrieved chunks
    created_at   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    document_id INTEGER REFERENCES documents(id) ON DELETE SET NULL,
    name        TEXT    NOT NULL,
    model       TEXT    DEFAULT '',
    created_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    question    TEXT    NOT NULL,
    reference   TEXT    DEFAULT '',
    approach    TEXT    NOT NULL,
    answer      TEXT    DEFAULT '',
    latency     REAL    DEFAULT 0,
    tokens_in   INTEGER DEFAULT 0,
    tokens_out  INTEGER DEFAULT 0,
    cost        REAL    DEFAULT 0,
    accuracy    REAL    DEFAULT 0,
    keyword_hit REAL    DEFAULT 0,
    naturalness REAL    DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_documents_user   ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user    ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_results_run      ON eval_results(run_id);
"""


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)


def query(sql: str, params: Iterable[Any] = ()) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    return [dict(row) for row in rows]


def query_one(sql: str, params: Iterable[Any] = ()) -> dict | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: Iterable[Any] = ()) -> int:
    """Run a write statement and return the new/affected row id."""
    with get_connection() as conn:
        cursor = conn.execute(sql, tuple(params))
        return cursor.lastrowid


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
def add_document(user_id: int, title: str, filename: str, filepath: str,
                 filetype: str, num_pages: int, num_chars: int,
                 num_chunks: int, index_dir: str) -> int:
    return execute(
        """INSERT INTO documents
           (user_id, title, filename, filepath, filetype, num_pages,
            num_chars, num_chunks, index_dir, uploaded_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (user_id, title, filename, filepath, filetype, num_pages,
         num_chars, num_chunks, index_dir, now()),
    )


def list_documents(user_id: int) -> list[dict]:
    return query(
        "SELECT * FROM documents WHERE user_id = ? ORDER BY uploaded_at DESC",
        (user_id,),
    )


def get_document(doc_id: int, user_id: int) -> dict | None:
    return query_one(
        "SELECT * FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id)
    )


def delete_document(doc_id: int, user_id: int) -> None:
    execute("DELETE FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id))


# ---------------------------------------------------------------------------
# Chat sessions and messages
# ---------------------------------------------------------------------------
def create_session(user_id: int, document_id: int | None, title: str) -> int:
    return execute(
        """INSERT INTO chat_sessions (user_id, document_id, title, summary, created_at)
           VALUES (?,?,?,'',?)""",
        (user_id, document_id, title, now()),
    )


def list_sessions(user_id: int) -> list[dict]:
    return query(
        "SELECT * FROM chat_sessions WHERE user_id = ? ORDER BY id DESC", (user_id,)
    )


def delete_session(session_id: int, user_id: int) -> None:
    execute(
        "DELETE FROM chat_sessions WHERE id = ? AND user_id = ?", (session_id, user_id)
    )


def update_session_summary(session_id: int, summary: str) -> None:
    execute("UPDATE chat_sessions SET summary = ? WHERE id = ?", (summary, session_id))


def add_message(session_id: int, role: str, content: str, approach: str = "",
                latency: float = 0.0, tokens_in: int = 0, tokens_out: int = 0,
                cost: float = 0.0, sources: list | None = None) -> int:
    return execute(
        """INSERT INTO messages
           (session_id, role, content, approach, latency, tokens_in,
            tokens_out, cost, sources, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (session_id, role, content, approach, latency, tokens_in, tokens_out,
         cost, json.dumps(sources or []), now()),
    )


def list_messages(session_id: int) -> list[dict]:
    rows = query(
        "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,)
    )
    for row in rows:
        try:
            row["sources"] = json.loads(row["sources"] or "[]")
        except json.JSONDecodeError:
            row["sources"] = []
    return rows


# ---------------------------------------------------------------------------
# Evaluation runs
# ---------------------------------------------------------------------------
def create_eval_run(user_id: int, document_id: int | None, name: str,
                    model: str) -> int:
    return execute(
        """INSERT INTO eval_runs (user_id, document_id, name, model, created_at)
           VALUES (?,?,?,?,?)""",
        (user_id, document_id, name, model, now()),
    )


def add_eval_result(run_id: int, **kwargs) -> int:
    return execute(
        """INSERT INTO eval_results
           (run_id, question, reference, approach, answer, latency, tokens_in,
            tokens_out, cost, accuracy, keyword_hit, naturalness)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (run_id,
         kwargs.get("question", ""), kwargs.get("reference", ""),
         kwargs.get("approach", ""), kwargs.get("answer", ""),
         kwargs.get("latency", 0.0), kwargs.get("tokens_in", 0),
         kwargs.get("tokens_out", 0), kwargs.get("cost", 0.0),
         kwargs.get("accuracy", 0.0), kwargs.get("keyword_hit", 0.0),
         kwargs.get("naturalness", 0.0)),
    )


def list_eval_runs(user_id: int) -> list[dict]:
    return query(
        "SELECT * FROM eval_runs WHERE user_id = ? ORDER BY id DESC", (user_id,)
    )


def list_eval_results(run_id: int) -> list[dict]:
    return query("SELECT * FROM eval_results WHERE run_id = ? ORDER BY id ASC", (run_id,))


def delete_eval_run(run_id: int, user_id: int) -> None:
    execute("DELETE FROM eval_runs WHERE id = ? AND user_id = ?", (run_id, user_id))
