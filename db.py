"""
db.py -- rag.db schema owner and sqlite-vec setup.

All settings are function arguments; no environment variables.
Matches the WAL / PRAGMA / connection style of ../merv/serve.py.
"""

import sqlite3
import struct
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def connect(path: str) -> sqlite3.Connection:
    """Open (or create) rag.db at *path*.

    Sets WAL journal mode and NORMAL synchronous for a good write/read
    concurrency tradeoff.  Loads the sqlite-vec extension so the vec0
    virtual table is available on the returned connection.
    """
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    # sqlite-vec ships its own SQLite extension; load it via the Python package
    # so uv can pull the right pre-built wheel for this platform.
    # enable_load_extension must be called before load_extension / sqlite_vec.load.
    import sqlite_vec
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)  # re-disable for safety after loading
    return conn


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_db(path: str) -> sqlite3.Connection:
    """Create every table (idempotent -- all CREATE … IF NOT EXISTS).

    Returns the open connection so callers can reuse it without a second
    connect() call.

    Tables
    ------
    documents   -- one row per ingested file; status tracks the pipeline stage
    chunks      -- text segments produced by the chunker; carries char offsets
                   for citation rendering
    chunk_vecs  -- vec0 virtual table: 768-dim float32 embeddings keyed to
                   chunk.id.  Embedder identity (model name + version) is
                   stored alongside each vector so stale embeddings can be
                   detected when the model changes.
    messages    -- chat transcript (mirrors merv's messages table); each row
                   carries the chat session id that groups the conversation
    requests    -- incoming work queue; status pending -> running -> done/error
    state       -- singleton row: which model is currently active/loading
    corrections -- (question, context, corrected_answer) training examples
    """
    conn = connect(path)

    # -- documents -----------------------------------------------------------
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            path        TEXT    NOT NULL UNIQUE,   -- relative path under ragdocs/
            status      TEXT    NOT NULL DEFAULT 'pending',
                                                   -- pending | vectorizing | ready | error
            n_chunks    INTEGER,                   -- filled in after chunking
            error_msg   TEXT,                      -- set on status='error'
            ts          TEXT    NOT NULL           -- ISO-8601 UTC, time of insertion
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)"
    )

    # -- chunks --------------------------------------------------------------
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,   -- 0-based position within the document
            text        TEXT    NOT NULL,
            char_start  INTEGER NOT NULL,   -- byte offset into the extracted plain-text
            char_end    INTEGER NOT NULL,
            ts          TEXT    NOT NULL    -- ISO-8601 UTC, time of insertion
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, chunk_index)"
    )

    # -- chunk_vecs (vec0 virtual table) -------------------------------------
    # Stores 768-dim float32 embeddings produced by nomic-embed-text v1.5.
    # embedder_id records the model name/version so we can detect and rebuild
    # vectors when the embedding model changes (README: "embeddings are versioned").
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vecs USING vec0(
            chunk_id    INTEGER PRIMARY KEY,
            embedding   FLOAT[768]
        )
    """)
    # embedder_id is metadata, not stored in vec0 itself (vec0 only holds
    # numeric columns).  Keep it in a companion table keyed to chunk_id.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunk_vec_meta (
            chunk_id    INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
            embedder_id TEXT NOT NULL   -- e.g. 'nomic-embed-text-v1.5'
        )
    """)

    # -- messages (chat transcript) ------------------------------------------
    # Mirrors merv's messages table; adds session_id and request_id for the
    # two correlation IDs described in the README logging section.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT    NOT NULL,   -- groups messages of one conversation
            request_id  TEXT,               -- ties this row to a requests row
            role        TEXT    NOT NULL,   -- 'user' | 'assistant'
            content     TEXT    NOT NULL,
            ts          TEXT    NOT NULL,
            status      TEXT    NOT NULL DEFAULT 'done',
                                            -- streaming | done | error
            n_tokens    INTEGER,            -- filled in when generation finishes
            gen_ms      INTEGER             -- wall-clock ms for generation
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id)"
    )

    # -- requests (work queue) -----------------------------------------------
    # Same shape as merv's requests table; kind expands to cover RAG-specific
    # operations (ingest, retrieve, chat).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            kind        TEXT    NOT NULL,   -- 'chat' | 'ingest' | 'retrieve'
            content     TEXT,               -- JSON payload
            request_id  TEXT,               -- UUID, ties log lines together
            session_id  TEXT,               -- chat session this request belongs to
            status      TEXT    NOT NULL DEFAULT 'pending',
                                            -- pending | running | done | error | cancelled
            error       TEXT,
            created_ts  TEXT    NOT NULL,
            done_ts     TEXT
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_requests_status ON requests(status, id)"
    )

    # -- state (singleton) ---------------------------------------------------
    # Mirrors merv's state table; tracks which model is resident/loading.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS state (
            id              INTEGER PRIMARY KEY CHECK (id = 1),
            active_model    TEXT,   -- name of the currently loaded model
            loading_model   TEXT    -- name of the model currently being loaded
        )
    """)
    conn.execute(
        "INSERT OR IGNORE INTO state(id, active_model, loading_model) VALUES(1, NULL, NULL)"
    )

    # -- corrections (training examples) ------------------------------------
    conn.execute("""
        CREATE TABLE IF NOT EXISTS corrections (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            question        TEXT    NOT NULL,
            retrieved_ctx   TEXT    NOT NULL,   -- the context that was retrieved
            corrected_ans   TEXT    NOT NULL,   -- the human-supplied correct answer
            session_id      TEXT,               -- which session this came from
            ts              TEXT    NOT NULL
        )
    """)

    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Helper: ISO-8601 UTC timestamp
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Vector serialisation helpers
# ---------------------------------------------------------------------------

def _serialize(vec: list[float]) -> bytes:
    """Pack a list of floats as little-endian float32 bytes (sqlite-vec format)."""
    return struct.pack(f"{len(vec)}f", *vec)


# ---------------------------------------------------------------------------
# Chunk + vector insertion
# ---------------------------------------------------------------------------

def insert_chunk(
    conn: sqlite3.Connection,
    doc_id: int,
    chunk_index: int,
    text: str,
    char_start: int,
    char_end: int,
    embedding: list[float],
    embedder_id: str = "nomic-embed-text-v1.5",
) -> int:
    """Insert one chunk and its embedding.  Returns the new chunk id.

    Inserts into three tables atomically (within the caller's transaction, or
    auto-committed if the caller does not manage a transaction):
      chunks          -- text + offsets
      chunk_vecs      -- 768-dim float32 embedding
      chunk_vec_meta  -- embedder identity
    """
    ts = _now()
    cur = conn.execute(
        "INSERT INTO chunks(doc_id, chunk_index, text, char_start, char_end, ts) "
        "VALUES(?,?,?,?,?,?)",
        (doc_id, chunk_index, text, char_start, char_end, ts),
    )
    chunk_id = cur.lastrowid

    conn.execute(
        "INSERT INTO chunk_vecs(chunk_id, embedding) VALUES(?, ?)",
        (chunk_id, _serialize(embedding)),
    )
    conn.execute(
        "INSERT INTO chunk_vec_meta(chunk_id, embedder_id) VALUES(?, ?)",
        (chunk_id, embedder_id),
    )
    return chunk_id


# ---------------------------------------------------------------------------
# k-NN vector query
# ---------------------------------------------------------------------------

def knn_chunks(
    conn: sqlite3.Connection,
    query_vec: list[float],
    k: int = 5,
    doc_id: int | None = None,
) -> list[int]:
    """Return the ids of the *k* nearest chunks to *query_vec*.

    Results are ordered by ascending L2 distance (closest first).

    Parameters
    ----------
    conn       : open connection returned by connect() / init_db()
    query_vec  : 768-dim probe vector (plain Python list of floats)
    k          : number of nearest neighbours to return
    doc_id     : when given, restrict results to chunks from that document
    """
    qblob = _serialize(query_vec)

    if doc_id is None:
        rows = conn.execute(
            """
            SELECT cv.chunk_id
            FROM chunk_vecs cv
            WHERE cv.embedding MATCH ?
              AND k = ?
            ORDER BY distance
            """,
            (qblob, k),
        ).fetchall()
    else:
        # Scope to a single document: join after the k-NN so sqlite-vec still
        # uses its brute-force index, then filter.  For personal-corpus scale
        # this is plenty fast.
        rows = conn.execute(
            """
            SELECT cv.chunk_id
            FROM chunk_vecs cv
            JOIN chunks c ON c.id = cv.chunk_id
            WHERE cv.embedding MATCH ?
              AND k = ?
              AND c.doc_id = ?
            ORDER BY distance
            """,
            (qblob, k, doc_id),
        ).fetchall()

    return [row[0] for row in rows]
