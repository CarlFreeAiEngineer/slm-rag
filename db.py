"""
db.py -- rag.db schema owner and sqlite-vec setup.

All settings are function arguments; no environment variables. Opens in WAL mode
with synchronous=NORMAL so the frequent readers (web + CLI polling) never block the
single writer.
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
    messages    -- chat transcript; each row carries the chat session id that
                   groups the conversation
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
    # Carries session_id and request_id -- the two correlation IDs described in
    # the README logging section.
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
    # The incoming work queue; kind covers the RAG-specific operations
    # (ingest, retrieve, chat).
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
    # Singleton row tracking which model is resident/loading.
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


def knn_chunks_with_score(
    conn: sqlite3.Connection,
    query_vec: list[float],
    k: int = 5,
    scope: str | None = None,
) -> list[tuple[int, float]]:
    """Return (chunk_id, distance) pairs for the *k* nearest chunks.

    Results are ordered by ascending L2 distance (closest first).

    Parameters
    ----------
    conn       : open connection returned by connect() / init_db()
    query_vec  : 768-dim probe vector (plain Python list of floats)
    k          : number of nearest neighbours to return
    scope      : optional path filter -- when given, only chunks whose
                 document path equals *scope* (exact file match) or starts
                 with *scope* followed by '/' (folder prefix) are returned.
                 The comparison is done after the k-NN using a JOIN on
                 documents.path, which is safe at personal-corpus scale.
    """
    qblob = _serialize(query_vec)

    if scope is None:
        rows = conn.execute(
            """
            SELECT cv.chunk_id, cv.distance
            FROM chunk_vecs cv
            WHERE cv.embedding MATCH ?
              AND k = ?
            ORDER BY distance
            """,
            (qblob, k),
        ).fetchall()
    else:
        # Normalise scope to forward slashes for consistent prefix matching.
        scope_norm = scope.replace("\\", "/").rstrip("/")
        # Retrieve a larger candidate set so that after filtering by scope we
        # still have up to k results.  Fetching k*10 is a conservative upper
        # bound that covers any realistic corpus layout while remaining fast.
        fetch_k = k * 10
        rows = conn.execute(
            """
            SELECT cv.chunk_id, cv.distance
            FROM chunk_vecs cv
            JOIN chunks c   ON c.id  = cv.chunk_id
            JOIN documents d ON d.id = c.doc_id
            WHERE cv.embedding MATCH ?
              AND k = ?
              AND (
                    d.path = ?
                    OR d.path LIKE ? ESCAPE '\\'
                  )
            ORDER BY distance
            """,
            (qblob, fetch_k, scope_norm, scope_norm + "/%"),
        ).fetchall()
        # Trim to k after scope filter (sqlite-vec applies k before the JOIN
        # filter, so rows may be fewer than k -- take what we have up to k).
        rows = rows[:k]

    return [(row[0], row[1]) for row in rows]


# ---------------------------------------------------------------------------
# Message helpers (P6 -- chat transcript)
# ---------------------------------------------------------------------------

def insert_message(
    conn: sqlite3.Connection,
    session_id: str,
    role: str,
    content: str,
    request_id: str | None = None,
    status: str = 'done',
    n_tokens: int | None = None,
    gen_ms: int | None = None,
) -> int:
    """Insert one chat message row.  Returns the new message id.

    Parameters
    ----------
    session_id : conversation grouping key
    role       : 'user' | 'assistant'
    content    : message text (may be partial for streaming rows)
    request_id : ties this message to a requests row (optional)
    status     : 'streaming' | 'done' | 'error'
    n_tokens   : token count (filled when generation finishes)
    gen_ms     : wall-clock ms for generation (filled when generation finishes)
    """
    ts = _now()
    cur = conn.execute(
        "INSERT INTO messages(session_id, request_id, role, content, ts, status, "
        "n_tokens, gen_ms) VALUES(?,?,?,?,?,?,?,?)",
        (session_id, request_id, role, content, ts, status, n_tokens, gen_ms),
    )
    conn.commit()
    return cur.lastrowid


def update_message(
    conn: sqlite3.Connection,
    message_id: int,
    content: str,
    status: str = 'done',
    n_tokens: int | None = None,
    gen_ms: int | None = None,
) -> None:
    """Update a message row in place (used to grow streaming content and
    mark it done when generation completes)."""
    conn.execute(
        "UPDATE messages SET content=?, status=?, n_tokens=?, gen_ms=? WHERE id=?",
        (content, status, n_tokens, gen_ms, message_id),
    )
    conn.commit()


def get_messages(
    conn: sqlite3.Connection,
    session_id: str,
) -> list[dict]:
    """Return all messages for *session_id* ordered by insertion time (id ASC).

    Each element is a dict with keys: id, session_id, request_id, role, content,
    ts, status, n_tokens, gen_ms.
    """
    rows = conn.execute(
        "SELECT id, session_id, request_id, role, content, ts, status, "
        "n_tokens, gen_ms FROM messages WHERE session_id=? ORDER BY id",
        (session_id,),
    ).fetchall()
    keys = ('id', 'session_id', 'request_id', 'role', 'content',
            'ts', 'status', 'n_tokens', 'gen_ms')
    return [dict(zip(keys, row)) for row in rows]


# ---------------------------------------------------------------------------
# Request queue helpers (P6 -- work queue)
# ---------------------------------------------------------------------------

def insert_request(
    conn: sqlite3.Connection,
    kind: str,
    content: str,
    request_id: str,
    session_id: str,
) -> int:
    """Insert a new pending request into the work queue.  Returns the row id."""
    ts = _now()
    cur = conn.execute(
        "INSERT INTO requests(kind, content, request_id, session_id, status, "
        "created_ts) VALUES(?,?,?,?,?,?)",
        (kind, content, request_id, session_id, 'pending', ts),
    )
    conn.commit()
    return cur.lastrowid


def mark_request(
    conn: sqlite3.Connection,
    row_id: int,
    status: str,
    error: str | None = None,
) -> None:
    """Transition a request row to *status* ('running'|'done'|'error')."""
    ts = _now() if status in ('done', 'error') else None
    conn.execute(
        "UPDATE requests SET status=?, error=?, done_ts=? WHERE id=?",
        (status, error, ts, row_id),
    )
    conn.commit()


def get_request(
    conn: sqlite3.Connection,
    request_id: str,
) -> dict | None:
    """Fetch a single request by its UUID request_id string.

    Returns a dict with keys: id, kind, content, request_id, session_id,
    status, error, created_ts, done_ts.  Returns None if not found.
    """
    row = conn.execute(
        "SELECT id, kind, content, request_id, session_id, status, error, "
        "created_ts, done_ts FROM requests WHERE request_id=?",
        (request_id,),
    ).fetchone()
    if row is None:
        return None
    keys = ('id', 'kind', 'content', 'request_id', 'session_id',
            'status', 'error', 'created_ts', 'done_ts')
    return dict(zip(keys, row))


def next_pending_request(conn: sqlite3.Connection) -> dict | None:
    """Return the oldest pending chat request, or None if the queue is empty."""
    row = conn.execute(
        "SELECT id, kind, content, request_id, session_id, status, error, "
        "created_ts, done_ts FROM requests "
        "WHERE status='pending' AND kind='chat' ORDER BY id LIMIT 1",
    ).fetchone()
    if row is None:
        return None
    keys = ('id', 'kind', 'content', 'request_id', 'session_id',
            'status', 'error', 'created_ts', 'done_ts')
    return dict(zip(keys, row))


def get_chunk_by_id(
    conn: sqlite3.Connection,
    chunk_id: int,
) -> dict | None:
    """Fetch full chunk metadata for a single chunk_id.

    Returns a dict with keys: id, doc_id, chunk_index, text, char_start,
    char_end, path (document relative path).  Returns None if not found.
    """
    row = conn.execute(
        """
        SELECT c.id, c.doc_id, c.chunk_index, c.text, c.char_start, c.char_end,
               d.path
        FROM chunks c
        JOIN documents d ON d.id = c.doc_id
        WHERE c.id = ?
        """,
        (chunk_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "id":          row[0],
        "doc_id":      row[1],
        "chunk_index": row[2],
        "text":        row[3],
        "char_start":  row[4],
        "char_end":    row[5],
        "path":        row[6],
    }
