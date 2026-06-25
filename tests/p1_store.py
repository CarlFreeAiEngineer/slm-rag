# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sqlite-vec",
# ]
# ///
"""
P1 self-test: SQLite schema + sqlite-vec store.

Creates a temporary rag.db, inserts 3 chunks with hand-crafted 768-dim vectors,
runs a k-NN query, and asserts the returned ids come back in the expected order.
Also confirms that the sqlite-vec extension loaded correctly.

Exit 0 = all checks pass.  Exit 1 = any check failed.
"""

import sys
import os
import math
import struct
import tempfile
import shutil

# -- make db.py importable when run from any cwd ----------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import db  # noqa: E402  (import after path fixup)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pass(msg: str) -> None:
    print(f"  PASS  {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}")


def _vec(hot_index: int, dims: int = 768, scale: float = 1.0) -> list[float]:
    """Return a unit-ish vector with a large spike at *hot_index* so vectors
    are clearly separated.  Similar to a one-hot but with a small background
    so the distance arithmetic is unambiguous."""
    v = [0.01] * dims
    v[hot_index] = scale
    # Normalise so comparisons are purely directional (L2 distance).
    mag = math.sqrt(sum(x * x for x in v))
    return [x / mag for x in v]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

failures = 0


def check(condition: bool, label: str) -> None:
    global failures
    if condition:
        _pass(label)
    else:
        _fail(label)
        failures += 1


def run_tests(db_path: str) -> None:
    # -----------------------------------------------------------------------
    # 1.  init_db creates the schema
    # -----------------------------------------------------------------------
    print("\n[1] init_db / schema creation")
    conn = db.init_db(db_path)

    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','shadow') "
            "AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    }
    for expected in ("documents", "chunks", "chunk_vecs", "chunk_vec_meta",
                     "messages", "requests", "state", "corrections"):
        check(expected in tables, f"table '{expected}' exists  (found: {sorted(tables)})")

    # -----------------------------------------------------------------------
    # 2.  sqlite-vec extension loaded
    # -----------------------------------------------------------------------
    print("\n[2] sqlite-vec extension")
    try:
        row = conn.execute("SELECT vec_version()").fetchone()
        check(row is not None and row[0], f"vec_version() = {row[0] if row else 'N/A'}")
    except Exception as exc:
        check(False, f"vec_version() raised: {exc}")

    # -----------------------------------------------------------------------
    # 3.  Insert a document (needed for FK constraints)
    # -----------------------------------------------------------------------
    print("\n[3] insert document")
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO documents(path, status, ts) VALUES(?,?,?)",
        ("test/sample.txt", "ready", ts),
    )
    doc_id = cur.lastrowid
    conn.commit()
    check(doc_id > 0, f"document inserted with id={doc_id}")

    # -----------------------------------------------------------------------
    # 4.  Insert 3 chunks with clearly separated 768-dim vectors
    #
    #     vec A: spike at index   0   (chunk_index 0)
    #     vec B: spike at index 100   (chunk_index 1)
    #     vec C: spike at index 500   (chunk_index 2)
    #
    #     Probe: spike at index 100 -> should be closest to B, then A or C.
    # -----------------------------------------------------------------------
    print("\n[4] insert 3 chunks with distinct vectors")
    vec_a = _vec(0)
    vec_b = _vec(100)
    vec_c = _vec(500)

    id_a = db.insert_chunk(conn, doc_id, 0, "chunk A text", 0,  50, vec_a)
    id_b = db.insert_chunk(conn, doc_id, 1, "chunk B text", 50, 100, vec_b)
    id_c = db.insert_chunk(conn, doc_id, 2, "chunk C text", 100, 150, vec_c)
    conn.commit()

    check(id_a > 0 and id_b > 0 and id_c > 0,
          f"chunks inserted with ids {id_a}, {id_b}, {id_c}")
    check(id_a != id_b != id_c, "all chunk ids are distinct")

    # -----------------------------------------------------------------------
    # 5.  k-NN query: probe near vec_b -> B must be rank-1, A/C further away
    # -----------------------------------------------------------------------
    print("\n[5] k-NN query (probe near vec_b)")
    probe = _vec(100, scale=0.99)   # slightly different magnitude, same direction
    results = db.knn_chunks(conn, probe, k=3)

    check(len(results) == 3,
          f"knn returned 3 results (got {len(results)}): {results}")
    check(results[0] == id_b,
          f"rank-1 is id_b={id_b} (got {results[0] if results else 'none'})")
    # A and C should both be further; order between them is fine either way.
    check(set(results[1:]) == {id_a, id_c},
          f"rank-2 and rank-3 are {{id_a={id_a}, id_c={id_c}}} (got {results[1:]})")

    # -----------------------------------------------------------------------
    # 6.  k-NN with doc_id scope
    # -----------------------------------------------------------------------
    print("\n[6] k-NN scoped to doc_id")
    scoped = db.knn_chunks(conn, probe, k=2, doc_id=doc_id)
    check(len(scoped) == 2,
          f"scoped knn returned 2 results (got {len(scoped)}): {scoped}")
    check(scoped[0] == id_b,
          f"scoped rank-1 is id_b={id_b} (got {scoped[0] if scoped else 'none'})")

    # -----------------------------------------------------------------------
    # 7.  k=1 query returns only the single closest
    # -----------------------------------------------------------------------
    print("\n[7] k=1 query")
    top1 = db.knn_chunks(conn, probe, k=1)
    check(len(top1) == 1 and top1[0] == id_b,
          f"k=1 returns exactly id_b={id_b} (got {top1})")

    # -----------------------------------------------------------------------
    # 8.  state singleton was created by init_db
    # -----------------------------------------------------------------------
    print("\n[8] state singleton")
    row = conn.execute(
        "SELECT id, active_model, loading_model FROM state WHERE id=1"
    ).fetchone()
    check(row is not None and row[0] == 1,
          f"state row exists with id=1 (got {row})")

    conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    tmpdir = tempfile.mkdtemp(prefix="rag_p1_test_")
    db_path = os.path.join(tmpdir, "rag.db")
    print(f"Test db: {db_path}")
    try:
        run_tests(db_path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"\nCleaned up {tmpdir}")

    print()
    if failures:
        print(f"RESULT: FAIL ({failures} check(s) failed)")
        return 1
    print("RESULT: PASS (all checks passed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
