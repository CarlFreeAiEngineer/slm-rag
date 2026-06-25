# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "sqlite-vec",
# ]
# ///
"""
tests/p4_ingest.py -- P4 acceptance test: ingestion pipeline.

Run with:
    bin\\uv.exe run tests\\p4_ingest.py

What it checks:
  1. Server starts and /health reports both backends ready.
  2. POST /ingest (multipart) with samples/sample.md -> 200, returns doc_id + status.
  3. GET /tree polls until the file shows status='ready' with chunk_count > 1 (timeout 10 min).
  4. GET /doc?path=sample.md returns the stored text (chunks list non-empty).
  5. Open rag.db directly: documents row status='ready', n_chunks matches tree's chunk_count.
  6. Number of chunks rows in DB matches chunk_count.
  7. Each chunk has a stored vector in chunk_vecs (non-zero length blob).

Uses a temp rag.db (via --db flag) and a temp ragdocs dir to keep the test hermetic.
The test cleans up the temp dir and db file on exit.

Exit 0 if all checks pass; non-zero otherwise.
"""

import sys
import os
import json
import math
import socket
import sqlite3
import struct
import subprocess
import tempfile
import threading
import time
import shutil
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UV_EXE     = os.path.join(BASE_DIR, 'bin', 'uv.exe')
if not os.path.isfile(UV_EXE):
    UV_EXE = os.path.join(BASE_DIR, 'bin', 'uv')

SERVE_PY   = os.path.join(BASE_DIR, 'serve.py')
SAMPLE_MD  = os.path.join(BASE_DIR, 'samples', 'sample.md')

# Timeouts -- first run may need to download weights and load two models
BOOT_TIMEOUT_S   = 30 * 60   # 30 min: weight download + two model loads
INGEST_TIMEOUT_S = 10 * 60   # 10 min: embedding all chunks (CPU-only)
POLL_INTERVAL_S  = 3.0
HEARTBEAT_EVERY  = 15.0


# ─────────────────────────────────────────────────────────────────────────────
# Result tracking
# ─────────────────────────────────────────────────────────────────────────────

_results = []   # list of (label, passed, detail)


def check(label, passed, detail=''):
    _results.append((label, passed, detail))
    mark = 'PASS' if passed else 'FAIL'
    msg  = f'[{mark}] {label}'
    if detail:
        msg += f' -- {detail}'
    print(msg, flush=True)
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _api_get(base, path_qs, timeout=10):
    with urlopen(base + path_qs, timeout=timeout) as r:
        return r.status, json.loads(r.read())


def _api_post_json(base, path, payload, timeout=120):
    req = Request(base + path,
                  data=json.dumps(payload).encode(),
                  headers={'Content-Type': 'application/json'},
                  method='POST')
    try:
        with urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}


def _api_post_multipart(base, path, filename, file_bytes, timeout=120):
    """Upload file_bytes as multipart/form-data with field name 'file'."""
    boundary = '----PY4TestBoundary'
    fname    = os.path.basename(filename)

    # Build multipart body by hand (no external deps needed)
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'
        f'Content-Type: application/octet-stream\r\n'
        f'\r\n'
    ).encode('utf-8') + file_bytes + f'\r\n--{boundary}--\r\n'.encode('utf-8')

    req = Request(
        base + path,
        data=body,
        headers={
            'Content-Type': f'multipart/form-data; boundary={boundary}',
            'Content-Length': str(len(body)),
        },
        method='POST',
    )
    try:
        with urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}


def _wait_health(base, deadline, proc):
    """Poll /health until both backends report ready=True, or raise."""
    last_status = {}
    last_hb     = time.time()
    while time.time() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else b''
            raise RuntimeError(
                f'serve.py exited early (code {proc.returncode}):\n'
                + out.decode('utf-8', 'replace')[-2000:])
        try:
            _, h = _api_get(base, '/health', timeout=5)
            bk   = h.get('backends', {})
            e_ok = bk.get('embedder', {}).get('ready', False)
            g_ok = bk.get('phi',      {}).get('ready', False)
            if h != last_status:
                print(f'  [health] status={h.get("status")} '
                      f'embed={e_ok} gen={g_ok}', flush=True)
                last_status = h
            if e_ok and g_ok:
                return
        except (URLError, OSError):
            pass
        now = time.time()
        if now - last_hb >= HEARTBEAT_EVERY:
            print(f'  [wait] still waiting ({int(deadline - now)}s left) ...', flush=True)
            last_hb = now
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f'backends not ready in {BOOT_TIMEOUT_S}s')


def _poll_tree_ready(base, deadline, rel_path):
    """Poll GET /tree until rel_path shows status='ready' with chunk_count > 1.
    Returns the file entry dict, or raises TimeoutError."""
    last_hb = time.time()
    while time.time() < deadline:
        try:
            _, resp = _api_get(base, '/tree', timeout=10)
            files = resp.get('files', [])
            for entry in files:
                if entry.get('path') == rel_path:
                    status = entry.get('status')
                    count  = entry.get('chunk_count', 0)
                    print(f'  [tree] {rel_path}: status={status} chunk_count={count}',
                          flush=True)
                    if status == 'ready' and count > 1:
                        return entry
                    if status == 'error':
                        raise RuntimeError(f'Document entered error state: {entry}')
        except (URLError, OSError) as e:
            print(f'  [tree] poll error: {e}', flush=True)
        now = time.time()
        if now - last_hb >= HEARTBEAT_EVERY:
            print(f'  [tree] still vectorizing ... ({int(deadline - now)}s left)',
                  flush=True)
            last_hb = now
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f'{rel_path} not ready after {INGEST_TIMEOUT_S}s')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    port = _free_port()
    base = f'http://127.0.0.1:{port}'

    # ── Temp workspace (hermetic: own db) ─────────────────────────────────────
    tmpdir  = tempfile.mkdtemp(prefix='p4_test_')
    db_path = os.path.join(tmpdir, 'test.db')

    print(f'[test] tmpdir:   {tmpdir}', flush=True)
    print(f'[test] db_path:  {db_path}', flush=True)
    print(f'[test] port:     {port}', flush=True)

    cmd = [UV_EXE, 'run', SERVE_PY, '--port', str(port), '--db', db_path]
    print(f'[test] command:  {" ".join(cmd)}', flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=BASE_DIR,
        env=dict(os.environ),   # pass full env (PATH etc.) so uv can find everything
    )

    # Drain stdout in a background thread so we see progress
    _log_lines = []
    _log_lock  = threading.Lock()

    def _drain():
        for raw in iter(proc.stdout.readline, b''):
            line = raw.decode('utf-8', 'replace').rstrip()
            with _log_lock:
                _log_lines.append(line)
            print(f'  [serve] {line}', flush=True)

    drain_th = threading.Thread(target=_drain, daemon=True)
    drain_th.start()

    try:
        # ── Wait for backends to be ready ─────────────────────────────────────
        deadline = time.time() + BOOT_TIMEOUT_S
        time.sleep(5)
        try:
            _wait_health(base, deadline, proc)
            check('server became ready', True)
        except Exception as e:
            check('server became ready', False, str(e))
            return   # can't run further checks

        print('[test] both backends ready -- proceeding with ingest checks', flush=True)

        # ── Check 1: POST /ingest with multipart upload ───────────────────────
        with open(SAMPLE_MD, 'rb') as f:
            md_bytes = f.read()

        print(f'[test] uploading {SAMPLE_MD} ({len(md_bytes)} bytes) ...', flush=True)
        code, resp = _api_post_multipart(base, '/ingest', 'sample.md', md_bytes,
                                         timeout=30)
        check('POST /ingest returns 200', code == 200, f'got {code}: {resp}')
        doc_id = resp.get('doc_id')
        ingest_status = resp.get('status')
        check('ingest response has doc_id', isinstance(doc_id, int),
              f'doc_id={doc_id!r}')
        check('ingest response status is vectorizing', ingest_status == 'vectorizing',
              f'status={ingest_status!r}')
        ingest_rel_path = resp.get('path', 'sample.md')

        # ── Check 2: poll GET /tree until ready ───────────────────────────────
        print(f'[test] polling /tree for {ingest_rel_path!r} to become ready '
              f'(up to {INGEST_TIMEOUT_S // 60} min) ...', flush=True)
        ingest_deadline = time.time() + INGEST_TIMEOUT_S
        try:
            tree_entry = _poll_tree_ready(base, ingest_deadline, ingest_rel_path)
            tree_chunk_count = tree_entry['chunk_count']
            check('file status is ready in /tree', True,
                  f'chunk_count={tree_chunk_count}')
            check('chunk_count > 1', tree_chunk_count > 1,
                  f'got {tree_chunk_count}')
        except TimeoutError as e:
            check('file status is ready in /tree', False, str(e))
            tree_chunk_count = 0

        # ── Check 3: GET /doc?path=sample.md returns text ─────────────────────
        doc_code, doc_resp = _api_get(base, f'/doc?path={ingest_rel_path}', timeout=10)
        check('GET /doc returns 200', doc_code == 200, f'got {doc_code}: {doc_resp}')
        doc_chunks = doc_resp.get('chunks', [])
        check('GET /doc returns non-empty chunks list', len(doc_chunks) > 0,
              f'{len(doc_chunks)} chunks')
        if doc_chunks:
            check('first chunk has text', bool(doc_chunks[0].get('text', '').strip()),
                  repr(doc_chunks[0].get('text', '')[:80]))

        # ── Check 4: verify rag.db directly ───────────────────────────────────
        # Open read-only (uri mode) so we don't disturb serve.py's WAL connection
        try:
            import sqlite_vec
            conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True,
                                   check_same_thread=False)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)

            doc_row = conn.execute(
                "SELECT id, status, n_chunks, byte_size, ext FROM documents WHERE path=?",
                (ingest_rel_path,)
            ).fetchone()

            if doc_row is None:
                check('documents row exists in DB', False,
                      f'no row for path={ingest_rel_path!r}')
                db_doc_id = None
                db_n_chunks = 0
            else:
                db_doc_id, db_status, db_n_chunks, db_byte_size, db_ext = doc_row
                check('documents row exists in DB', True,
                      f'doc_id={db_doc_id}')
                check("documents status is 'ready' in DB", db_status == 'ready',
                      f'status={db_status!r}')
                check('DB n_chunks matches tree chunk_count',
                      db_n_chunks == tree_chunk_count,
                      f'db={db_n_chunks} tree={tree_chunk_count}')

                # Verify blob was stored: byte_size must match the uploaded byte count
                check('documents row has byte_size == uploaded bytes',
                      db_byte_size == len(md_bytes),
                      f'db_byte_size={db_byte_size} uploaded={len(md_bytes)}')
                # Verify ext is set
                check('documents row has ext set',
                      bool(db_ext),
                      f'ext={db_ext!r}')

                # Count actual chunks rows
                n_chunk_rows = conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE doc_id=?", (db_doc_id,)
                ).fetchone()[0]
                check('chunks row count matches n_chunks',
                      n_chunk_rows == db_n_chunks,
                      f'rows={n_chunk_rows} n_chunks={db_n_chunks}')

                # Verify each chunk has a vector in chunk_vecs
                chunk_ids = [r[0] for r in conn.execute(
                    "SELECT id FROM chunks WHERE doc_id=?", (db_doc_id,)
                ).fetchall()]

                vec_count = conn.execute(
                    "SELECT COUNT(*) FROM chunk_vecs WHERE chunk_id IN ({})".format(
                        ','.join('?' * len(chunk_ids))
                    ),
                    chunk_ids
                ).fetchone()[0] if chunk_ids else 0

                check('every chunk has a vector in chunk_vecs',
                      vec_count == len(chunk_ids),
                      f'vec_count={vec_count} chunk_count={len(chunk_ids)}')

                # Verify embedder_id is recorded in chunk_vec_meta
                meta_count = conn.execute(
                    "SELECT COUNT(*) FROM chunk_vec_meta WHERE chunk_id IN ({})".format(
                        ','.join('?' * len(chunk_ids))
                    ),
                    chunk_ids
                ).fetchone()[0] if chunk_ids else 0
                check('every chunk has embedder_id in chunk_vec_meta',
                      meta_count == len(chunk_ids),
                      f'meta_count={meta_count}')

                # Spot-check one vector: deserialise and verify it is 768 finite floats
                one_vec_blob = conn.execute(
                    "SELECT embedding FROM chunk_vecs WHERE chunk_id=?",
                    (chunk_ids[0],)
                ).fetchone()[0] if chunk_ids else None

                if one_vec_blob:
                    n_floats = len(one_vec_blob) // 4
                    floats   = struct.unpack(f'{n_floats}f', one_vec_blob)
                    check('stored vector is 768 floats', n_floats == 768,
                          f'got {n_floats}')
                    all_finite = all(math.isfinite(v) for v in floats)
                    check('stored vector contains finite floats', all_finite)
                    nonzero = any(v != 0.0 for v in floats)
                    check('stored vector is non-zero', nonzero)
                else:
                    check('stored vector blob is present', False)

            conn.close()

        except ImportError:
            print('[test] sqlite_vec not importable for direct DB check -- skipping',
                  flush=True)
            check('DB direct check (sqlite_vec available)', False,
                  'sqlite_vec not importable in test process')
        except Exception as e:
            check('DB direct check', False, str(e))

        # ── Check 5: delete the document via POST /delete ─────────────────────
        print(f'[test] deleting {ingest_rel_path!r} via POST /delete ...', flush=True)
        del_code, del_resp = _api_post_json(base, '/delete',
                                            {'path': ingest_rel_path}, timeout=10)
        check('POST /delete returns 200', del_code == 200,
              f'got {del_code}: {del_resp}')
        check('POST /delete response ok=true', del_resp.get('ok') is True,
              f'resp={del_resp}')
        check('POST /delete response deleted=true', del_resp.get('deleted') is True,
              f'resp={del_resp}')

        # File must no longer appear in /tree
        _, tree_after = _api_get(base, '/tree', timeout=10)
        paths_after = [e.get('path') for e in tree_after.get('files', [])]
        check('file absent from /tree after delete',
              ingest_rel_path not in paths_after,
              f'paths={paths_after}')

        # Direct DB check: 0 chunks and 0 chunk_vecs after delete
        try:
            import sqlite_vec
            conn2 = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True,
                                    check_same_thread=False)
            conn2.enable_load_extension(True)
            sqlite_vec.load(conn2)
            conn2.enable_load_extension(False)

            doc_gone = conn2.execute(
                "SELECT id FROM documents WHERE path=?", (ingest_rel_path,)
            ).fetchone()
            check('documents row gone after delete', doc_gone is None,
                  f'row={doc_gone}')

            # db_doc_id from above; if it was set, verify chunks and vecs are gone
            if db_doc_id is not None:
                remaining_chunks = conn2.execute(
                    "SELECT COUNT(*) FROM chunks WHERE doc_id=?", (db_doc_id,)
                ).fetchone()[0]
                check('0 chunks remain after delete', remaining_chunks == 0,
                      f'remaining={remaining_chunks}')

                # chunk_vecs check using the chunk_ids we gathered earlier
                if chunk_ids:
                    remaining_vecs = conn2.execute(
                        "SELECT COUNT(*) FROM chunk_vecs WHERE chunk_id IN ({})".format(
                            ','.join('?' * len(chunk_ids))
                        ),
                        chunk_ids
                    ).fetchone()[0]
                    check('0 chunk_vecs remain after delete', remaining_vecs == 0,
                          f'remaining={remaining_vecs}')

            conn2.close()
        except Exception as e:
            check('DB delete verification', False, str(e))

        # A second delete of the same path should return 404 / deleted=false
        del2_code, del2_resp = _api_post_json(base, '/delete',
                                              {'path': ingest_rel_path}, timeout=10)
        check('second POST /delete returns 404', del2_code == 404,
              f'got {del2_code}: {del2_resp}')
        check('second POST /delete deleted=false', del2_resp.get('deleted') is False,
              f'resp={del2_resp}')

    finally:
        # ── Clean up ──────────────────────────────────────────────────────────
        print('[test] shutting down server ...', flush=True)
        try:
            _api_post_json(base, '/shutdown', {}, timeout=10)
        except Exception:
            pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        print('[test] removing temp dir ...', flush=True)
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception as e:
            print(f'[test] cleanup warning: {e}', flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '=' * 60, flush=True)
    print('[test] RESULTS:', flush=True)
    passed = 0
    failed = 0
    for label, ok, detail in _results:
        mark = 'PASS' if ok else 'FAIL'
        line = f'  [{mark}] {label}'
        if detail:
            line += f'\n         {detail}'
        print(line, flush=True)
        if ok:
            passed += 1
        else:
            failed += 1

    print(f'\n  {passed} passed, {failed} failed', flush=True)
    print('=' * 60, flush=True)

    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
