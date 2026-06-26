# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "sqlite-vec",
# ]
# ///
"""
tests/p5_retrieve.py -- P5 acceptance test: retrieval (question -> chunks).

Run with:
    bin\\uv.exe run tests\\p5_retrieve.py

What it checks:
  1. Server starts and /health reports both backends ready.
  2. POST /ingest (multipart) with samples/sample.md -> 200, status=vectorizing.
  3. GET /tree polls until status='ready' (reuse P4's poll pattern).
  4. POST /retrieve with a question whose answer is in a known chunk:
       "How does K-means clustering assign points to clusters?"
     The K-means paragraph in sample.md is a well-isolated passage; the top-1
     hit must contain the phrase 'k-means' (case-insensitive).
  5. Response is HTTP 200 with a non-empty 'hits' list.
  6. Top-1 hit contains the expected passage (K-means).
  7. Every hit carries the required citation fields:
       path, chunk_index, char_start, char_end, text, distance.
  8. scope=sample.md returns at least one hit (the K-means chunk is in it).
  9. scope pointing at a non-existent path returns zero hits.

Uses a temp db and temp ragdocs dir (--db flag).  Cleans up on exit.
Exit 0 if all checks pass; non-zero otherwise.
"""

import sys
import os
import json
import socket
import subprocess
import tempfile
import threading
import time
import shutil
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UV_EXE    = os.path.join(BASE_DIR, 'bin', 'uv.exe')
if os.name != 'nt':
    UV_EXE = os.path.join(BASE_DIR, 'bin', 'uv.mac' if sys.platform == 'darwin' else 'uv.linux')

SERVE_PY  = os.path.join(BASE_DIR, 'serve.py')
SAMPLE_MD = os.path.join(BASE_DIR, 'samples', 'sample.md')

# Timeouts -- allow for weight download + two-model cold start on first run
BOOT_TIMEOUT_S   = 30 * 60   # 30 min
INGEST_TIMEOUT_S = 10 * 60   # 10 min
POLL_INTERVAL_S  = 3.0
HEARTBEAT_EVERY  = 15.0

# The question to ask and the substring that must appear in the top-1 hit.
# K-means is discussed in a single, well-isolated paragraph in sample.md.
RETRIEVE_QUESTION  = "How does K-means clustering assign points to clusters?"
EXPECTED_SUBSTRING = "k-means"   # case-insensitive check


# ─────────────────────────────────────────────────────────────────────────────
# Result tracking
# ─────────────────────────────────────────────────────────────────────────────

_results = []


def check(label, passed, detail=''):
    _results.append((label, passed, detail))
    mark = 'PASS' if passed else 'FAIL'
    msg  = f'[{mark}] {label}'
    if detail:
        msg += f' -- {detail}'
    print(msg, flush=True)
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers (stdlib only)
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
    """Upload file_bytes as multipart/form-data."""
    boundary = '----P5TestBoundary'
    fname    = os.path.basename(filename)
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


# ─────────────────────────────────────────────────────────────────────────────
# Wait helpers (same pattern as P4)
# ─────────────────────────────────────────────────────────────────────────────

def _wait_health(base, deadline, proc):
    """Poll /health until both backends report ready, or raise."""
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
            print(f'  [wait] still waiting ({int(deadline - now)}s left) ...',
                  flush=True)
            last_hb = now
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f'backends not ready in {BOOT_TIMEOUT_S}s')


def _poll_tree_ready(base, deadline, rel_path):
    """Poll GET /tree until rel_path shows status='ready' with chunk_count > 1."""
    last_hb = time.time()
    while time.time() < deadline:
        try:
            _, resp = _api_get(base, '/tree', timeout=10)
            for entry in resp.get('files', []):
                if entry.get('path') == rel_path:
                    status = entry.get('status')
                    count  = entry.get('chunk_count', 0)
                    print(f'  [tree] {rel_path}: status={status} '
                          f'chunk_count={count}', flush=True)
                    if status == 'ready' and count > 1:
                        return entry
                    if status == 'error':
                        raise RuntimeError(
                            f'Document entered error state: {entry}')
        except (URLError, OSError) as e:
            print(f'  [tree] poll error: {e}', flush=True)
        now = time.time()
        if now - last_hb >= HEARTBEAT_EVERY:
            print(f'  [tree] still vectorizing ... '
                  f'({int(deadline - now)}s left)', flush=True)
            last_hb = now
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f'{rel_path} not ready after {INGEST_TIMEOUT_S}s')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    port = _free_port()
    base = f'http://127.0.0.1:{port}'

    tmpdir  = tempfile.mkdtemp(prefix='p5_test_')
    db_path = os.path.join(tmpdir, 'test.db')
    ragdocs_dir = os.path.join(tmpdir, 'ragdocs')
    os.makedirs(ragdocs_dir, exist_ok=True)

    print(f'[test] tmpdir:  {tmpdir}', flush=True)
    print(f'[test] db_path: {db_path}', flush=True)
    print(f'[test] port:    {port}', flush=True)

    cmd = [UV_EXE, 'run', SERVE_PY, '--port', str(port), '--db', db_path]
    print(f'[test] command: {" ".join(cmd)}', flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=BASE_DIR,
        env=dict(os.environ),
    )

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
        # ── Check 1: wait for server to be ready ─────────────────────────────
        deadline = time.time() + BOOT_TIMEOUT_S
        time.sleep(5)
        try:
            _wait_health(base, deadline, proc)
            check('server became ready', True)
        except Exception as e:
            check('server became ready', False, str(e))
            return

        print('[test] backends ready -- uploading sample.md ...', flush=True)

        # ── Check 2: ingest sample.md ─────────────────────────────────────────
        with open(SAMPLE_MD, 'rb') as f:
            md_bytes = f.read()

        code, resp = _api_post_multipart(base, '/ingest', 'sample.md', md_bytes,
                                         timeout=30)
        check('POST /ingest returns 200', code == 200, f'got {code}: {resp}')
        ingest_path = resp.get('path', 'sample.md')

        # ── Check 3: poll until ready ─────────────────────────────────────────
        print(f'[test] polling /tree for {ingest_path!r} (up to '
              f'{INGEST_TIMEOUT_S // 60} min) ...', flush=True)
        ingest_deadline = time.time() + INGEST_TIMEOUT_S
        try:
            tree_entry = _poll_tree_ready(base, ingest_deadline, ingest_path)
            chunk_count = tree_entry['chunk_count']
            check('file is ready in /tree', True,
                  f'chunk_count={chunk_count}')
        except Exception as e:
            check('file is ready in /tree', False, str(e))
            return   # cannot retrieve from a file that never became ready

        # ── Check 4: POST /retrieve -- unscoped ───────────────────────────────
        print(f'\n[test] retrieving: {RETRIEVE_QUESTION!r}', flush=True)
        ret_code, ret_resp = _api_post_json(
            base, '/retrieve',
            {'question': RETRIEVE_QUESTION, 'k': 5},
            timeout=60,
        )
        check('POST /retrieve returns 200', ret_code == 200,
              f'got {ret_code}: {ret_resp}')

        if ret_code != 200:
            # Cannot perform the remaining hit checks
            return

        hits = ret_resp.get('hits', [])
        check('retrieve response has non-empty hits list', len(hits) > 0,
              f'{len(hits)} hits')

        if not hits:
            return

        # ── Check 5: top-1 hit contains the K-means passage ──────────────────
        top1 = hits[0]
        top1_text_lower = top1.get('text', '').lower()
        check(
            f'top-1 hit contains "{EXPECTED_SUBSTRING}"',
            EXPECTED_SUBSTRING in top1_text_lower,
            f'top-1 text[:120]: {top1.get("text","")[:120]!r}',
        )

        # ── Check 6: every hit carries the required citation fields ───────────
        required_fields = ('path', 'chunk_index', 'char_start',
                           'char_end', 'text', 'distance')
        all_have_fields = all(
            all(f in hit for f in required_fields)
            for hit in hits
        )
        missing = [
            f'hit[{i}] missing: {[f for f in required_fields if f not in hit]}'
            for i, hit in enumerate(hits)
            if not all(f in hit for f in required_fields)
        ]
        check(
            'every hit carries citation fields '
            '(path, chunk_index, char_start, char_end, text, distance)',
            all_have_fields,
            '; '.join(missing) if missing else '',
        )

        # ── Check 7: distance is a finite float, char_start < char_end ───────
        import math
        offsets_ok = all(
            isinstance(h.get('char_start'), int)
            and isinstance(h.get('char_end'), int)
            and h['char_start'] < h['char_end']
            for h in hits
        )
        check('all hits have valid char_start < char_end', offsets_ok,
              str([(h.get('char_start'), h.get('char_end')) for h in hits[:3]]))

        distances_ok = all(
            isinstance(h.get('distance'), (int, float))
            and math.isfinite(h['distance'])
            for h in hits
        )
        check('all hit distances are finite numbers', distances_ok)

        # Distances should be non-decreasing (closest first)
        dists = [h['distance'] for h in hits]
        check('hits are ordered by distance (ascending)',
              all(dists[i] <= dists[i + 1] for i in range(len(dists) - 1)),
              str(dists))

        # ── Check 8: scope=sample.md returns the K-means hit ─────────────────
        print(f'\n[test] retrieving with scope={ingest_path!r} ...', flush=True)
        sc_code, sc_resp = _api_post_json(
            base, '/retrieve',
            {'question': RETRIEVE_QUESTION, 'scope': ingest_path, 'k': 5},
            timeout=60,
        )
        check('scoped retrieve returns 200', sc_code == 200,
              f'got {sc_code}: {sc_resp}')

        sc_hits = sc_resp.get('hits', []) if sc_code == 200 else []
        check('scoped retrieve returns at least one hit', len(sc_hits) > 0,
              f'{len(sc_hits)} hits with scope={ingest_path!r}')

        if sc_hits:
            all_in_scope = all(h.get('path') == ingest_path for h in sc_hits)
            check(
                'all scoped hits belong to the scoped file',
                all_in_scope,
                str([h.get('path') for h in sc_hits]),
            )
            sc_top1_lower = sc_hits[0].get('text', '').lower()
            check(
                f'scoped top-1 hit contains "{EXPECTED_SUBSTRING}"',
                EXPECTED_SUBSTRING in sc_top1_lower,
                f'text[:120]: {sc_hits[0].get("text","")[:120]!r}',
            )

        # ── Check 9: scope pointing at a non-existent path returns zero hits ──
        print('\n[test] retrieving with non-existent scope ...', flush=True)
        nx_code, nx_resp = _api_post_json(
            base, '/retrieve',
            {'question': RETRIEVE_QUESTION, 'scope': 'no_such_folder/phantom.md',
             'k': 5},
            timeout=60,
        )
        check('retrieve with nonexistent scope returns 200', nx_code == 200,
              f'got {nx_code}: {nx_resp}')
        nx_hits = nx_resp.get('hits', []) if nx_code == 200 else ['non-empty']
        check('retrieve with nonexistent scope returns empty hits', len(nx_hits) == 0,
              f'got {len(nx_hits)} hits')

    finally:
        # ── Cleanup ───────────────────────────────────────────────────────────
        print('\n[test] shutting down server ...', flush=True)
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
    passed = failed = 0
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
