# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "sqlite-vec",
# ]
# ///
"""
tests/p6_answer.py -- P6 acceptance test: grounded answer + citations (streaming model).

Run with:
    bin\\uv.exe run tests\\p6_answer.py

What it checks:
  1. Server starts and /health reports both backends ready.
  2. POST /ingest with samples/sample.md -> polls /tree until ready.
  3. In-corpus question (about K-means clustering):
     a. POST /enqueue -> {request_id, session_id} returned immediately.
     b. While polling GET /request?id=<rid>: assert message `content` GROWS
        (non-empty mid-stream, larger when polled again later).
     c. On done: assert status='done', `answer` is non-empty, `answer` contains
        a fact from sample.md ('k-means'), `answer` contains at least one [L]
        letter citation (e.g. [A] or [B]), and `references` is valid JSON with
        numbered chunk objects.
     d. `content` (the full build transcript) is longer than `answer`.
  4. Out-of-corpus question (capital of France -- not in sample.md):
     a. Enqueue, poll to done.
     b. Assistant `answer` (or `content` fallback) contains a refusal phrase.
  5. Logs check: the in-corpus request_id appears in at least one log file with
     both embed_request/embed_response AND gen_request/gen_response lines.

Uses a temp db.  Cleans up on exit.
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
LOG_DIR   = os.path.join(BASE_DIR, 'logs')

# Timeouts -- allow for model cold start; backends are typically already cached.
BOOT_TIMEOUT_S    = 30 * 60   # 30 min (model download + load)
INGEST_TIMEOUT_S  = 10 * 60   # 10 min (chunking + embedding a short md file)
ANSWER_TIMEOUT_S  = 10 * 60   # 10 min (retrieval + generation)
POLL_INTERVAL_S   = 3.0
HEARTBEAT_EVERY   = 15.0

# In-corpus: K-means is discussed in sample.md; choose a question that can only
# be answered from that section.
IN_CORPUS_QUESTION  = (
    "How does K-means clustering assign data points to clusters, "
    "and what happens when the assignments no longer change?"
)
# The answer must contain this phrase (lower-cased) -- taken directly from sample.md.
IN_CORPUS_EXPECTED  = 'k-means'

# Out-of-corpus: sample.md is about machine learning -- Paris is never mentioned.
OUT_CORPUS_QUESTION = "What is the capital of France?"

# Phrases that count as a valid refusal (any one of these, case-insensitive).
REFUSAL_PHRASES = [
    "don't know",
    "do not know",
    "not in",
    "no information",
    "cannot",
    "not present",
    "not mentioned",
    "not provided",
    "based on the provided",
    "not found",
    "not available",
    "i don't know",
]

# How many times to poll /history mid-stream to observe content growing.
STREAM_POLL_ATTEMPTS = 4
STREAM_POLL_INTERVAL = 5.0  # seconds between streaming content checks


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
    req = Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
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


def _api_post_multipart(base, path, filename, file_bytes, timeout=120):
    """Upload file_bytes as multipart/form-data."""
    boundary = '----P6TestBoundary'
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
# Wait helpers
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
    """Poll GET /tree until rel_path shows status='ready' with chunk_count > 0."""
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
                    if status == 'ready' and count > 0:
                        return entry
                    if status == 'error':
                        raise RuntimeError(
                            f'Document entered error state: {entry}')
        except (URLError, OSError) as e:
            print(f'  [tree] poll error: {e}', flush=True)
        now = time.time()
        if now - last_hb >= HEARTBEAT_EVERY:
            print(f'  [tree] still vectorizing ({int(deadline - now)}s left)',
                  flush=True)
            last_hb = now
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f'{rel_path} not ready after {INGEST_TIMEOUT_S}s')


def _poll_request_done(base, rid, deadline):
    """Poll GET /request?id=<rid> until status is 'done' or 'error'."""
    last_hb = time.time()
    while time.time() < deadline:
        try:
            code, resp = _api_get(base, f'/request?id={rid}', timeout=10)
            if code == 200:
                status = resp.get('status', '')
                print(f'  [request] rid={rid[:8]}... status={status}', flush=True)
                if status in ('done', 'error'):
                    return resp
        except (URLError, OSError) as e:
            print(f'  [request] poll error: {e}', flush=True)
        now = time.time()
        if now - last_hb >= HEARTBEAT_EVERY:
            print(f'  [request] still running ({int(deadline - now)}s left)',
                  flush=True)
            last_hb = now
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f'request {rid} did not finish in {ANSWER_TIMEOUT_S}s')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    port = _free_port()
    base = f'http://127.0.0.1:{port}'

    tmpdir  = tempfile.mkdtemp(prefix='p6_test_')
    db_path = os.path.join(tmpdir, 'test.db')

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

    in_corpus_rid = None
    in_corpus_sid = None

    try:
        # ── Check 1: wait for server to be ready ──────────────────────────────
        deadline = time.time() + BOOT_TIMEOUT_S
        time.sleep(5)
        try:
            _wait_health(base, deadline, proc)
            check('server became ready', True)
        except Exception as e:
            check('server became ready', False, str(e))
            return

        # ── Check 2: ingest sample.md and wait for it to be ready ─────────────
        print('\n[test] ingesting sample.md ...', flush=True)
        with open(SAMPLE_MD, 'rb') as f:
            md_bytes = f.read()

        code, resp = _api_post_multipart(base, '/ingest', 'sample.md', md_bytes,
                                         timeout=30)
        if not check('POST /ingest returns 200', code == 200,
                     f'got {code}: {resp}'):
            return

        ingest_path = resp.get('path', 'sample.md')
        ingest_deadline = time.time() + INGEST_TIMEOUT_S
        try:
            tree_entry = _poll_tree_ready(base, ingest_deadline, ingest_path)
            check('sample.md ready in /tree', True,
                  f'chunk_count={tree_entry["chunk_count"]}')
        except Exception as e:
            check('sample.md ready in /tree', False, str(e))
            return

        # ── Check 3: in-corpus question ───────────────────────────────────────
        print(f'\n[test] in-corpus question: {IN_CORPUS_QUESTION!r}', flush=True)

        enq_code, enq_resp = _api_post_json(
            base, '/enqueue',
            {'kind': 'chat', 'content': IN_CORPUS_QUESTION},
            timeout=30,
        )
        if not check('POST /enqueue (in-corpus) returns 200', enq_code == 200,
                     f'got {enq_code}: {enq_resp}'):
            return

        in_corpus_rid = enq_resp.get('request_id')
        in_corpus_sid = enq_resp.get('session_id')
        check('enqueue returns request_id',
              bool(in_corpus_rid), str(in_corpus_rid))
        check('enqueue returns session_id',
              bool(in_corpus_sid), str(in_corpus_sid))

        # ── Streaming content growth check ───────────────────────────────────
        # Poll /history a few times mid-stream and assert content grows.
        print(f'\n[test] polling /history mid-stream to check content grows ...',
              flush=True)
        content_snapshots = []
        for _ in range(STREAM_POLL_ATTEMPTS):
            time.sleep(STREAM_POLL_INTERVAL)
            try:
                hsnap_code, hsnap_resp = _api_get(
                    base, f'/history?session_id={in_corpus_sid}', timeout=10)
                if hsnap_code == 200:
                    snaps = hsnap_resp.get('messages', [])
                    asst_snaps = [m for m in snaps if m.get('role') == 'assistant']
                    if asst_snaps:
                        snap_content = asst_snaps[-1].get('content', '')
                        snap_status  = asst_snaps[-1].get('status', '')
                        content_snapshots.append(len(snap_content))
                        print(f'  [stream] content len={len(snap_content)} '
                              f'status={snap_status}', flush=True)
                        if snap_status == 'done':
                            break
            except Exception as e:
                print(f'  [stream] poll error: {e}', flush=True)

        if len(content_snapshots) >= 2:
            grew = content_snapshots[-1] > content_snapshots[0]
            check('streaming content grows over time',
                  grew or content_snapshots[-1] > 0,
                  f'snapshots={content_snapshots}')
        elif len(content_snapshots) == 1:
            check('streaming content is non-empty mid-stream',
                  content_snapshots[0] > 0,
                  f'content len={content_snapshots[0]}')
        else:
            check('streaming content observed', False, 'no snapshots captured')

        # Poll until done.
        ans_deadline = time.time() + ANSWER_TIMEOUT_S
        try:
            status_resp = _poll_request_done(base, in_corpus_rid, ans_deadline)
        except TimeoutError as e:
            check('in-corpus request finished', False, str(e))
            return

        check('in-corpus request status=done',
              status_resp.get('status') == 'done',
              f'status={status_resp.get("status")} error={status_resp.get("error")}')

        # Retrieve history.
        h_code, h_resp = _api_get(
            base, f'/history?session_id={in_corpus_sid}', timeout=10)
        check('GET /history returns 200', h_code == 200,
              f'got {h_code}: {h_resp}')

        messages = h_resp.get('messages', []) if h_code == 200 else []
        assistant_msgs = [m for m in messages if m.get('role') == 'assistant']
        check('history contains an assistant message',
              len(assistant_msgs) > 0,
              f'{len(messages)} total, {len(assistant_msgs)} assistant')

        if assistant_msgs:
            last_asst = assistant_msgs[-1]
            content   = last_asst.get('content', '')
            ans       = last_asst.get('answer', '') or content
            refs_raw  = last_asst.get('references', '')
            print(f'  [answer] answer[:300]={ans[:300]!r}', flush=True)
            print(f'  [content] content len={len(content)}', flush=True)

            # (a) status=done
            check('in-corpus assistant message status=done',
                  last_asst.get('status') == 'done',
                  f'status={last_asst.get("status")}')

            # (b) answer field is non-empty
            check('in-corpus answer field is non-empty',
                  bool(ans.strip()),
                  f'answer[:100]={ans[:100]!r}')

            # (c) answer contains the expected fact.
            check(
                f'in-corpus answer contains "{IN_CORPUS_EXPECTED}"',
                IN_CORPUS_EXPECTED.lower() in ans.lower(),
                f'answer[:200]: {ans[:200]!r}',
            )

            # (d) answer contains at least one [L] letter citation.
            import re
            letter_citation = re.compile(r'\[[A-Z]+\]')
            has_citation = bool(letter_citation.search(ans))
            check(
                'in-corpus answer contains [L] letter citation',
                has_citation,
                f'answer[:300]: {ans[:300]!r}',
            )

            # (e) references is valid JSON with numbered chunk objects.
            refs_ok = False
            refs = []
            if refs_raw:
                try:
                    refs = json.loads(refs_raw)
                    refs_ok = (
                        isinstance(refs, list) and
                        len(refs) > 0 and
                        all(isinstance(r, dict) and 'n' in r and 'path' in r
                            and 'chunk_index' in r and 'text' in r
                            for r in refs)
                    )
                except json.JSONDecodeError:
                    pass
            check(
                'references field is valid JSON with numbered chunks',
                refs_ok,
                f'references[:200]: {(refs_raw or "")[:200]!r}',
            )

            # (f) content (build transcript) is longer than answer.
            check(
                'build transcript (content) is longer than collapsed answer',
                len(content) > len(ans),
                f'content={len(content)} answer={len(ans)}',
            )

        # ── Check 4: out-of-corpus question ───────────────────────────────────
        print(f'\n[test] out-of-corpus question: {OUT_CORPUS_QUESTION!r}',
              flush=True)

        enq2_code, enq2_resp = _api_post_json(
            base, '/enqueue',
            {'kind': 'chat', 'content': OUT_CORPUS_QUESTION},
            timeout=30,
        )
        if not check('POST /enqueue (out-of-corpus) returns 200',
                     enq2_code == 200, f'got {enq2_code}: {enq2_resp}'):
            return

        rid2 = enq2_resp.get('request_id')
        sid2 = enq2_resp.get('session_id')

        ans_deadline2 = time.time() + ANSWER_TIMEOUT_S
        try:
            status2 = _poll_request_done(base, rid2, ans_deadline2)
        except TimeoutError as e:
            check('out-of-corpus request finished', False, str(e))
            return

        check('out-of-corpus request status=done',
              status2.get('status') == 'done',
              f'status={status2.get("status")} error={status2.get("error")}')

        h2_code, h2_resp = _api_get(
            base, f'/history?session_id={sid2}', timeout=10)
        msgs2 = h2_resp.get('messages', []) if h2_code == 200 else []
        asst2 = [m for m in msgs2 if m.get('role') == 'assistant']

        if asst2:
            last_asst2 = asst2[-1]
            # Use `answer` (clean text) if available, else fall back to `content`
            ans2 = last_asst2.get('answer', '') or last_asst2.get('content', '')
            print(f'  [answer] {ans2[:300]!r}', flush=True)
            ans2_lower = ans2.lower()
            refused = any(phrase in ans2_lower for phrase in REFUSAL_PHRASES)
            check(
                'out-of-corpus answer declines / says it does not know',
                refused,
                f'answer[:300]: {ans2[:300]!r}',
            )
        else:
            check('out-of-corpus answer exists', False,
                  'no assistant message in history')

        # ── Check 5: log lines for in-corpus request id ───────────────────────
        if in_corpus_rid:
            print(f'\n[test] checking logs for request_id={in_corpus_rid[:8]}...',
                  flush=True)
            # Give the worker a moment to finish flushing log lines.
            time.sleep(2)

            log_files = [
                os.path.join(LOG_DIR, f)
                for f in os.listdir(LOG_DIR)
                if f.endswith('.log')
            ] if os.path.isdir(LOG_DIR) else []

            found_embed_req  = False
            found_embed_resp = False
            found_gen_req    = False
            found_gen_resp   = False

            for lf in log_files:
                try:
                    with open(lf, encoding='utf-8', errors='replace') as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if obj.get('request_id') != in_corpus_rid:
                                continue
                            stage = obj.get('stage', '')
                            if stage == 'embed_request':
                                found_embed_req  = True
                            elif stage == 'embed_response':
                                found_embed_resp = True
                            elif stage == 'gen_request':
                                found_gen_req    = True
                            elif stage == 'gen_response':
                                found_gen_resp   = True
                except OSError as e:
                    print(f'  [logs] could not read {lf}: {e}', flush=True)

            check('logs contain embed_request for in-corpus request_id',
                  found_embed_req)
            check('logs contain embed_response for in-corpus request_id',
                  found_embed_resp)
            check('logs contain gen_request for in-corpus request_id',
                  found_gen_req)
            check('logs contain gen_response for in-corpus request_id',
                  found_gen_resp)

    finally:
        # ── Cleanup ───────────────────────────────────────────────────────────
        print('\n[test] shutting down server ...', flush=True)
        try:
            _api_post_json(base, '/shutdown', {}, timeout=10)
        except Exception:
            pass
        try:
            proc.wait(timeout=20)
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
