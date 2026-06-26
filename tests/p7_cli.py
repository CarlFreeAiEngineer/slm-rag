# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
tests/p7_cli.py -- P7 acceptance test: CLI REPL.

Run with:
    bin\\uv.exe run tests\\p7_cli.py

What it checks:
  1. Start serve.py --cli on a free port with a temp db.
  2. Wait for /health to report both backends ready (web side is up even in
     --cli mode because the web server starts before the REPL).
  3. POST /ingest samples/sample.md over HTTP; poll /tree until 'ready'.
  4. Drive the REPL via stdin:
     a. Write a known-answer question about K-means (from sample.md).
     b. Wait (generous timeout, ~3 min) until accumulated stdout contains
        the expected fact AND a [Source: ...] citation.
  5. Write /clear via stdin; verify via HTTP GET /history?session_id= that
     the transcript was cleared (empty messages list).
  6. Write /quit; assert the process exits cleanly (code 0) within 10 s.

Uses a temp db.  Cleans up on exit.
Exit 0 only if all checks pass; non-zero otherwise.

Implementation note: serve.py --cli blocks the main thread in cli_repl()
while the web server runs in a daemon thread.  We drive it by writing to
stdin (text mode, line-buffered) and reading from stdout via a background
accumulator thread.  /quit causes cli_repl() to call begin_shutdown() which
stops the HTTP server, so the process exits with code 0.
"""

import sys
import os
import json
import re
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

# ── Timeouts ─────────────────────────────────────────────────────────────────
BOOT_TIMEOUT_S   = 30 * 60   # 30 min (model load; weights already cached)
INGEST_TIMEOUT_S = 10 * 60   # 10 min (chunk + embed a short .md file)
ANSWER_TIMEOUT_S = 10 * 60   # 10 min (retrieval + generation)
POLL_INTERVAL_S  = 3.0
HEARTBEAT_EVERY  = 15.0

# Known-answer question -- K-means is discussed in sample.md
IN_CORPUS_QUESTION = (
    "How does K-means clustering assign data points to clusters, "
    "and what happens when the assignments no longer change?"
)
# The model's answer must contain this substring (case-insensitive)
IN_CORPUS_EXPECTED = 'k-means'


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
    boundary = '----P7TestBoundary'
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
    """Poll /health until both backends report ready."""
    last_status = {}
    last_hb     = time.time()
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f'serve.py exited early (code {proc.returncode})')
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
    raise TimeoutError(f'backends not ready within {BOOT_TIMEOUT_S}s')


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


def _wait_stdout_contains(stdout_lines_lock, stdout_lines, pattern, deadline,
                          label=''):
    """Block until accumulated stdout contains *pattern* (case-insensitive)."""
    last_hb = time.time()
    pat = re.compile(pattern, re.IGNORECASE)
    while time.time() < deadline:
        with stdout_lines_lock:
            combined = '\n'.join(stdout_lines)
        if pat.search(combined):
            return True
        now = time.time()
        if now - last_hb >= HEARTBEAT_EVERY:
            remaining = int(deadline - now)
            print(f'  [stdout-wait{" " + label if label else ""}] '
                  f'still waiting ({remaining}s left) ...',
                  flush=True)
            last_hb = now
        time.sleep(POLL_INTERVAL_S)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    port = _free_port()
    base = f'http://127.0.0.1:{port}'

    tmpdir  = tempfile.mkdtemp(prefix='p7_test_')
    db_path = os.path.join(tmpdir, 'test.db')

    print(f'[test] tmpdir:  {tmpdir}', flush=True)
    print(f'[test] db_path: {db_path}', flush=True)
    print(f'[test] port:    {port}', flush=True)

    cmd = [UV_EXE, 'run', SERVE_PY, '--cli', '--port', str(port),
           '--db', db_path]
    print(f'[test] command: {" ".join(cmd)}', flush=True)

    # Start the process with stdin=PIPE so we can drive the REPL, and
    # stdout=PIPE so we can accumulate its output.  text=True + line-buffered
    # so writes/reads are immediately visible to each side.
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,          # line-buffered
        encoding='utf-8',
        errors='replace',
        cwd=BASE_DIR,
        env=dict(os.environ),
    )

    # Accumulate stdout in a background thread so we can search it at any time.
    stdout_lines      = []
    stdout_lines_lock = threading.Lock()

    def _drain():
        for line in iter(proc.stdout.readline, ''):
            line = line.rstrip('\n').rstrip('\r')
            with stdout_lines_lock:
                stdout_lines.append(line)
            print(f'  [serve] {line}', flush=True)

    drain_th = threading.Thread(target=_drain, daemon=True)
    drain_th.start()

    # We need to know the session_id the CLI minted so we can query /history
    # after /clear.  We'll extract it from the stdout once the REPL is live.
    cli_session_id = None

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

        # Extract the session id the CLI printed at startup:
        # "[cli] Session: <8chars>..."
        # We need it for the /history verification after /clear.
        # We wait a short while for the REPL prompt to appear.
        time.sleep(3)
        with stdout_lines_lock:
            combined = '\n'.join(stdout_lines)
        m = re.search(r'\[cli\] Session:\s+([0-9a-f-]+)', combined)
        if m:
            # The line only prints the first 8 chars; we need the full id.
            # Read it from /history by listing all messages for any session --
            # but we don't have one yet, so instead we'll capture it from the
            # enqueue response later.  For now just note that the REPL started.
            print(f'  [cli] REPL started (session prefix: {m.group(1)})',
                  flush=True)

        # ── Check 2: ingest sample.md over HTTP ───────────────────────────────
        print('\n[test] ingesting sample.md over HTTP ...', flush=True)
        with open(SAMPLE_MD, 'rb') as f:
            md_bytes = f.read()

        code, resp = _api_post_multipart(base, '/ingest', 'sample.md', md_bytes,
                                         timeout=30)
        if not check('POST /ingest returns 200', code == 200,
                     f'got {code}: {resp}'):
            return

        ingest_path     = resp.get('path', 'sample.md')
        ingest_deadline = time.time() + INGEST_TIMEOUT_S
        try:
            tree_entry = _poll_tree_ready(base, ingest_deadline, ingest_path)
            check('sample.md ready in /tree', True,
                  f'chunk_count={tree_entry["chunk_count"]}')
        except Exception as e:
            check('sample.md ready in /tree', False, str(e))
            return

        # ── Check 3: drive the REPL with a known-answer question ─────────────
        print(f'\n[test] writing question to REPL stdin: '
              f'{IN_CORPUS_QUESTION!r}', flush=True)
        proc.stdin.write(IN_CORPUS_QUESTION + '\n')
        proc.stdin.flush()

        ans_deadline = time.time() + ANSWER_TIMEOUT_S

        # Wait until a [Source: ...] citation appears in stdout.  The REPL
        # prints the full answer text (with inline citations) before returning
        # to the prompt, so the citation appearing means the answer is complete.
        # We check for the citation first because it necessarily comes AFTER
        # the answer text, so if the citation is present the fact is too.
        got_citation = _wait_stdout_contains(
            stdout_lines_lock, stdout_lines,
            r'\[Source:[^\]]*\]',
            ans_deadline,
            label='citation marker',
        )
        check(
            'CLI stdout contains a [Source: ...] citation',
            got_citation,
            'scanned accumulated stdout for citation marker',
        )

        # Check for the expected fact in the same accumulated output.
        with stdout_lines_lock:
            combined_after = '\n'.join(stdout_lines)
        got_fact = IN_CORPUS_EXPECTED.lower() in combined_after.lower()
        check(
            f'CLI stdout contains "{IN_CORPUS_EXPECTED}"',
            got_fact,
            'scanned accumulated stdout for expected fact',
        )

        # Grab the session id from the enqueue call.  The REPL uses POST
        # /enqueue internally; the session id is printed in the REPL output as
        # "[cli] Session: <prefix>..." only at startup.  Since we can't see the
        # actual UUID from stdout alone, we use a small trick: POST /enqueue
        # directly via HTTP using the same session prefix, which won't work --
        # instead we read it from a follow-up HTTP GET /history check.
        # The cleanest approach: after the answer appears, read /history for
        # ALL sessions is not possible; instead we capture the session id from
        # the serve.py stdout which prints the session id in the worker log:
        # "[worker] processing request_id=... session_id=..."
        worker_m = re.search(
            r'\[worker\] processing request_id=\S+ session_id=([0-9a-f-]+)',
            combined_after,
        )
        if worker_m:
            cli_session_id = worker_m.group(1)
            print(f'  [test] captured session_id={cli_session_id[:8]}...',
                  flush=True)

        # ── Check 4: /clear via REPL stdin; verify via HTTP /history ─────────
        print('\n[test] writing /clear to REPL stdin ...', flush=True)

        # Snapshot stdout length before /clear so we can detect the new output.
        with stdout_lines_lock:
            lines_before_clear = len(stdout_lines)

        proc.stdin.write('/clear\n')
        proc.stdin.flush()

        # Wait for the CLI to acknowledge the clear in stdout.
        clear_deadline = time.time() + 30
        cleared_printed = _wait_stdout_contains(
            stdout_lines_lock, stdout_lines,
            r'Transcript cleared',
            clear_deadline,
            label='/clear confirmation',
        )
        check(
            'CLI prints "Transcript cleared" after /clear',
            cleared_printed,
            'scanned stdout for confirmation message',
        )

        # Verify via HTTP: the OLD session should now have zero messages.
        if cli_session_id:
            h_code, h_resp = _api_get(
                base, f'/history?session_id={cli_session_id}', timeout=10
            )
            messages_after_clear = (
                h_resp.get('messages', []) if h_code == 200 else None
            )
            check(
                'GET /history for old session returns empty list after /clear',
                h_code == 200 and isinstance(messages_after_clear, list)
                and len(messages_after_clear) == 0,
                f'code={h_code} messages={messages_after_clear}',
            )
        else:
            # We couldn't capture the session id; skip the HTTP history check
            # but note it.
            print('  [test] WARNING: could not capture session_id from stdout; '
                  'skipping /history verification', flush=True)
            check(
                'GET /history for old session returns empty list after /clear',
                False,
                'session_id not captured from stdout (worker log line not found)',
            )

        # ── Check 5: /quit exits the process cleanly ──────────────────────────
        print('\n[test] writing /quit to REPL stdin ...', flush=True)
        proc.stdin.write('/quit\n')
        proc.stdin.flush()

        try:
            exit_code = proc.wait(timeout=30)
            check(
                'process exits cleanly after /quit',
                exit_code == 0,
                f'exit code={exit_code}',
            )
        except subprocess.TimeoutExpired:
            check(
                'process exits cleanly after /quit',
                False,
                'process did not exit within 30s after /quit',
            )

    finally:
        # ── Cleanup ───────────────────────────────────────────────────────────
        print('\n[test] cleaning up ...', flush=True)
        if proc.poll() is None:
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
