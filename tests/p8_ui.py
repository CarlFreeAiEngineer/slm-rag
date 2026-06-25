# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "playwright",
#   "sqlite-vec",
# ]
# ///
"""
tests/p8_ui.py -- P8 acceptance test: web UI smoke test.

Run with:
    bin\\uv.exe run tests\\p8_ui.py

Strategy:
  1. Try to run a headless Playwright/Chromium test (the preferred path).
     - Install Chromium if it is not already present (playwright install chromium).
     - Start serve.py on a free port with a temp DB.
     - Open the page in headless Chromium.
     - Upload samples/sample.md via the file input (click-to-upload fallback).
     - Wait until the tree row shows 'ready' (poll the DOM).
     - Type a K-means question and wait for the assistant answer to render.
     - Assert a [Source: ...] citation element is present and clickable.
  2. If Playwright or Chromium cannot be installed/launched, fall back to a
     non-browser check that still proves the wiring:
     - Serve the page, fetch index.html and assert it contains expected elements.
     - Exercise the same endpoints the UI calls (ingest -> tree ready -> enqueue
       -> request -> history) via urllib, asserting a cited answer comes back.
     - Log clearly that the visual/browser path was not driven.

Either way: print PASS/FAIL per check, exit 0 only on success, clean up temp db
+ subprocess.

All timeouts are generous: model download + load + generation can take minutes.
"""

import sys
import os
import json
import socket
import subprocess
import tempfile
import time
import shutil
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UV_EXE    = os.path.join(BASE_DIR, 'bin', 'uv.exe')
if not os.path.isfile(UV_EXE):
    UV_EXE = os.path.join(BASE_DIR, 'bin', 'uv')

SERVE_PY  = os.path.join(BASE_DIR, 'serve.py')
SAMPLE_MD = os.path.join(BASE_DIR, 'samples', 'sample.md')

# Timeouts
BOOT_TIMEOUT_S   = 30 * 60   # 30 min (model download + load)
INGEST_TIMEOUT_S = 10 * 60   # 10 min (chunking + embedding)
ANSWER_TIMEOUT_S = 12 * 60   # 12 min (retrieval + generation)
POLL_INTERVAL_S  = 3.0
HEARTBEAT_EVERY  = 15.0

IN_CORPUS_QUESTION = (
    "How does K-means clustering assign data points to clusters, "
    "and what happens when the assignments no longer change?"
)
IN_CORPUS_EXPECTED = 'k-means'

# ── Helpers ────────────────────────────────────────────────────────────────────

checks_passed = 0
checks_failed = 0


def check(label, cond, detail=''):
    global checks_passed, checks_failed
    if cond:
        checks_passed += 1
        print(f'  PASS  {label}')
    else:
        checks_failed += 1
        print(f'  FAIL  {label}' + (f': {detail}' if detail else ''))


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def wait_health(base_url, timeout=BOOT_TIMEOUT_S):
    """Poll /health until both backends report ready. Returns True on success."""
    deadline   = time.time() + timeout
    last_hb    = time.time()
    print(f'  [boot] waiting for /health at {base_url} (up to {timeout//60} min) ...', flush=True)
    while time.time() < deadline:
        try:
            with urlopen(base_url + '/health', timeout=5) as r:
                data = json.loads(r.read())
                if data.get('status') == 'ok':
                    print('  [boot] server ready', flush=True)
                    return True
        except Exception:
            pass
        now = time.time()
        if now - last_hb >= HEARTBEAT_EVERY:
            print(f'  [boot] still waiting... ({int(deadline - now)}s left)', flush=True)
            last_hb = now
        time.sleep(POLL_INTERVAL_S)
    return False


def http_get(base_url, path, timeout=15):
    try:
        with urlopen(base_url + path, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}
    except Exception as e:
        return None, {'error': str(e)}


def http_post(base_url, path, payload, timeout=30):
    data = json.dumps(payload).encode()
    req  = Request(base_url + path, data=data,
                   headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}
    except Exception as e:
        return None, {'error': str(e)}


def http_post_multipart(base_url, path, field_name, filename, file_bytes,
                        content_type='application/octet-stream', timeout=60):
    """POST a single-file multipart/form-data."""
    import email.generator
    import io
    boundary = '----FormBoundary' + str(int(time.time() * 1000))
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
        f'Content-Type: {content_type}\r\n'
        f'\r\n'
    ).encode() + file_bytes + f'\r\n--{boundary}--\r\n'.encode()

    req = Request(
        base_url + path,
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
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
    except Exception as e:
        return None, {'error': str(e)}


def wait_tree_ready(base_url, filename, timeout=INGEST_TIMEOUT_S):
    """Poll /tree until the file's status is 'ready'. Returns the entry or None."""
    deadline = time.time() + timeout
    last_hb  = time.time()
    print(f'  [ingest] polling /tree for {filename!r} to become ready ...', flush=True)
    while time.time() < deadline:
        _, data = http_get(base_url, '/tree')
        for f in (data.get('files') or []):
            if f.get('path') == filename and f.get('status') == 'ready':
                return f
        now = time.time()
        if now - last_hb >= HEARTBEAT_EVERY:
            statuses = {f['path']: f['status']
                        for f in (data.get('files') or [])}
            print(f'  [ingest] still waiting... ({int(deadline-now)}s left) statuses={statuses}', flush=True)
            last_hb = now
        time.sleep(POLL_INTERVAL_S)
    return None


def wait_request_done(base_url, rid, timeout=ANSWER_TIMEOUT_S):
    """Poll /request?id=<rid> until done or error. Returns the final status dict."""
    deadline = time.time() + timeout
    last_hb  = time.time()
    print(f'  [answer] polling /request?id={rid[:8]}... (up to {timeout//60} min) ...', flush=True)
    while time.time() < deadline:
        _, data = http_get(base_url, f'/request?id={rid}')
        status = data.get('status', '')
        if status in ('done', 'error'):
            return data
        now = time.time()
        if now - last_hb >= HEARTBEAT_EVERY:
            print(f'  [answer] still waiting... ({int(deadline-now)}s left) status={status!r}', flush=True)
            last_hb = now
        time.sleep(POLL_INTERVAL_S)
    return None


# ── Non-browser fallback test ──────────────────────────────────────────────────

def run_fallback_test(base_url, reason):
    """
    Pure-urllib smoke test that exercises every endpoint the browser UI calls.
    Prints PASS/FAIL per check. Returns True if all passed.
    """
    print()
    print('=' * 64)
    print('[p8] FALLBACK MODE (non-browser) -- reason:', reason)
    print('[p8] Exercising the same API endpoints the UI calls via urllib.')
    print('[p8] Visual/browser path was NOT driven.')
    print('=' * 64)

    # 1. Fetch index.html and assert it contains key UI elements
    print('\n[check] index.html content')
    status, _ = http_get(base_url, '/')
    # Actually fetch as text
    try:
        with urlopen(base_url + '/', timeout=10) as r:
            html = r.read().decode('utf-8', 'replace')
            html_ok = True
    except Exception as e:
        html = ''
        html_ok = False
    check('GET / returns 200', html_ok)
    check('index.html contains drop-zone', 'drop-zone' in html)
    check('index.html contains chat-messages', 'chat-messages' in html)
    check('index.html contains /ingest call', '/ingest' in html)
    check('index.html contains /enqueue call', '/enqueue' in html)
    check('index.html contains /tree call', '/tree' in html)
    check('index.html contains /request call', '/request' in html)
    check('index.html contains /history call', '/history' in html)
    check('index.html contains /clear call', '/clear' in html)
    check('index.html contains citation rendering (renderCitations)', 'renderCitations' in html)
    check('index.html contains fix-it control (/correct wired, P9)', '/correct' in html)
    check('index.html notes /correct lands in P9', 'P9' in html)

    # 2. GET /tree -- should succeed (may be empty)
    print('\n[check] GET /tree')
    code, tree = http_get(base_url, '/tree')
    check('GET /tree returns 200', code == 200, str(tree))
    check('GET /tree has files key', 'files' in (tree or {}))

    # 3. POST /ingest with sample.md
    print('\n[check] POST /ingest (multipart)')
    with open(SAMPLE_MD, 'rb') as f:
        sample_bytes = f.read()
    code, resp = http_post_multipart(base_url, '/ingest', 'file',
                                     'sample.md', sample_bytes,
                                     'text/markdown', timeout=60)
    check('POST /ingest returns 200', code == 200, str(resp))
    check('ingest response has doc_id', 'doc_id' in (resp or {}))
    check('ingest response status is vectorizing',
          (resp or {}).get('status') == 'vectorizing')

    # 4. Poll tree until sample.md is ready
    print('\n[check] /tree polls to ready')
    entry = wait_tree_ready(base_url, 'sample.md')
    check('sample.md becomes ready in /tree', entry is not None,
          'timed out' if entry is None else '')
    if entry:
        check('sample.md has chunk_count > 0', (entry.get('chunk_count') or 0) > 0)

    # 5. GET /doc?path=sample.md
    print('\n[check] GET /doc?path=sample.md')
    code, doc = http_get(base_url, '/doc?path=sample.md')
    check('GET /doc returns 200', code == 200, str(doc))
    check('/doc response has chunks', len((doc or {}).get('chunks', [])) > 0)

    # 6. POST /enqueue (chat)
    print('\n[check] POST /enqueue (chat question)')
    code, enq = http_post(base_url, '/enqueue', {
        'kind':    'chat',
        'content': IN_CORPUS_QUESTION,
    })
    check('POST /enqueue returns 200', code == 200, str(enq))
    rid = (enq or {}).get('request_id')
    sid = (enq or {}).get('session_id')
    check('enqueue response has request_id', bool(rid))
    check('enqueue response has session_id', bool(sid))

    if not rid:
        print('[p8] Cannot continue without request_id.')
        return checks_failed == 0

    # 7. Poll /request until done
    print('\n[check] Polling /request until done')
    done = wait_request_done(base_url, rid)
    check('request reaches done status', done is not None and done.get('status') == 'done',
          str(done))

    # 8. GET /history and assert cited answer
    print('\n[check] GET /history (cited answer)')
    code, hist = http_get(base_url, f'/history?session_id={sid}')
    check('GET /history returns 200', code == 200)
    msgs = (hist or {}).get('messages', [])
    asst = [m for m in msgs if m.get('role') == 'assistant' and m.get('status') == 'done']
    check('history contains assistant message', len(asst) > 0)
    if asst:
        answer = asst[-1].get('content', '')
        check('answer mentions k-means',
              IN_CORPUS_EXPECTED.lower() in answer.lower(), answer[:120])
        check('answer contains [Source: ...] citation',
              '[Source:' in answer, answer[:200])

    # 9. POST /clear
    print('\n[check] POST /clear')
    code, clr = http_post(base_url, '/clear', {'session_id': sid})
    check('POST /clear returns 200', code == 200, str(clr))
    check('clear response ok', (clr or {}).get('ok') is True)

    return checks_failed == 0


# ── Playwright browser test ────────────────────────────────────────────────────

def try_install_playwright():
    """Attempt to install Chromium via playwright. Returns True on success."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'playwright', 'install', 'chromium',
             '--with-deps'],
            capture_output=True, text=True, timeout=300
        )
        return result.returncode == 0
    except Exception:
        return False


def run_playwright_test(base_url):
    """
    Headless Chromium smoke test.
    Returns True if all checks passed, False if anything failed,
    and raises ImportError / RuntimeError if Playwright is unavailable.
    """
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(headless=True)
        except Exception as e:
            raise RuntimeError(f'chromium launch failed: {e}') from e

        page = browser.new_page()
        page.goto(base_url + '/', wait_until='domcontentloaded', timeout=15000)

        # 1. Page loaded
        print('\n[check] Page load')
        title = page.title()
        check('page title is slm-rag', title == 'slm-rag', repr(title))
        check('drop-zone is present',
              page.query_selector('#drop-zone') is not None)
        check('chat-messages is present',
              page.query_selector('#chat-messages') is not None)

        # 2. Upload sample.md via the file input (click-to-upload fallback)
        print('\n[check] File upload via file input')
        page.set_input_files('#file-input', SAMPLE_MD)
        # The JS change handler calls uploadFile which calls /ingest

        # 3. Poll the DOM until the tree row shows 'ready'
        print('\n[check] Tree row becomes ready')
        deadline_ms = INGEST_TIMEOUT_S * 1000
        try:
            # Wait for a .badge-ready element to appear in the tree
            page.wait_for_selector('.badge-ready', timeout=deadline_ms)
            ready_ok = True
        except PWTimeout:
            ready_ok = False
        check('tree row shows ready badge', ready_ok)

        # 4. Type a question and submit
        print('\n[check] Chat: send question')
        page.fill('#chat-input', IN_CORPUS_QUESTION)
        page.click('#send-btn')

        # 5. Wait for the assistant message to render (long timeout)
        print('\n[check] Chat: wait for answer')
        answer_deadline_ms = ANSWER_TIMEOUT_S * 1000
        try:
            page.wait_for_selector('.msg-assistant', timeout=answer_deadline_ms)
            # Wait until the thinking indicator is gone (answer fully rendered)
            page.wait_for_function(
                "() => !document.getElementById('thinking-indicator')",
                timeout=answer_deadline_ms,
            )
            answer_rendered = True
        except PWTimeout:
            answer_rendered = False
        check('assistant answer rendered', answer_rendered)

        # 6. Assert a [Source: ...] citation span is present and clickable
        print('\n[check] Citation element')
        citation_el = page.query_selector('.citation')
        check('[Source: ...] citation element present', citation_el is not None)

        if citation_el:
            # Click the citation and verify preview overlay opens
            citation_el.click()
            try:
                page.wait_for_selector('.preview-overlay.open', timeout=5000)
                overlay_opened = True
            except PWTimeout:
                overlay_opened = False
            check('clicking citation opens doc preview', overlay_opened)

            # Close the overlay
            close_btn = page.query_selector('#preview-close')
            if close_btn:
                close_btn.click()

        # 7. Verify fix-it control is present
        print('\n[check] Fix-it control')
        fix_it = page.query_selector('.fix-it-btn')
        check('fix-it button present on assistant message', fix_it is not None)

        browser.close()

    return checks_failed == 0


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    port = free_port()
    base_url = f'http://127.0.0.1:{port}'

    tmp_dir = tempfile.mkdtemp(prefix='slm_p8_')
    db_path = os.path.join(tmp_dir, 'p8_test.db')

    print(f'[p8] starting serve.py --port {port} --db {db_path}')
    proc = subprocess.Popen(
        [UV_EXE, 'run', SERVE_PY, '--port', str(port), '--db', db_path],
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    def cleanup():
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print('[p8] cleaned up temp db and server process')

    # Wait for server to be healthy
    healthy = wait_health(base_url, timeout=BOOT_TIMEOUT_S)
    check('server /health reports ok', healthy)
    if not healthy:
        print('[p8] server did not become healthy; aborting')
        cleanup()
        sys.exit(1)

    # ── Try Playwright first ───────────────────────────────────────────────────
    playwright_reason = None
    try:
        import playwright
    except ImportError:
        playwright_reason = 'playwright not importable'

    if playwright_reason is None:
        # Try to ensure Chromium is installed
        print('[p8] checking Playwright Chromium installation ...')
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'playwright', 'install', 'chromium'],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                playwright_reason = f'playwright install chromium failed: {result.stdout[-500:]}'
        except Exception as e:
            playwright_reason = f'playwright install failed: {e}'

    if playwright_reason is None:
        print('[p8] Running headless Playwright/Chromium test ...')
        try:
            run_playwright_test(base_url)
            pw_used = True
        except (ImportError, RuntimeError, Exception) as e:
            playwright_reason = str(e)
            pw_used = False
    else:
        pw_used = False

    if not pw_used:
        run_fallback_test(base_url, playwright_reason or 'unknown')

    # ── Summary ────────────────────────────────────────────────────────────────
    cleanup()
    print()
    print('=' * 64)
    print(f'[p8] {"PLAYWRIGHT" if pw_used else "FALLBACK (non-browser)"} test')
    print(f'[p8] {checks_passed} passed, {checks_failed} failed')
    print('=' * 64)

    sys.exit(0 if checks_failed == 0 else 1)


if __name__ == '__main__':
    main()
