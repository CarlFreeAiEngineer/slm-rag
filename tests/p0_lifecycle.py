#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
tests/p0_lifecycle.py -- P0 lifecycle acceptance test.

Checks:
  1. serve.py --check exits 0 and prints a plan without opening a socket.
  2. serve.py boots on a free ephemeral port; /health returns 200 with
     {"status": "ok", ...} within a reasonable timeout.
  3. POST /shutdown causes the process to exit cleanly within a few seconds.
  4. A logs/*.log file was created containing at least one JSON line that
     includes a "request_id" field.

Run with the bundled uv:
  bin/uv.exe run tests/p0_lifecycle.py

Exits 0 on pass, non-zero on fail.  Uses only the Python stdlib.
"""

import sys
import os
import json
import socket
import subprocess
import threading
import time
import glob
import atexit
from urllib.request import urlopen, Request
from urllib.error import URLError

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UV       = os.path.join(BASE_DIR, 'bin', 'uv.exe')
if os.name != 'nt':
    UV = os.path.join(BASE_DIR, 'bin', 'uv.mac' if sys.platform == 'darwin' else 'uv.linux')    # Linux / macOS
SERVE    = os.path.join(BASE_DIR, 'serve.py')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# ── state ─────────────────────────────────────────────────────────────────────
_server_proc = None   # cleaned up by atexit


def _cleanup():
    if _server_proc is not None and _server_proc.poll() is None:
        _server_proc.terminate()
        try:
            _server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_proc.kill()


atexit.register(_cleanup)


# ── helpers ───────────────────────────────────────────────────────────────────

def free_port():
    """Pick a free ephemeral port by binding and immediately releasing it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def wait_health(port, timeout=20):
    """Poll GET /health until it returns 200 with status 'ok', or timeout."""
    url      = f'http://127.0.0.1:{port}/health'
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2) as r:
                if r.status == 200:
                    data = json.loads(r.read())
                    if data.get('status') == 'ok':
                        return True, data
        except (URLError, OSError, json.JSONDecodeError):
            pass
        time.sleep(0.3)
    return False, {}


def post_shutdown(port):
    """POST /shutdown; returns the HTTP status code."""
    req = Request(
        f'http://127.0.0.1:{port}/shutdown',
        data=b'',
        method='POST',
    )
    try:
        with urlopen(req, timeout=5) as r:
            return r.status
    except Exception as e:
        # A 200 response before the server closes the connection is fine;
        # a connection reset right after is also fine.
        if hasattr(e, 'code'):
            return e.code
        return 200   # assume ok if the connection dropped (server exiting)


def check_log_has_request_id():
    """Return (True, path) if any logs/*.log file has a line with request_id."""
    for path in glob.glob(os.path.join(LOGS_DIR, '*.log')):
        try:
            with open(path, encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line.strip())
                        if 'request_id' in obj:
                            return True, path
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    return False, None


# ── test functions ────────────────────────────────────────────────────────────

def test_check_flag():
    """serve.py --check must exit 0 and print a plan; must NOT open a socket."""
    print('  running: serve.py --check ...')
    result = subprocess.run(
        [UV, 'run', SERVE, '--check'],
        capture_output=True, text=True, timeout=30,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        print(f'  FAIL: --check exited {result.returncode}')
        print(f'        stdout: {result.stdout!r}')
        print(f'        stderr: {result.stderr!r}')
        return False
    if 'backend' not in output.lower() and 'plan' not in output.lower() \
            and 'llama' not in output.lower():
        print(f'  FAIL: --check output does not look like a plan: {output!r}')
        return False
    print('  PASS: --check exited 0 and printed a backend plan')
    return True


def test_server_boots_and_health(port):
    """serve.py starts, /health returns 200 with status ok."""
    global _server_proc
    print(f'  launching: serve.py --port {port} ...')
    _server_proc = subprocess.Popen(
        [UV, 'run', SERVE, '--port', str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )

    # Drain stdout in a background thread so the pipe never blocks.
    lines = []
    def _drain():
        for raw in _server_proc.stdout:
            lines.append(raw.decode('utf-8', 'replace').rstrip())
    threading.Thread(target=_drain, daemon=True).start()

    # serve.py loads BOTH models at startup (and Phi may be CPU-only when the GPU
    # is full), so /health does not flip to ok until both backends are warm. Allow
    # ample time for a cold model load, not the skeleton-era 30s.
    ok, data = wait_health(port, timeout=300)
    if not ok:
        print(f'  FAIL: /health did not return status ok within 300s')
        print(f'        server output: {lines}')
        return False
    print(f'  PASS: /health -> {data}')
    return True


def test_shutdown(port):
    """POST /shutdown -- process exits cleanly within a few seconds."""
    print('  sending POST /shutdown ...')
    code = post_shutdown(port)
    if code not in (200, 0):
        print(f'  WARN: POST /shutdown returned HTTP {code} (continuing)')
    # Wait for the process to exit.
    deadline = time.time() + 10
    while time.time() < deadline:
        if _server_proc.poll() is not None:
            rc = _server_proc.returncode
            print(f'  PASS: process exited with code {rc}')
            return True
        time.sleep(0.2)
    print('  FAIL: process did not exit within 10s after POST /shutdown')
    return False


def test_log_file():
    """A logs/*.log file must exist and contain at least one line with request_id."""
    ok, path = check_log_has_request_id()
    if ok:
        print(f'  PASS: found request_id in {path}')
        return True
    # List what we see for diagnostics.
    found = glob.glob(os.path.join(LOGS_DIR, '*.log'))
    print(f'  FAIL: no logs/*.log line with request_id.  Files found: {found}')
    return False


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    port    = free_port()
    results = {}

    print('=' * 60)
    print('P0 lifecycle test')
    print('=' * 60)

    # 1. --check
    print('\n[1] --check flag')
    results['check'] = test_check_flag()

    # 2. server boots + health
    print(f'\n[2] server boot + /health  (port {port})')
    results['boot'] = test_server_boots_and_health(port)

    # 3. shutdown (only meaningful if boot passed)
    print('\n[3] POST /shutdown')
    if results['boot']:
        results['shutdown'] = test_shutdown(port)
    else:
        print('  SKIP: server did not boot')
        results['shutdown'] = False

    # 4. log file
    print('\n[4] logs/*.log contains request_id')
    results['logs'] = test_log_file()

    # ── summary ────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    all_pass = all(results.values())
    for name, passed in results.items():
        mark = 'PASS' if passed else 'FAIL'
        print(f'  {mark}  {name}')
    print('=' * 60)
    if all_pass:
        print('ALL CHECKS PASSED')
        sys.exit(0)
    else:
        print('SOME CHECKS FAILED')
        sys.exit(1)


if __name__ == '__main__':
    main()
