#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
slm-rag serve.py -- P0 skeleton: HTTP server, lifecycle, logging.

No models are loaded yet.  This phase establishes:
  - arg parsing (--web / --cli / --port / --check / --help; NO env vars)
  - static index.html at /
  - GET /health   -> JSON status
  - POST /shutdown and GET /shutdown?<UTC-timestamp> (localhost-only)
  - SIGINT / SIGTERM handlers that drain the HTTP server and exit cleanly
  - ./logs/ JSONL writer (hourly-rotated UTC, one lock), every line carrying
    both a request_id (per HTTP exchange) and a chat_session_id (null here)

Backends (Phi-4-mini generation, nomic-embed-text embeddings) are stubbed;
they will be wired in P3.  --check prints the planned backend layout and exits.

Port default: 51548 (the only port-selection mechanism; no env vars).
"""

import sys
import os
import json
import signal
import threading
import re
import time
import uuid
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import unquote_plus
from datetime import datetime, timezone

# ── stdout / stderr always UTF-8, even on Windows ────────────────────────────
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Default listen port.  --port is the ONLY override; there is no env var.
DEFAULT_PORT = 51548
HOST         = '127.0.0.1'   # localhost-only; slm-rag is a personal desktop app


##############################################################################
# --check: planned backend layout (stub for P0; fleshed out in P3)
##############################################################################

def describe_plan(port):
    """Print the intended backend layout and exit.  No downloads, no sockets."""
    print(f'[serve] host:    {sys.platform}  python: {sys.version.split()[0]}')
    print(f'[serve] bind:    http://{HOST}:{port}')
    print('[serve] backends planned (not yet started):')
    print('[serve]   generation  Phi-4-mini Q4_K_M GGUF  port 52851'
          '  (llama-server or llama-cpp-python)')
    print('[serve]   embeddings  nomic-embed-text v1.5    port 52852'
          '  (CPU-only, n-gpu-layers 0)')
    print('[serve]   vector store  SQLite + sqlite-vec  (rag.db, no external server)')
    print('[serve] --check: no downloads, no backends started.')


##############################################################################
# ./logs/ JSONL writer -- hourly-rotated UTC files, written under a lock.
# Every line carries request_id (per HTTP exchange) and chat_session_id.
##############################################################################

LOG_DIR   = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
_log_lock = threading.Lock()


def log_event(stage, ip, method, path, *,
              request_id=None, chat_session_id=None,
              status=None, request_body=None, response_body=None,
              error=None, **extra):
    """Append one JSON line to the current hourly log file.

    Every line carries:
      ts              ISO-8601 UTC timestamp (millisecond precision)
      stage           e.g. 'http_request', 'http_response', 'shutdown'
      ip              caller IP
      method / path   HTTP verb and path
      request_id      uuid4, set per HTTP request at arrival; binds request+response
      chat_session_id uuid4 of the chat session, or null at this skeleton stage
    """
    now      = datetime.now(timezone.utc)
    filename = now.strftime('%Y-%m-%d-%HZ.log')
    entry    = {
        'ts':              now.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
        'stage':           stage,
        'ip':              ip,
        'method':          method,
        'path':            path,
        'request_id':      request_id,      # always present (may be None before assigned)
        'chat_session_id': chat_session_id, # null in P0; populated in P6+
    }
    if status is not None:
        entry['status'] = status
    if request_body is not None:
        entry['request'] = request_body
    if response_body is not None:
        entry['response'] = response_body
    if error is not None:
        entry['error'] = error
    # Merge any caller-supplied extra fields (e.g. model, latency_ms).
    entry.update({k: v for k, v in extra.items() if v is not None})

    line = json.dumps(entry, ensure_ascii=False) + '\n'
    with _log_lock:
        with open(os.path.join(LOG_DIR, filename), 'a', encoding='utf-8') as f:
            f.write(line)


##############################################################################
# Shutdown helpers -- idempotent, drains the HTTP server
##############################################################################

_shutdown_lock    = threading.Lock()
_shutdown_started = False


def begin_shutdown(server):
    """Trigger a clean shutdown exactly once.  Safe to call from any thread."""
    global _shutdown_started
    with _shutdown_lock:
        if _shutdown_started:
            return          # idempotent -- first caller wins
        _shutdown_started = True

    def _worker():
        print('[serve] shutdown: draining HTTP server ...', flush=True)
        server.shutdown()   # blocks until serve_forever() returns
        print('[serve] shutdown: done.', flush=True)

    threading.Thread(target=_worker, daemon=True).start()


def fresh_shutdown_timestamp(query, now=None, window_s=300):
    """Accept a forgiving UTC timestamp anywhere in the raw query string.

    Strips separators and accepts YYYYMMDDHHMM or YYYYMMDDHHMMSS within
    +/- window_s seconds of now.  Returns (True, datetime) or (False, None).
    This is the same guard used by merv to prevent cached/stray GETs from
    killing the server.
    """
    now  = now or datetime.now(timezone.utc)
    text = unquote_plus(query or '')
    digits = ''.join(re.findall(r'\d', text))
    for start in range(len(digits)):
        for width, fmt in ((12, '%Y%m%d%H%M'), (14, '%Y%m%d%H%M%S')):
            stamp = digits[start:start + width]
            if len(stamp) < width:
                continue
            try:
                ts = datetime.strptime(stamp, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if abs((now - ts).total_seconds()) <= window_s:
                return True, ts
    return False, None


##############################################################################
# HTTP handler
##############################################################################

class RagHandler(SimpleHTTPRequestHandler):
    """Handles HTTP requests for slm-rag.

    Serves the static index.html at /, responds to /health and /shutdown,
    and logs every request + response pair with a per-request uuid4.
    """

    def __init__(self, *args, **kwargs):
        # Serve static files out of BASE_DIR so index.html is found at /.
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    # Silence the default stderr logging; we write structured JSONL instead.
    def log_message(self, format, *args):
        pass

    # ── request entry points ──────────────────────────────────────────────

    def do_GET(self):
        path, _, query = self.path.partition('?')
        request_id = str(uuid.uuid4())
        ip = self.client_address[0]

        log_event('http_request', ip, 'GET', path, request_id=request_id)

        if path == '/health':
            self._handle_health(request_id)
        elif path == '/shutdown':
            self._handle_shutdown('GET', query, request_id)
        elif path == '/':
            # Rewrite to the static placeholder; SimpleHTTPRequestHandler serves it.
            self.path = '/index.html'
            # Capture response via super() -- log after the fact.
            super().do_GET()
            log_event('http_response', ip, 'GET', '/', request_id=request_id, status=200)
        else:
            super().do_GET()
            log_event('http_response', ip, 'GET', path, request_id=request_id, status=200)

    def do_POST(self):
        content_len = int(self.headers.get('Content-Length', 0))
        body        = self.rfile.read(content_len)
        path        = self.path
        request_id  = str(uuid.uuid4())
        ip          = self.client_address[0]

        log_event('http_request', ip, 'POST', path,
                  request_id=request_id,
                  request_body=body.decode('utf-8', 'replace') if body else None)

        if path == '/shutdown':
            self._handle_shutdown('POST', '', request_id)
        else:
            self._json_response({'error': 'Not found'}, 404, request_id=request_id)

    # ── endpoint handlers ─────────────────────────────────────────────────

    def _handle_health(self, request_id):
        """GET /health -- always 'ok' at this skeleton stage (no models yet)."""
        payload = {
            'status':    'ok',
            'version':   'p0-skeleton',
            'host':      HOST,
            'backends':  'not_started',   # will be 'ready' in P3
        }
        self._json_response(payload, 200, request_id=request_id)

    def _handle_shutdown(self, method, query, request_id):
        """POST /shutdown (unconditional) or GET /shutdown?<UTC-timestamp>.

        Both are localhost-only.  The timestamp guard on GET prevents a cached
        or stray hyperlink from killing the server.
        """
        ip = self.client_address[0]
        if ip not in ('127.0.0.1', '::1'):
            self._json_response(
                {'error': 'Shutdown is only allowed from localhost'}, 403,
                request_id=request_id)
            log_event('http_response', ip, method, '/shutdown',
                      request_id=request_id, status=403)
            return

        if method == 'GET':
            ok, ts = fresh_shutdown_timestamp(query)
            if not ok:
                self._json_response(
                    {'error': 'GET /shutdown requires a UTC timestamp within 5 minutes'},
                    400, request_id=request_id)
                log_event('http_response', ip, method, '/shutdown',
                          request_id=request_id, status=400)
                return
            detail = f'timestamp {ts.isoformat()}'
        else:
            detail = 'POST'

        log_event('shutdown', ip, method, '/shutdown',
                  request_id=request_id, detail=detail)
        self._json_response({'status': 'shutting_down'}, 200, request_id=request_id)
        begin_shutdown(self.server)

    # ── response helpers ──────────────────────────────────────────────────

    def _json_response(self, obj, status=200, *, request_id=None):
        data = json.dumps(obj).encode()
        ip   = self.client_address[0]
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        log_event('http_response', ip, self.command, self.path.partition('?')[0],
                  request_id=request_id, status=status)


##############################################################################
# Threaded HTTP server -- one daemon thread per connection
##############################################################################

class ThreadedHTTPServer(HTTPServer):
    """Each connection runs in its own daemon thread so one slow client
    does not stall others.  Same pattern as merv."""

    def process_request(self, request, client_address):
        t = threading.Thread(target=self._handle, args=(request, client_address))
        t.daemon = True
        t.start()

    def _handle(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


##############################################################################
# CLI stub -- will be fleshed out in P7
##############################################################################

def cli_stub(base):
    """Minimal --cli REPL placeholder.

    In P7 this will be replaced with a full thin-HTTP-client REPL (ask a
    question, /clear, /help, /quit, citations inline).  For now it just
    prints a note and returns, leaving the web server running.
    """
    print('[serve] --cli: terminal chat not yet implemented (P7 stub).')
    print('[serve] Web server is still running.  Press Ctrl-C to stop.')


##############################################################################
# Argument parsing -- NO argparse; manual loop mirrors merv's style
##############################################################################

COMMAND_LINE_HELP = """\
[serve] slm-rag serve.py -- command-line args:
  --web        Run the web server only.  This is the default.
  --cli        Run the web server and attach a terminal chat (stub in P0).
  --port <n>   Listen port (default 51548).  This flag is the only override.
  --check      Print the planned backend layout and exit.  No downloads, no
               models start, no socket opened.
  --help       Print this help and exit.

No environment variables are used; every setting has a baked-in default.
"""


def parse_args(argv):
    mode         = 'web'
    mode_flags   = []
    port         = DEFAULT_PORT
    check        = False
    help_wanted  = False

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ('--web', '--cli'):
            mode_flags.append(arg)
            mode = arg[2:]
        elif arg == '--port':
            i += 1
            if i >= len(argv):
                print('[serve] --port requires a number, e.g. --port 8080', flush=True)
                sys.exit(2)
            try:
                port = int(argv[i])
            except ValueError:
                print('[serve] --port requires a number, e.g. --port 8080', flush=True)
                sys.exit(2)
        elif arg == '--check':
            check = True
        elif arg in ('--help', '-h'):
            help_wanted = True
        else:
            print(f'[serve] unknown argument: {arg!r}', flush=True)
            sys.exit(2)
        i += 1

    if len(set(mode_flags)) > 1:
        print('[serve] choose one mode: --web or --cli', flush=True)
        sys.exit(2)

    return {'mode': mode, 'port': port, 'check': check, 'help': help_wanted}


##############################################################################
# Entry point
##############################################################################

def main():
    args = parse_args(sys.argv[1:])

    if args['help']:
        print(COMMAND_LINE_HELP.rstrip(), flush=True)
        return

    port = args['port']

    if args['check']:
        describe_plan(port)
        return                  # no socket, no downloads, exit 0

    # ── start the HTTP server ─────────────────────────────────────────────
    try:
        server = ThreadedHTTPServer((HOST, port), RagHandler)
    except OSError as e:
        print(f'[serve] ERROR: cannot bind to {HOST}:{port} -- {e}', flush=True)
        print(f'[serve] Try a different port: --port 51549 (or any free port)', flush=True)
        sys.exit(1)

    # SIGINT / SIGTERM: drain the HTTP server and exit cleanly.
    # os._exit(0) ensures we do not hang on non-daemon threads if any appear later.
    def _signal_handler(*_):
        print('[serve] signal received -- shutting down ...', flush=True)
        log_event('shutdown', '127.0.0.1', 'SIGNAL', '/shutdown')
        begin_shutdown(server)

    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Serve in a daemon thread so the main thread can block on join() / the CLI.
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f'[serve] listening on http://{HOST}:{port}', flush=True)
    log_event('startup', '127.0.0.1', 'START', '/',
              port=port, version='p0-skeleton')

    if args['mode'] == 'cli':
        cli_stub(f'http://{HOST}:{port}')
        # After cli_stub returns (or on Ctrl-C inside it), begin a clean shutdown.
        begin_shutdown(server)

    # Block until server_thread finishes (i.e. server.shutdown() was called).
    server_thread.join()
    print('[serve] server stopped.', flush=True)


if __name__ == '__main__':
    main()
