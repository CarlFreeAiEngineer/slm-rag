#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface_hub",
#     "llama-cpp-python==0.3.30; sys_platform == 'linux'",
#     "pypdf>=4.0",
#     "sqlite-vec",
# ]
# ///
"""
slm-rag serve.py -- P5: retrieval (question -> chunks).

Extends P4 to add the retrieve endpoint:

  - POST /retrieve {"question": "...", "scope": "<optional path>", "k": <optional int>}:
    embed the question (embed gate, 'search_query: ' prefix for nomic v1.5) ->
    sqlite-vec k-NN -> return top-k chunks with full citation metadata.
    Optional 'scope' restricts results to a specific file or folder prefix.

P4 endpoints retained: POST /ingest, GET /tree, GET /doc.
P3 endpoints retained: POST /embed, POST /generate, GET /health.

EMBEDDING NOTE: nomic-embed-text v1.5 uses task prefixes.  Document chunks are
prefixed with 'search_document: ' before embedding.  Retrieval MUST use
'search_query: ' for the question vector so they live in the same embedding space.

Port default: 51548 (the only port-selection mechanism; no env vars).
Model ports: 52851 (Phi/gen), 52852 (embedder).
"""

import sys
import os
import json
import signal
import threading
import re
import time
import uuid
import shutil
import subprocess
import http.client
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.request import urlopen, Request
from urllib.error import URLError
from urllib.parse import unquote_plus, parse_qs
from datetime import datetime, timezone

# ── stdout / stderr always UTF-8, even on Windows ────────────────────────────
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
RAGDOCS_DIR = os.path.join(BASE_DIR, 'ragdocs')  # ingested source files live here

# Default listen port.  --port is the ONLY override; there is no env var.
DEFAULT_PORT = 51548
HOST         = '127.0.0.1'   # localhost-only; slm-rag is a personal desktop app

# Model ports (fixed; no flags needed -- each model is always on the same port)
PHI_PORT    = 52851
EMBED_PORT  = 52852

# CPU threads for llama-server (used for both backends)
THREADS = 4

# ── Local module imports (db.py, ingest_lib.py live next to serve.py) ─────────
# These must come after BASE_DIR is defined so the path is available for sys.path.
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import db as _db_mod           # db.py: init_db, insert_chunk, knn_chunks …
import ingest_lib as _ingest   # ingest_lib.py: extract_text, chunk_text, ingest_file

# ── Global database connection (set in main()) ────────────────────────────────
# init_db() is called once at startup with the path from --db (or the default).
# The connection is shared across threads; db.connect() uses check_same_thread=False.
_db_conn = None   # sqlite3.Connection, set in main()
_db_lock = threading.Lock()   # serialises multi-step transactions from different threads

# Default db path (can be overridden with --db flag; no env var)
DEFAULT_DB = os.path.join(BASE_DIR, 'rag.db')


##############################################################################
# Hardware detection -- GPU VRAM for deciding Phi offload
##############################################################################

def nvidia_vram_gb():
    """(total_gb, free_gb) for the primary NVIDIA GPU, or (None, None) if absent."""
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5)
        if out.returncode != 0 or not out.stdout.strip():
            return (None, None)
        total, free = (float(x) for x in out.stdout.strip().splitlines()[0].split(','))
        return (total / 1024.0, free / 1024.0)   # MiB -> GiB
    except Exception:
        return (None, None)


GPU_TOTAL_GB, GPU_FREE_GB = nvidia_vram_gb()

# Headroom over weight size for KV cache + compute buffers at 4096 context.
GPU_HEADROOM_GB = 1.3


def gpu_layers_for_phi():
    """How many layers to offload the Phi model to the GPU.
    '99' = all layers (GPU), '0' = CPU-only.

    Rules (no env var override -- slm-rag uses flags only):
      macOS   -> '99' (Apple Metal, unified memory)
      NVIDIA  -> '99' if Phi (~2.4 GB + headroom) fits in free VRAM, else '0'
      no GPU  -> '0'
    """
    if sys.platform == 'darwin':
        return '99'
    if GPU_FREE_GB is None:
        return '0'           # no NVIDIA GPU detected
    phi_need = 2.4 + GPU_HEADROOM_GB
    return '99' if phi_need <= GPU_FREE_GB else '0'


##############################################################################
# llama-server discovery and bundled-binary bootstrap
##############################################################################

def find_llama_server():
    """Locate a llama-server binary.  Prefer the bundled copy when present."""
    bundled = os.path.join(BASE_DIR, 'bin', 'llama.cpp',
                           'llama-server.exe' if os.name == 'nt' else 'llama-server')
    if os.path.isfile(bundled):
        return bundled
    found = shutil.which('llama-server')
    if found:
        return found
    for cand in ('/opt/homebrew/bin/llama-server',
                 '/usr/local/bin/llama-server',
                 '/usr/bin/llama-server'):
        if os.path.isfile(cand):
            return cand
    return None


# llama.cpp release tag and CUDA version for Windows bundled build
LLAMA_CPP_TAG  = 'b9761'
LLAMA_CPP_CUDA = '12.4'


def ensure_llama_server():
    """On Windows, make sure the bundled GPU-capable llama.cpp build is present.
    Downloads the CUDA server (llama-server.exe + ggml-cuda.dll) and CUDA runtime
    DLLs from the llama.cpp GitHub release if anything is missing.  No-op on other
    platforms and when everything is already present."""
    if os.name != 'nt':
        return
    dest = os.path.join(BASE_DIR, 'bin', 'llama.cpp')
    exe  = os.path.join(dest, 'llama-server.exe')
    have_exe    = os.path.isfile(exe)
    have_cudart = any(f.lower().startswith('cudart64') and f.lower().endswith('.dll')
                      for f in (os.listdir(dest) if os.path.isdir(dest) else []))
    server_zip = f'llama-{LLAMA_CPP_TAG}-bin-win-cuda-{LLAMA_CPP_CUDA}-x64.zip'
    cudart_zip = f'cudart-llama-bin-win-cuda-{LLAMA_CPP_CUDA}-x64.zip'
    needed = []
    if not have_exe:
        needed.append(server_zip)
    if not have_cudart:
        needed.append(cudart_zip)
    if not needed:
        return

    import zipfile
    os.makedirs(dest, exist_ok=True)
    base = f'https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_TAG}'
    for asset in needed:
        tmp = os.path.join(dest, asset)
        print(f'[serve] downloading {asset} (GPU-capable llama.cpp build) ...', flush=True)
        try:
            req = Request(f'{base}/{asset}', headers={'User-Agent': 'slm-rag-serve'})
            with urlopen(req, timeout=120) as r, open(tmp, 'wb') as f:
                shutil.copyfileobj(r, f)
            with zipfile.ZipFile(tmp) as z:
                z.extractall(dest)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
    if not os.path.isfile(exe):
        print('[serve] ERROR: llama-server.exe missing after download', flush=True)


##############################################################################
# HuggingFace weight download
# Weights are downloaded smallest first so the embedder is ready before Phi.
##############################################################################

# Model weight descriptors.  Local paths are under model/ (git-ignored).
HF_WEIGHTS = {
    'embedder': {
        'repo':      'nomic-ai/nomic-embed-text-v1.5-GGUF',
        'filename':  'nomic-embed-text-v1.5.Q4_K_M.gguf',
        'local':     os.path.join(BASE_DIR, 'model', 'nomic-embed',
                                  'nomic-embed-text-v1.5.Q4_K_M.gguf'),
        'approx_gb': 0.08,   # ~80 MB
    },
    'phi': {
        # The UNCUSTOMIZED base Phi-4-mini-instruct (MIT). slm-rag starts uncustomized
        # and fine-tunes on its own RAG corrections, so it must NOT point at a persona
        # fine-tune (one that injects its own voice instead of answering from context).
        'repo':      'bartowski/microsoft_Phi-4-mini-instruct-GGUF',
        'filename':  'microsoft_Phi-4-mini-instruct-Q4_K_M.gguf',
        'local':     os.path.join(BASE_DIR, 'model', 'phi4mini',
                                  'microsoft_Phi-4-mini-instruct-Q4_K_M.gguf'),
        'approx_gb': 2.49,
    },
}

# Download-pause machinery: while the gen backend is generating we pause the
# background downloader so we don't evict the model from the OS page cache.
_infer_active     = 0
_infer_lock       = threading.Lock()
_downloads_paused = threading.Event()   # set => pause streamed downloads


def infer_enter():
    global _infer_active
    with _infer_lock:
        _infer_active += 1
        _downloads_paused.set()


def infer_exit():
    global _infer_active
    with _infer_lock:
        _infer_active = max(0, _infer_active - 1)
        if _infer_active == 0:
            _downloads_paused.clear()


def _streamed_download(cfg):
    """Stream the GGUF to <local>.part, pausing while inference runs, then
    atomically rename into place.  Raises on any error."""
    from huggingface_hub import hf_hub_url
    url  = hf_hub_url(repo_id=cfg['repo'], filename=cfg['filename'])
    dst  = cfg['local']
    part = dst + '.part'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    req = Request(url, headers={'User-Agent': 'slm-rag-serve'})
    try:
        with urlopen(req, timeout=60) as resp, open(part, 'wb') as f:
            while True:
                while _downloads_paused.is_set():
                    time.sleep(0.1)
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(part, dst)
    except BaseException:
        try:
            os.remove(part)
        except OSError:
            pass
        raise


def download_one(key):
    """Download one model's weights (blocking).  Returns True on success or if
    the weights are already present; False on failure."""
    cfg = HF_WEIGHTS.get(key)
    if not cfg:
        return False
    if os.path.isfile(cfg['local']):
        return True
    print(f'[serve] {key}: downloading {cfg["filename"]} from {cfg["repo"]} '
          f'(~{cfg["approx_gb"]:.2f} GB) ...', flush=True)
    try:
        _streamed_download(cfg)
        print(f'[serve] {key}: download complete', flush=True)
        return True
    except Exception as e:
        print(f'[serve] {key}: streamed download failed ({e}); '
              f'falling back to hf_hub_download', flush=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print('[serve] huggingface_hub not installed -- cannot download', flush=True)
        return False
    try:
        os.makedirs(os.path.dirname(cfg['local']), exist_ok=True)
        hf_hub_download(repo_id=cfg['repo'], filename=cfg['filename'],
                        local_dir=os.path.dirname(cfg['local']))
        print(f'[serve] {key}: download complete', flush=True)
        return True
    except Exception as e:
        print(f'[serve] {key}: download failed: {e}', flush=True)
        return False


def ensure_weights():
    """Download both model GGUFs if not already cached, embedder first."""
    for key in ('embedder', 'phi'):
        if not os.path.isfile(HF_WEIGHTS[key]['local']):
            ok = download_one(key)
            if not ok:
                print(f'[serve] FATAL: could not obtain weights for {key}', flush=True)
                sys.exit(1)


##############################################################################
# Two gates -- one per model.
#
# Each gate is a plain threading.Lock.  A request takes them SEQUENTIALLY
# (never nested):
#
#   embed gate -> embed text -> release
#   (sqlite-vec k-NN, no gate)
#   gen gate   -> generate   -> release
#
# While a long ingest holds the embed gate between batches it YIELDS it between
# each batch (P4 -- the seam is here: ingest will call embed_gate.acquire() /
# embed_gate.release() per batch so an interactive query can slip in between).
##############################################################################

embed_gate = threading.Lock()   # serializes the vector/embedder model
gen_gate   = threading.Lock()   # serializes the language/Phi model

# ── Ingestion constants ───────────────────────────────────────────────────────

# nomic-embed-text v1.5 model identifier stored in chunk_vec_meta.
# If the model changes, stored embedders can be detected and rebuilt.
EMBEDDER_ID = 'nomic-embed-text-v1.5'

# Number of chunks to embed per gate-acquisition.  After each batch the embed
# gate is released so an interactive query-embed can slip in between batches.
INGEST_BATCH_SIZE = 4


##############################################################################
# Ingestion pipeline -- background worker
##############################################################################

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ingest_background(doc_id: int, file_path: str, rel_path: str, request_id: str):
    """
    Background worker: extract -> chunk -> embed (in batches, yielding the embed
    gate between batches) -> store chunks + vectors -> set doc status ready/error.

    Runs in a daemon thread; logs errors to stdout and sets doc status='error'.

    EMBEDDING PREFIX: nomic-embed-text v1.5 requires task-type prefixes for
    asymmetric retrieval quality.  Each document chunk is prefixed with
    'search_document: ' here.  P5 retrieval MUST use 'search_query: ' for the
    question so query and document vectors live in the same embedding space.
    """
    try:
        # ── Step 1: extract text and chunk ───────────────────────────────────
        chunks = _ingest.ingest_file(file_path)

        # Update n_chunks in the documents row now that we know the count
        with _db_lock:
            _db_conn.execute(
                "UPDATE documents SET n_chunks=? WHERE id=?",
                (len(chunks), doc_id),
            )
            _db_conn.commit()

        # ── Step 2: embed in batches, yielding the embed gate between batches ─
        # Breaking into INGEST_BATCH_SIZE chunks per gate-acquisition lets an
        # interactive /embed call slip in between batches instead of waiting for
        # the entire document to be embedded before it can proceed.
        for batch_start in range(0, len(chunks), INGEST_BATCH_SIZE):
            batch = chunks[batch_start: batch_start + INGEST_BATCH_SIZE]

            # ── YIELD POINT: acquire the embed gate for this batch only ───────
            # After the batch is embedded and stored we release the gate before
            # taking it again for the next batch.  Any /embed request waiting on
            # the lock will be served between batches.
            with embed_gate:   # <-- acquire gate for one batch
                embedded_batch = []
                for chunk in batch:
                    # Prefix with 'search_document: ' for nomic-embed-text v1.5
                    # asymmetric retrieval.  P5 uses 'search_query: ' on the question.
                    prefixed_text = 'search_document: ' + chunk['text']
                    vector = _embed_backend.embed(prefixed_text)
                    embedded_batch.append((chunk, vector))
            # embed_gate is released here -- next /embed or next batch can proceed

            # ── Step 3: store each chunk + vector ────────────────────────────
            with _db_lock:
                for chunk, vector in embedded_batch:
                    _db_mod.insert_chunk(
                        _db_conn,
                        doc_id=doc_id,
                        chunk_index=chunk['chunk_index'],
                        text=chunk['text'],
                        char_start=chunk['char_start'],
                        char_end=chunk['char_end'],
                        embedding=vector,
                        embedder_id=EMBEDDER_ID,
                    )
                _db_conn.commit()

        # ── Step 4: mark document ready ──────────────────────────────────────
        with _db_lock:
            _db_conn.execute(
                "UPDATE documents SET status='ready' WHERE id=?", (doc_id,)
            )
            _db_conn.commit()
        print(f'[serve] ingest complete: doc_id={doc_id} rel_path={rel_path!r} '
              f'n_chunks={len(chunks)}', flush=True)

    except Exception as exc:
        print(f'[serve] ingest ERROR: doc_id={doc_id} rel_path={rel_path!r}: {exc}',
              flush=True)
        try:
            with _db_lock:
                _db_conn.execute(
                    "UPDATE documents SET status='error', error_msg=? WHERE id=?",
                    (str(exc)[:1000], doc_id),
                )
                _db_conn.commit()
        except Exception:
            pass


##############################################################################
# Backend: ProxyBackend -- wraps a llama-server subprocess
##############################################################################

class ProxyBackend:
    """Runs llama-server as a subprocess and proxies HTTP to it.

    slm-rag keeps TWO backends resident at once (one embed, one gen) -- each is a
    separate ProxyBackend that boots independently and stays up for the lifetime
    of the server.
    """

    def __init__(self, name, cmd, port, ready_kind='llama'):
        """
        name       : short label, e.g. 'phi' or 'embedder'
        cmd        : list[str] -- full command for Popen
        port       : int -- the llama-server listen port
        ready_kind : 'llama' -> poll /health; 'openai' -> poll /v1/models
        """
        self.name       = name
        self.cmd        = cmd
        self.port       = port
        self.ready_kind = ready_kind
        self.proc       = None

    def _gpu_layers(self):
        try:
            return self.cmd[self.cmd.index('--n-gpu-layers') + 1]
        except (ValueError, IndexError):
            return '0'

    def _force_cpu(self):
        try:
            idx = self.cmd.index('--n-gpu-layers')
            self.cmd[idx + 1] = '0'
        except (ValueError, IndexError):
            pass

    def _alive(self):
        return self.proc is not None and self.proc.poll() is None

    def _wait_ready(self, timeout=300):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise TimeoutError(f'process exited with code {self.proc.returncode}')
            try:
                if self.ready_kind == 'llama':
                    resp = urlopen(f'http://127.0.0.1:{self.port}/health', timeout=2)
                    if resp.status == 200 and json.loads(resp.read()).get('status') == 'ok':
                        return
                else:
                    resp = urlopen(f'http://127.0.0.1:{self.port}/v1/models', timeout=2)
                    if resp.status == 200:
                        return
            except (URLError, OSError, json.JSONDecodeError):
                pass
            time.sleep(2)
        raise TimeoutError(f'{self.name} not ready in {timeout}s')

    def _boot_once(self):
        ngl   = self._gpu_layers()
        where = f'GPU (ngl={ngl})' if ngl != '0' else 'CPU (ngl=0)'
        print(f'[serve] loading {self.name} on port {self.port} [{where}] ...', flush=True)
        self.proc = subprocess.Popen(
            self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            self._wait_ready()
            print(f'[serve] {self.name} ready on port {self.port} [{where}]', flush=True)
            return True
        except Exception as e:
            out = ''
            if self.proc and self.proc.stdout:
                try:
                    out = self.proc.stdout.read(4096).decode('utf-8', 'replace')
                except Exception:
                    pass
            print(f'[serve] {self.name} failed to start ({e}):\n{out}', flush=True)
            self.stop()
            return False

    def boot(self):
        """Start the backend.  If GPU launch fails, retry CPU-only (gen only)."""
        if self._boot_once():
            return True
        # Only retry GPU -> CPU for the gen model; embedder is always CPU
        if self._gpu_layers() != '0':
            print(f'[serve] {self.name}: GPU launch failed -- retrying CPU-only', flush=True)
            self._force_cpu()
            return self._boot_once()
        return False

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None

    # ── Embedding (embedder backend only) ────────────────────────────────────

    def embed(self, text):
        """POST /v1/embeddings to llama-server; return list of floats.
        llama-server --embedding exposes the OpenAI-compatible embeddings endpoint."""
        payload = json.dumps({'input': text, 'model': 'nomic-embed'}).encode()
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=60)
        try:
            conn.request('POST', '/v1/embeddings', payload,
                         headers={'Content-Type': 'application/json'})
            resp = conn.getresponse()
            data = resp.read()
            if resp.status >= 400:
                raise RuntimeError(f'embed HTTP {resp.status}: '
                                   f'{data.decode("utf-8","replace")[:200]}')
            obj = json.loads(data)
            return obj['data'][0]['embedding']
        finally:
            conn.close()

    # ── Generation (Phi backend only) ────────────────────────────────────────

    def generate(self, prompt, max_tokens=256, temperature=0.7, top_p=0.9):
        """POST /v1/chat/completions; return the reply text."""
        payload = json.dumps({
            'messages':    [{'role': 'user', 'content': prompt}],
            'max_tokens':  max_tokens,
            'temperature': temperature,
            'top_p':       top_p,
            'stream':      False,
        }).encode()
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=300)
        try:
            conn.request('POST', '/v1/chat/completions', payload,
                         headers={'Content-Type': 'application/json'})
            resp = conn.getresponse()
            data = resp.read()
            if resp.status >= 400:
                raise RuntimeError(f'gen HTTP {resp.status}: '
                                   f'{data.decode("utf-8","replace")[:200]}')
            obj  = json.loads(data)
            msg  = obj['choices'][0]['message']
            text = (msg.get('content') or msg.get('reasoning') or
                    msg.get('reasoning_content') or '')
            return text
        finally:
            conn.close()


##############################################################################
# In-process backend (Linux only -- llama-cpp-python)
##############################################################################

class InProcEmbedBackend:
    """llama-cpp-python in-process embedder (Linux only)."""

    _llm = None

    def __init__(self, path):
        self.path = path

    def boot(self):
        from llama_cpp import Llama
        print(f'[serve] loading embedder in-process from {self.path} ...', flush=True)
        InProcEmbedBackend._llm = Llama(
            model_path=self.path,
            n_ctx=512,
            n_threads=THREADS,
            n_threads_batch=THREADS,
            embedding=True,
            n_gpu_layers=0,   # embedder always CPU
            verbose=False,
        )
        print('[serve] embedder ready (in-process, CPU)', flush=True)

    def embed(self, text):
        return InProcEmbedBackend._llm.embed(text)

    def stop(self):
        InProcEmbedBackend._llm = None


class InProcGenBackend:
    """llama-cpp-python in-process Phi backend (Linux only)."""

    _llm = None

    def __init__(self, path):
        self.path = path

    def boot(self):
        from llama_cpp import Llama
        ngl = int(gpu_layers_for_phi())
        where = f'GPU (ngl={ngl})' if ngl > 0 else 'CPU'
        print(f'[serve] loading phi in-process from {self.path} [{where}] ...', flush=True)
        InProcGenBackend._llm = Llama(
            model_path=self.path,
            n_ctx=4096,
            n_threads=THREADS,
            n_threads_batch=THREADS,
            n_gpu_layers=ngl,
            verbose=False,
        )
        print(f'[serve] phi ready (in-process, {where})', flush=True)

    def generate(self, prompt, max_tokens=256, temperature=0.7, top_p=0.9):
        result = InProcGenBackend._llm.create_chat_completion(
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
        )
        return result['choices'][0]['message'].get('content', '')

    def stop(self):
        InProcGenBackend._llm = None


##############################################################################
# Global backend instances -- built in main(), used by HTTP handlers
##############################################################################

_embed_backend = None   # set in main()
_gen_backend   = None   # set in main()


def _use_inproc():
    """True when we should run in-process (Linux without a bundled llama-server)."""
    return sys.platform == 'linux' and find_llama_server() is None


def build_backends():
    """Create (but do not boot) the embed and gen backend objects."""
    global _embed_backend, _gen_backend

    embed_gguf = HF_WEIGHTS['embedder']['local']
    phi_gguf   = HF_WEIGHTS['phi']['local']
    llama_srv  = find_llama_server()

    if _use_inproc():
        # Linux without llama-server binary: run both in-process
        _embed_backend = InProcEmbedBackend(embed_gguf)
        _gen_backend   = InProcGenBackend(phi_gguf)
    else:
        # Windows / macOS (or Linux with llama-server): subprocess backends
        if not llama_srv:
            print('[serve] FATAL: llama-server not found and not Linux', flush=True)
            sys.exit(1)

        # Embedder: CPU-only, --embedding flag
        embed_cmd = [
            llama_srv,
            '--model',         embed_gguf,
            '--port',          str(EMBED_PORT),
            '--host',          '127.0.0.1',
            '--ctx-size',      '512',
            '--n-gpu-layers',  '0',          # CPU-only: leave all VRAM for Phi
            '--threads',       str(THREADS),
            '--embedding',                   # enable embeddings endpoint
            '--no-webui',
        ]
        _embed_backend = ProxyBackend('embedder', embed_cmd, EMBED_PORT, 'llama')

        # Gen (Phi): GPU if it fits, else CPU
        phi_ngl = gpu_layers_for_phi()
        phi_cmd = [
            llama_srv,
            '--model',         phi_gguf,
            '--port',          str(PHI_PORT),
            '--host',          '127.0.0.1',
            '--ctx-size',      '4096',
            '--n-gpu-layers',  phi_ngl,
            '--threads',       str(THREADS),
            '--no-webui',
        ]
        _gen_backend = ProxyBackend('phi', phi_cmd, PHI_PORT, 'llama')


def boot_backends():
    """Boot both backends concurrently so the server is ready faster."""
    errors = []

    def _boot(b):
        try:
            ok = b.boot()
            if not ok:
                errors.append(f'{b.name if hasattr(b,"name") else type(b).__name__} failed to boot')
        except Exception as e:
            errors.append(str(e))

    t_embed = threading.Thread(target=_boot, args=(_embed_backend,))
    t_gen   = threading.Thread(target=_boot, args=(_gen_backend,))
    t_embed.start()
    t_gen.start()
    t_embed.join()
    t_gen.join()

    if errors:
        for e in errors:
            print(f'[serve] ERROR: {e}', flush=True)
        sys.exit(1)

    print('[serve] both backends ready', flush=True)


def stop_backends():
    """Terminate both backends cleanly."""
    for b in (_embed_backend, _gen_backend):
        if b is not None:
            try:
                b.stop()
            except Exception:
                pass


##############################################################################
# --check: planned backend layout
##############################################################################

def describe_plan(port):
    """Print the intended backend layout and exit.  No downloads, no sockets."""
    llama_srv = find_llama_server()
    phi_ngl   = gpu_layers_for_phi()

    print(f'[serve] host:    {sys.platform}  python: {sys.version.split()[0]}')
    print(f'[serve] bind:    http://{HOST}:{port}')
    if GPU_TOTAL_GB is not None:
        print(f'[serve] GPU:     NVIDIA {GPU_TOTAL_GB:.1f} GB total, {GPU_FREE_GB:.1f} GB free')
    elif sys.platform == 'darwin':
        print('[serve] GPU:     Apple Metal (unified memory)')
    else:
        print('[serve] GPU:     none detected (nvidia-smi unavailable)')
    print(f'[serve] llama-server: {llama_srv or "(none -- will use in-process on Linux)"}')
    print('[serve] backends planned (not yet started):')
    if _use_inproc():
        print('[serve]   embedder  nomic-embed-text-v1.5 Q4_K_M  in-process  CPU (ngl=0)')
        phi_where = f'GPU (ngl={phi_ngl})' if phi_ngl != '0' else 'CPU (ngl=0)'
        print(f'[serve]   phi       Phi-4-mini Q4_K_M              in-process  {phi_where}')
    else:
        print(f'[serve]   embedder  nomic-embed-text-v1.5 Q4_K_M  port {EMBED_PORT}  CPU (ngl=0)')
        phi_where = f'GPU (ngl={phi_ngl})' if phi_ngl != '0' else 'CPU (ngl=0)'
        print(f'[serve]   phi       Phi-4-mini Q4_K_M              port {PHI_PORT}  {phi_where}')
    print('[serve]   vector store  SQLite + sqlite-vec  (rag.db, no external server)')
    print('[serve] HF weight sources:')
    for key, cfg in HF_WEIGHTS.items():
        cached = 'cached' if os.path.isfile(cfg['local']) else 'not cached'
        print(f'[serve]   {key:8s}  {cfg["repo"]}  {cfg["filename"]}  ({cached})')
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
    """Append one JSON line to the current hourly log file."""
    now      = datetime.now(timezone.utc)
    filename = now.strftime('%Y-%m-%d-%HZ.log')
    entry    = {
        'ts':              now.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
        'stage':           stage,
        'ip':              ip,
        'method':          method,
        'path':            path,
        'request_id':      request_id,
        'chat_session_id': chat_session_id,
    }
    if status is not None:
        entry['status'] = status
    if request_body is not None:
        entry['request'] = request_body
    if response_body is not None:
        entry['response'] = response_body
    if error is not None:
        entry['error'] = error
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
            return
        _shutdown_started = True

    def _worker():
        print('[serve] shutdown: stopping backends ...', flush=True)
        stop_backends()
        print('[serve] shutdown: draining HTTP server ...', flush=True)
        server.shutdown()
        print('[serve] shutdown: done.', flush=True)

    threading.Thread(target=_worker, daemon=True).start()


def fresh_shutdown_timestamp(query, now=None, window_s=300):
    """Accept a forgiving UTC timestamp anywhere in the raw query string."""
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
    """Handles HTTP requests for slm-rag."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def log_message(self, format, *args):
        pass   # silence default stderr logging; we write structured JSONL instead

    def do_GET(self):
        path, _, query = self.path.partition('?')
        request_id = str(uuid.uuid4())
        ip = self.client_address[0]

        log_event('http_request', ip, 'GET', path, request_id=request_id)

        if path == '/health':
            self._handle_health(request_id)
        elif path == '/shutdown':
            self._handle_shutdown('GET', query, request_id)
        elif path == '/tree':
            self._handle_tree(request_id)
        elif path == '/doc':
            self._handle_doc(query, request_id)
        elif path == '/':
            self.path = '/index.html'
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
                  request_body=body.decode('utf-8', 'replace')[:500] if body else None)

        if path == '/shutdown':
            self._handle_shutdown('POST', '', request_id)
        elif path == '/embed':
            self._handle_embed(body, request_id)
        elif path == '/generate':
            self._handle_generate(body, request_id)
        elif path == '/ingest':
            self._handle_ingest(body, request_id)
        elif path == '/retrieve':
            self._handle_retrieve(body, request_id)
        else:
            self._json_response({'error': 'Not found'}, 404, request_id=request_id)

    # ── endpoint handlers ─────────────────────────────────────────────────

    def _handle_health(self, request_id):
        """GET /health -- report backend readiness."""
        embed_ready = (_embed_backend is not None and
                       getattr(_embed_backend, 'proc', True) is not None and
                       (not hasattr(_embed_backend, 'proc') or
                        (_embed_backend.proc is not None and
                         _embed_backend.proc.poll() is None)))
        gen_ready   = (_gen_backend is not None and
                       getattr(_gen_backend, 'proc', True) is not None and
                       (not hasattr(_gen_backend, 'proc') or
                        (_gen_backend.proc is not None and
                         _gen_backend.proc.poll() is None)))
        status = 'ok' if (embed_ready and gen_ready) else 'starting'
        payload = {
            'status':   status,
            'version':  'p3-backends',
            'host':     HOST,
            'backends': {
                'embedder': {
                    'port':      EMBED_PORT,
                    'placement': 'CPU (ngl=0)',
                    'ready':     embed_ready,
                },
                'phi': {
                    'port':      PHI_PORT,
                    'ngl':       gpu_layers_for_phi(),
                    'placement': f'GPU (ngl={gpu_layers_for_phi()})' if gpu_layers_for_phi() != '0' else 'CPU (ngl=0)',
                    'ready':     gen_ready,
                },
            },
        }
        self._json_response(payload, 200, request_id=request_id)

    def _handle_embed(self, body, request_id):
        """POST /embed {"text":"..."} -> {"embedding":[...768 floats...]}

        Acquires embed_gate (serializes access to the embedder).
        Released before returning so ingestion can yield it between batches (P4).
        """
        ip = self.client_address[0]
        try:
            req  = json.loads(body)
            text = req.get('text', '')
        except (json.JSONDecodeError, AttributeError):
            self._json_response({'error': 'Invalid JSON or missing "text" field'},
                                400, request_id=request_id)
            return
        if not text:
            self._json_response({'error': '"text" must be a non-empty string'},
                                400, request_id=request_id)
            return

        log_event('embed_request', ip, 'POST', '/embed', request_id=request_id,
                  request_body=text[:200])
        t0 = time.time()
        with embed_gate:
            try:
                vector = _embed_backend.embed(text)
            except Exception as e:
                log_event('embed_response', ip, 'POST', '/embed',
                          request_id=request_id, status=500, error=str(e))
                self._json_response({'error': str(e)}, 500, request_id=request_id)
                return
        latency_ms = int((time.time() - t0) * 1000)
        log_event('embed_response', ip, 'POST', '/embed', request_id=request_id,
                  status=200, dims=len(vector), latency_ms=latency_ms)
        self._json_response({'embedding': vector, 'dims': len(vector)},
                            200, request_id=request_id)

    def _handle_generate(self, body, request_id):
        """POST /generate {"prompt":"..."} -> {"text":"..."}

        Acquires gen_gate (serializes access to the Phi model).
        This is the minimal generate path for P3 debug/test; the full queued
        chat flow (enqueue/worker/stream) is built in P6.
        """
        ip = self.client_address[0]
        try:
            req    = json.loads(body)
            prompt = req.get('prompt', '')
        except (json.JSONDecodeError, AttributeError):
            self._json_response({'error': 'Invalid JSON or missing "prompt" field'},
                                400, request_id=request_id)
            return
        if not prompt:
            self._json_response({'error': '"prompt" must be a non-empty string'},
                                400, request_id=request_id)
            return

        log_event('gen_request', ip, 'POST', '/generate', request_id=request_id,
                  request_body=prompt[:200])
        t0 = time.time()
        infer_enter()
        with gen_gate:
            try:
                text = _gen_backend.generate(
                    prompt,
                    max_tokens=req.get('max_tokens', 256),
                    temperature=req.get('temperature', 0.7),
                    top_p=req.get('top_p', 0.9),
                )
            except Exception as e:
                infer_exit()
                log_event('gen_response', ip, 'POST', '/generate',
                          request_id=request_id, status=500, error=str(e))
                self._json_response({'error': str(e)}, 500, request_id=request_id)
                return
        infer_exit()
        latency_ms = int((time.time() - t0) * 1000)
        log_event('gen_response', ip, 'POST', '/generate', request_id=request_id,
                  status=200, latency_ms=latency_ms,
                  response_body=text[:500] if text else '')
        self._json_response({'text': text}, 200, request_id=request_id)

    # ── P4 endpoints ──────────────────────────────────────────────────────────

    def _parse_multipart(self, body: bytes):
        """Parse a multipart/form-data body manually.
        Returns (filename, file_bytes) or raises ValueError.

        Parses the boundary from Content-Type, splits the body on boundary
        markers, and returns the first part that has a filename= in its
        Content-Disposition header.
        """
        content_type = self.headers.get('Content-Type', '')
        if 'multipart/form-data' not in content_type:
            raise ValueError(f'Expected multipart/form-data, got: {content_type!r}')

        # Extract boundary from Content-Type: multipart/form-data; boundary=XXXX
        boundary = None
        for seg in content_type.split(';'):
            seg = seg.strip()
            if seg.lower().startswith('boundary='):
                boundary = seg[9:].strip().strip('"\'')
                break
        if not boundary:
            raise ValueError(f'No boundary in Content-Type: {content_type!r}')

        # Split body on --boundary (RFC 2046 uses CRLF--boundary)
        delim = ('--' + boundary).encode('ascii')
        parts = body.split(delim)

        for part in parts:
            # Each part: \r\n<headers>\r\n\r\n<content>\r\n
            # Strip leading \r\n
            if part.startswith(b'\r\n'):
                part = part[2:]
            # Skip the final boundary terminator '--\r\n'
            if part in (b'--', b'--\r\n', b''):
                continue

            # Split part headers from content at the first blank line (\r\n\r\n)
            sep = b'\r\n\r\n'
            sep_idx = part.find(sep)
            if sep_idx < 0:
                continue
            raw_headers = part[:sep_idx].decode('utf-8', 'replace')
            content = part[sep_idx + 4:]
            # Content ends with \r\n before the next boundary; strip trailing \r\n
            if content.endswith(b'\r\n'):
                content = content[:-2]

            # Parse Content-Disposition to find filename
            fname = None
            for header_line in raw_headers.splitlines():
                hl = header_line.strip()
                if hl.lower().startswith('content-disposition:'):
                    for token in hl.split(';'):
                        token = token.strip()
                        if token.lower().startswith('filename='):
                            fname = token[9:].strip().strip('"\'')
                            break
                if fname:
                    break

            if fname:
                return fname, content

        raise ValueError('No file part with filename= found in multipart body')

    def _handle_ingest(self, body: bytes, request_id: str):
        """POST /ingest -- multipart file upload or JSON {path, content}.

        Saves the file under ragdocs/, inserts a documents row (status='vectorizing'),
        fires off a background thread for the ingestion pipeline, and returns
        immediately with {doc_id, status, path}.
        """
        ip = self.client_address[0]
        content_type = self.headers.get('Content-Type', '')

        # ── Parse the incoming file ──────────────────────────────────────────
        filename = None
        file_bytes = None

        if 'multipart/form-data' in content_type:
            try:
                filename, file_bytes = self._parse_multipart(body)
            except ValueError as e:
                self._json_response({'error': str(e)}, 400, request_id=request_id)
                return
        elif 'application/json' in content_type or not content_type:
            # JSON: {"path": "relative/name.txt", "content": "<text>"}
            try:
                obj = json.loads(body)
                filename = obj.get('path') or obj.get('filename')
                content  = obj.get('content', '')
                if not filename:
                    raise ValueError('"path" or "filename" required in JSON body')
                file_bytes = content.encode('utf-8') if isinstance(content, str) else content
            except (json.JSONDecodeError, ValueError, AttributeError) as e:
                self._json_response({'error': str(e)}, 400, request_id=request_id)
                return
        else:
            self._json_response(
                {'error': 'Unsupported Content-Type; use multipart/form-data or application/json'},
                415, request_id=request_id)
            return

        if not filename:
            self._json_response({'error': 'No filename provided'}, 400, request_id=request_id)
            return

        # Sanitise: strip leading path separators / drive letters to avoid escaping ragdocs/
        rel_path = filename.replace('\\', '/').lstrip('/')
        if not rel_path:
            self._json_response({'error': 'Empty filename after sanitisation'}, 400,
                                request_id=request_id)
            return

        # ── Save file under ragdocs/ ─────────────────────────────────────────
        dest_path = os.path.join(RAGDOCS_DIR, rel_path.replace('/', os.sep))
        dest_dir  = os.path.dirname(dest_path)
        os.makedirs(dest_dir if dest_dir else RAGDOCS_DIR, exist_ok=True)
        with open(dest_path, 'wb') as f:
            f.write(file_bytes)

        # ── Insert documents row with status='vectorizing' ───────────────────
        ts = _now_iso()
        try:
            with _db_lock:
                cur = _db_conn.execute(
                    "INSERT OR REPLACE INTO documents(path, status, ts) VALUES(?,?,?)",
                    (rel_path, 'vectorizing', ts),
                )
                doc_id = cur.lastrowid
                _db_conn.commit()
        except Exception as e:
            self._json_response({'error': f'DB insert failed: {e}'}, 500,
                                request_id=request_id)
            return

        log_event('ingest_queued', ip, 'POST', '/ingest', request_id=request_id,
                  doc_id=doc_id, rel_path=rel_path)

        # ── Kick off background ingestion ────────────────────────────────────
        t = threading.Thread(
            target=_ingest_background,
            args=(doc_id, dest_path, rel_path, request_id),
            daemon=True,
        )
        t.start()

        self._json_response({'doc_id': doc_id, 'status': 'vectorizing', 'path': rel_path},
                            200, request_id=request_id)

    def _handle_tree(self, request_id: str):
        """GET /tree -- return the ingested corpus as JSON.

        Returns a list of file entries under ragdocs/, each with:
            path        : relative path under ragdocs/
            status      : 'vectorizing' | 'ready' | 'error' | 'pending'
            chunk_count : number of stored chunks (from DB n_chunks column)
        """
        ip = self.client_address[0]
        if _db_conn is None:
            self._json_response({'error': 'DB not initialised'}, 503, request_id=request_id)
            return

        try:
            with _db_lock:
                rows = _db_conn.execute(
                    "SELECT path, status, n_chunks FROM documents ORDER BY path"
                ).fetchall()
        except Exception as e:
            self._json_response({'error': str(e)}, 500, request_id=request_id)
            return

        files = []
        for path, status, n_chunks in rows:
            files.append({
                'path':        path,
                'status':      status,
                'chunk_count': n_chunks or 0,
            })

        self._json_response({'files': files}, 200, request_id=request_id)

    def _handle_doc(self, query: str, request_id: str):
        """GET /doc?path=<rel_path> -- return stored text of an ingested file.

        Returns {path, status, chunks: [{chunk_index, text, char_start, char_end}, ...]}.
        If the file is still vectorizing, returns what chunks are available so far.
        """
        ip = self.client_address[0]
        if _db_conn is None:
            self._json_response({'error': 'DB not initialised'}, 503, request_id=request_id)
            return

        # Parse ?path= from query string
        params = parse_qs(query or '')
        rel_path_list = params.get('path', [])
        if not rel_path_list:
            self._json_response({'error': 'path parameter required'}, 400,
                                request_id=request_id)
            return
        rel_path = rel_path_list[0]

        try:
            with _db_lock:
                doc_row = _db_conn.execute(
                    "SELECT id, path, status, n_chunks FROM documents WHERE path=?",
                    (rel_path,)
                ).fetchone()
        except Exception as e:
            self._json_response({'error': str(e)}, 500, request_id=request_id)
            return

        if doc_row is None:
            self._json_response({'error': f'Document not found: {rel_path!r}'}, 404,
                                request_id=request_id)
            return

        doc_id, path, status, n_chunks = doc_row

        try:
            with _db_lock:
                chunk_rows = _db_conn.execute(
                    "SELECT chunk_index, text, char_start, char_end "
                    "FROM chunks WHERE doc_id=? ORDER BY chunk_index",
                    (doc_id,)
                ).fetchall()
        except Exception as e:
            self._json_response({'error': str(e)}, 500, request_id=request_id)
            return

        chunks_out = [
            {'chunk_index': ci, 'text': t, 'char_start': cs, 'char_end': ce}
            for ci, t, cs, ce in chunk_rows
        ]

        self._json_response({
            'path':        path,
            'status':      status,
            'n_chunks':    n_chunks,
            'chunks':      chunks_out,
        }, 200, request_id=request_id)

    def _handle_retrieve(self, body: bytes, request_id: str):
        """POST /retrieve {"question": "...", "scope": "<optional path>", "k": <optional int>}

        Embeds the question through the embed gate using the 'search_query: '
        prefix (required by nomic-embed-text v1.5 for asymmetric retrieval --
        document chunks were stored with 'search_document: '; these two prefixes
        must match for the vectors to be comparable), then runs sqlite-vec k-NN
        to return the top-k most relevant chunks.

        If 'scope' is given, results are restricted to chunks whose document
        path equals that value (exact file) or starts with it followed by '/'
        (folder prefix).  The filtering is done by db.knn_chunks_with_score so
        that the scope never requires a second round-trip to the embedder.

        Returns JSON:
          {
            "hits": [
              {
                "path":        "<relative path under ragdocs/>",
                "chunk_index": <int>,
                "char_start":  <int>,
                "char_end":    <int>,
                "text":        "<chunk text>",
                "distance":    <float>   // L2; lower = closer
              },
              ...
            ]
          }

        This is the contract that P6 (answer) and the web UI consume.
        """
        ip = self.client_address[0]

        # ── Parse request body ────────────────────────────────────────────────
        try:
            req = json.loads(body)
        except (json.JSONDecodeError, AttributeError):
            self._json_response({'error': 'Invalid JSON body'}, 400,
                                request_id=request_id)
            return

        question = req.get('question', '').strip()
        if not question:
            self._json_response({'error': '"question" must be a non-empty string'},
                                400, request_id=request_id)
            return

        k     = req.get('k', 5)
        scope = req.get('scope')   # optional str; None means no scope filter

        if not isinstance(k, int) or k < 1:
            self._json_response({'error': '"k" must be a positive integer'},
                                400, request_id=request_id)
            return

        if _db_conn is None:
            self._json_response({'error': 'DB not initialised'}, 503,
                                request_id=request_id)
            return

        # ── Log the incoming retrieve request ─────────────────────────────────
        log_event('retrieve_request', ip, 'POST', '/retrieve',
                  request_id=request_id,
                  request_body=json.dumps({
                      'question': question[:200],
                      'scope':    scope,
                      'k':        k,
                  }))

        t0 = time.time()

        # ── Step 1: embed the question (search_query: prefix for nomic v1.5) ──
        # nomic-embed-text v1.5 uses task prefixes for asymmetric retrieval.
        # Documents were stored with 'search_document: '; queries MUST use
        # 'search_query: ' so they project into the same embedding space.
        prefixed_question = 'search_query: ' + question
        log_event('embed_request', ip, 'POST', '/retrieve',
                  request_id=request_id,
                  request_body=prefixed_question[:200])
        embed_t0 = time.time()
        with embed_gate:
            try:
                query_vec = _embed_backend.embed(prefixed_question)
            except Exception as e:
                log_event('embed_response', ip, 'POST', '/retrieve',
                          request_id=request_id, status=500, error=str(e))
                self._json_response({'error': f'Embedding failed: {e}'}, 500,
                                    request_id=request_id)
                return
        embed_ms = int((time.time() - embed_t0) * 1000)
        log_event('embed_response', ip, 'POST', '/retrieve',
                  request_id=request_id, status=200,
                  dims=len(query_vec), latency_ms=embed_ms)

        # ── Step 2: sqlite-vec k-NN (no gate -- pure SQLite, no model) ────────
        try:
            with _db_lock:
                hits_raw = _db_mod.knn_chunks_with_score(
                    _db_conn, query_vec, k=k, scope=scope
                )
        except Exception as e:
            log_event('retrieve_response', ip, 'POST', '/retrieve',
                      request_id=request_id, status=500, error=str(e))
            self._json_response({'error': f'k-NN query failed: {e}'}, 500,
                                request_id=request_id)
            return

        # ── Step 3: fetch citation metadata for each hit ──────────────────────
        hits_out = []
        try:
            with _db_lock:
                for chunk_id, distance in hits_raw:
                    meta = _db_mod.get_chunk_by_id(_db_conn, chunk_id)
                    if meta is None:
                        continue
                    hits_out.append({
                        'path':        meta['path'],
                        'chunk_index': meta['chunk_index'],
                        'char_start':  meta['char_start'],
                        'char_end':    meta['char_end'],
                        'text':        meta['text'],
                        'distance':    distance,
                    })
        except Exception as e:
            log_event('retrieve_response', ip, 'POST', '/retrieve',
                      request_id=request_id, status=500, error=str(e))
            self._json_response({'error': f'Chunk fetch failed: {e}'}, 500,
                                request_id=request_id)
            return

        latency_ms = int((time.time() - t0) * 1000)
        response_payload = {'hits': hits_out}

        log_event('retrieve_response', ip, 'POST', '/retrieve',
                  request_id=request_id, status=200,
                  n_hits=len(hits_out), scope=scope, latency_ms=latency_ms,
                  response_body=json.dumps({
                      'n_hits': len(hits_out),
                      'top1_path': hits_out[0]['path'] if hits_out else None,
                      'top1_chunk_index': hits_out[0]['chunk_index'] if hits_out else None,
                  }))

        self._json_response(response_payload, 200, request_id=request_id)

    def _handle_shutdown(self, method, query, request_id):
        """POST /shutdown (unconditional) or GET /shutdown?<UTC-timestamp>."""
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
    print('[serve] --cli: terminal chat not yet implemented (P7 stub).')
    print('[serve] Web server is still running.  Press Ctrl-C to stop.')


##############################################################################
# Argument parsing -- NO argparse; a small manual loop keeps deps to the stdlib
##############################################################################

COMMAND_LINE_HELP = """\
[serve] slm-rag serve.py -- command-line args:
  --web        Run the web server only.  This is the default.
  --cli        Run the web server and attach a terminal chat (stub in P3; full in P7).
  --port <n>   Listen port (default 51548).  This flag is the only override.
  --db <path>  Path to rag.db SQLite file (default BASE_DIR/rag.db).  No env var.
  --check      Print the planned backend layout and exit.  No downloads, no
               models start, no socket opened.
  --help       Print this help and exit.

No environment variables are used; every setting has a baked-in default.

Backends:
  Embedder  nomic-embed-text-v1.5 Q4_K_M  port 52852  CPU-only (n-gpu-layers 0)
  Phi       Phi-4-mini Q4_K_M              port 52851  GPU if VRAM fits, else CPU
"""


def parse_args(argv):
    mode         = 'web'
    mode_flags   = []
    port         = DEFAULT_PORT
    db_path      = DEFAULT_DB
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
        elif arg == '--db':
            i += 1
            if i >= len(argv):
                print('[serve] --db requires a path, e.g. --db /tmp/test.db', flush=True)
                sys.exit(2)
            db_path = argv[i]
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

    return {'mode': mode, 'port': port, 'db_path': db_path, 'check': check,
            'help': help_wanted}


##############################################################################
# Entry point
##############################################################################

def main():
    global _db_conn

    args = parse_args(sys.argv[1:])

    if args['help']:
        print(COMMAND_LINE_HELP.rstrip(), flush=True)
        return

    port    = args['port']
    db_path = args['db_path']

    if args['check']:
        describe_plan(port)
        return       # no socket, no downloads, exit 0

    # ── initialise the SQLite database (P1: init_db creates all tables) ───────
    print(f'[serve] opening rag.db at {db_path}', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    os.makedirs(RAGDOCS_DIR, exist_ok=True)
    _db_conn = _db_mod.init_db(db_path)
    print('[serve] rag.db ready', flush=True)

    # ── ensure llama-server binary is present (Windows: download if missing) ─
    ensure_llama_server()

    # ── download weights (embedder first, then Phi) ───────────────────────────
    ensure_weights()

    # ── build and boot both backends ──────────────────────────────────────────
    build_backends()
    boot_backends()

    # ── start the HTTP server ─────────────────────────────────────────────────
    try:
        server = ThreadedHTTPServer((HOST, port), RagHandler)
    except OSError as e:
        print(f'[serve] ERROR: cannot bind to {HOST}:{port} -- {e}', flush=True)
        print(f'[serve] Try a different port: --port 51549 (or any free port)', flush=True)
        stop_backends()
        sys.exit(1)

    # SIGINT / SIGTERM: drain the HTTP server and exit cleanly.
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
              port=port, version='p5-retrieve', db_path=db_path)

    if args['mode'] == 'cli':
        cli_stub(f'http://{HOST}:{port}')
        begin_shutdown(server)

    # Block until server_thread finishes (i.e. server.shutdown() was called).
    server_thread.join()
    print('[serve] server stopped.', flush=True)


if __name__ == '__main__':
    main()
