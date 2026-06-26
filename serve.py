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
slm-rag serve.py -- P6: grounded answer + citations.

Extends P5 to add the full chat flow:

  - POST /enqueue {"kind":"chat","content":"<question>","session_id":"<optional>"}:
    creates a requests row (kind=chat) and a messages row for the user turn;
    mints a session_id if none given; returns {request_id, session_id} immediately.

  - A single worker thread drains pending chat requests in order (generation is
    serialized by the gen gate so exactly one answer is produced at a time):
      1. Embed the question (embed gate, 'search_query: ' prefix) and retrieve
         top-k chunks.
      2. Build a grounded prompt from SYSTEM_PROMPT + retrieved context +
         recent conversation history.
      3. Insert an assistant messages row with status='streaming', stream Phi's
         tokens into that row, then mark it done.
      4. Log the chain: embed_request/embed_response and gen_request/gen_response
         lines, all carrying the request id and session id.

  - GET /request?id=<rid>: return {status, error} so the client can poll.

  - GET /history?session_id=<sid>: return the ordered conversation messages.

P5 endpoints retained: POST /retrieve.
P4 endpoints retained: POST /ingest, GET /tree, GET /doc.
P3 endpoints retained: POST /embed, POST /generate, GET /health.

EMBEDDING NOTE: nomic-embed-text v1.5 uses task prefixes.  Document chunks are
prefixed with 'search_document: ' before embedding.  Retrieval MUST use
'search_query: ' for the question vector so they live in the same embedding space.

PROMPT NOTE: SYSTEM_PROMPT and build_prompt() below define the exact shape sent
to Phi at inference.  training/finetune_phi_rag.ipynb MUST mirror these exactly
so training reinforces the same grounded, cited behaviour the server expects.

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
# P6 -- Grounded prompt constants
#
# SYSTEM_PROMPT and build_prompt() define the exact chat template sent to Phi
# at inference time.  training/finetune_phi_rag.ipynb MUST mirror these exactly:
# fine-tuning and serving share the same template so the model reinforces the
# grounded, cited behaviour that the server expects here.
#
# Citation format: [N] where N is the 1-based index of the numbered data chunk.
# Chunks are numbered [1]..[N] in retrieval order (most relevant first).
# The model is instructed to cite chunks by number, e.g. [1] or [2][3].
# The "I don't know" phrase is fixed so tests can match it reliably.
#
# DESIGN (chosen from the Colab model x prompt eval, 2025-06):
#   The full grounding instruction lives ONLY in SYSTEM_PROMPT.  The user
#   turn carries numbered chunks + bare "Question: ..." with NO repeated
#   footer.  The eval showed that a redundant instruction footer caused
#   over-refusal on answers stated in narrative/dialogue; moving it
#   exclusively to the system message fixes this while preserving correct
#   "I don't know" behaviour on genuinely out-of-corpus questions.
#
# NOTE: This prompt format MUST stay in sync with training/finetune_phi_rag.ipynb.
##############################################################################

SYSTEM_PROMPT = (
    "You answer using ONLY the numbered chunks below. "
    "The answer may be stated indirectly, in narration or dialogue -- extract it and state it plainly "
    "in 1-3 sentences in your own words. "
    "Cite the chunk number(s) you used like [1] or [2][3]. "
    "Only if none of the chunks are relevant, reply with exactly: "
    "\"I don't know based on the provided documents.\""
)

# Number of recent conversation turns to include in the prompt for multi-turn
# context.  Each turn is one user message + one assistant message.  Capped to
# avoid overflowing Phi's 8192-token context when chunks are large.
PROMPT_TURNS = 3

# Default number of chunks to retrieve for each chat question.
CHAT_TOP_K = 5


def build_prompt(question: str, hits: list[dict], history: list[dict]) -> str:
    """Assemble the user-turn content that is sent to Phi alongside SYSTEM_PROMPT.

    Context chunks are numbered [1]..[N] in retrieval order (most relevant first).
    The model cites chunks by number (e.g. [1] or [2][3]).

    DESIGN: The full grounding instruction lives in SYSTEM_PROMPT only.  The
    user turn contains numbered chunks + bare "Question: ..." with NO repeated
    footer.  This was chosen from the Colab model x prompt eval (2025-06):
    placing the instruction solely in the system message fixes over-refusal on
    answers stated in narrative/dialogue while preserving correct "I don't know"
    behaviour on genuinely out-of-corpus questions.

    NOTE: This format MUST stay in sync with training/finetune_phi_rag.ipynb.

    Parameters
    ----------
    question : the current user question (bare text)
    hits     : list of chunk dicts returned by knn_chunks_with_score / get_chunk_by_id;
               each must have keys 'path', 'chunk_index', and 'text'.
               hits[0] is the MOST relevant (closest k-NN hit).
    history  : recent messages from get_messages() (excluding the current user turn);
               only the last PROMPT_TURNS * 2 rows are used

    Returns
    -------
    The string to use as the 'user' role content in the chat completion request.
    Numbered chunks appear first ([1] = most relevant), then the bare Question
    line.  No instruction footer -- the instruction lives solely in SYSTEM_PROMPT.
    """
    # Build numbered context chunks: [1] = most relevant (retrieval order).
    chunk_lines = []
    for i, hit in enumerate(hits, start=1):
        chunk_lines.append(
            f"[{i}] (source: {hit['path']}, chunk {hit['chunk_index']})\n{hit['text']}"
        )
    context_block = '\n\n'.join(chunk_lines) if chunk_lines else '[no context retrieved]'

    # Include recent conversation history so the model can answer follow-up
    # questions that refer back to earlier turns.  We limit to the last
    # PROMPT_TURNS * 2 message rows (one user + one assistant per turn).
    history_block = ''
    recent = [m for m in history if m['role'] in ('user', 'assistant')]
    recent = recent[-(PROMPT_TURNS * 2):]
    if recent:
        lines = []
        for m in recent:
            prefix = 'Human' if m['role'] == 'user' else 'Assistant'
            # Use clean answer text for history if available, else full content
            content = m.get('answer') or m.get('content') or ''
            lines.append(f"{prefix}: {content}")
        history_block = '\n'.join(lines) + '\n\n'

    # User content: history (if any) + numbered chunks + bare question.
    # No instruction footer -- the instruction lives solely in SYSTEM_PROMPT.
    return (
        f"{history_block}"
        f"{context_block}\n\n"
        f"Question: {question}"
    )


def build_prompt_pieces(question: str, hits: list[dict], history: list[dict]) -> list[str]:
    """Return the prompt as an ordered list of pieces for paced streaming.

    Pieces (in order):
      1. History block (if any) -- one piece
      2. Each numbered chunk [1]..[N] -- one piece per chunk
      3. Bare question -- one piece (no instruction footer; instruction is in SYSTEM_PROMPT)

    The caller streams these into the DB one by one with a small sleep between
    pieces so the browser sees the prompt build up visually.
    """
    pieces = []

    # History block
    recent = [m for m in history if m['role'] in ('user', 'assistant')]
    recent = recent[-(PROMPT_TURNS * 2):]
    if recent:
        lines = []
        for m in recent:
            prefix = 'Human' if m['role'] == 'user' else 'Assistant'
            content = m.get('answer') or m.get('content') or ''
            lines.append(f"{prefix}: {content}")
        pieces.append('\n'.join(lines) + '\n\n')

    # Numbered chunks (most relevant first)
    for i, hit in enumerate(hits, start=1):
        pieces.append(
            f"[{i}] (source: {hit['path']}, chunk {hit['chunk_index']})\n{hit['text']}\n\n"
        )

    # Bare question -- no instruction footer (instruction lives solely in SYSTEM_PROMPT).
    pieces.append(f"Question: {question}")

    return pieces


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

# Headroom over GPU weight size for KV cache + compute buffers at 8192 context.
# Measured on Qwen2.5-7B Q4_K_M: KV 448 MiB + compute 494 MiB ~= 0.93 GiB at 8192
# (small KV thanks to 4-head GQA); 1.2 GiB reserves a ~0.27 GiB cushion.
GPU_HEADROOM_GB = 1.2


def gpu_layers_for_phi():
    """How many layers to offload the Phi model to the GPU.
    '99' = all layers (GPU), '0' = CPU-only.

    Rules (no env var override -- slm-rag uses flags only):
      macOS   -> '99' (Apple Metal, unified memory)
      NVIDIA  -> '99' if the gen model (~4.1 GB GPU weights + headroom) fits in free VRAM, else '0'
      no GPU  -> '0'

    The free reading is taken AFTER reap_orphan_backends() (see main()), which kills
    any leftover llama-server -- we are the only llama.cpp process, so any other one is
    an orphan holding VRAM (the "trick"). Browser/desktop VRAM use is real and IS
    respected, so we read free (not total) once the orphans are gone. A genuine OOM
    still falls back to CPU via ProxyBackend.boot().
    """
    if sys.platform == 'darwin':
        return '99'
    if GPU_FREE_GB is None:
        return '0'           # no NVIDIA GPU detected
    # Qwen2.5-7B Q4_K_M puts ~4.07 GiB of weights on the GPU (token_embd stays on CPU);
    # GPU_HEADROOM_GB covers the ~0.93 GiB KV+compute at 8192 ctx. ~5.0 GiB total fits 6 GB.
    gen_need = 4.1 + GPU_HEADROOM_GB
    return '99' if gen_need <= GPU_FREE_GB else '0'


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
        # Generative-model slot (key kept as 'phi' for the code that references it).
        # Now Qwen2.5-7B-Instruct (Apache-2.0), chosen from the 2026-06 Colab model x
        # prompt eval: it EXTRACTS answers stated indirectly in narrative/dialogue
        # (e.g. "a stake is survivable" -> "no, it doesn't kill the vampire") that
        # Phi-4-mini over-refuses, while still refusing out-of-corpus questions exactly.
        # Q4_K_M fits a 6 GB GPU at 8192 ctx (~5.0 GiB: 4.07 weights + 0.45 KV + 0.49
        # compute; small KV thanks to 4-head GQA). Base/uncustomized (no persona
        # fine-tune). NOTE: training/finetune_phi_rag.ipynb still targets Phi-4-mini --
        # revisit the fine-tune target if the self-improvement loop moves to Qwen.
        'repo':      'bartowski/Qwen2.5-7B-Instruct-GGUF',
        'filename':  'Qwen2.5-7B-Instruct-Q4_K_M.gguf',
        'local':     os.path.join(BASE_DIR, 'model', 'qwen25-7b',
                                  'Qwen2.5-7B-Instruct-Q4_K_M.gguf'),
        'approx_gb': 4.36,
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


def _embed_resilient(text, attempts=3, timeout=120):
    """Embed with retries and a generous timeout.

    The embedder is CPU-bound (the GPU is reserved for Phi), so under a load spike
    a single embed can transiently exceed the default timeout. One blip should not
    fail a whole ingest or a user's question, so we retry a few times before giving
    up. Used by both the ingest worker and the chat query-embed."""
    last = None
    for attempt in range(attempts):
        try:
            return _embed_backend.embed(text, timeout=timeout)
        except Exception as e:
            last = e
            print(f'[serve] embed attempt {attempt + 1}/{attempts} failed ({e}); '
                  f'retrying ...', flush=True)
            time.sleep(1.0)
    raise last


def _ingest_background(doc_id: int, rel_path: str, request_id: str):
    """
    Background worker: read blob -> extract -> chunk -> embed (in batches,
    yielding the embed gate between batches) -> store chunks + vectors ->
    set doc status ready/error.

    Runs in a daemon thread; logs errors to stdout and sets doc status='error'.

    EMBEDDING PREFIX: nomic-embed-text v1.5 requires task-type prefixes for
    asymmetric retrieval quality.  Each document chunk is prefixed with
    'search_document: ' here.  P5 retrieval MUST use 'search_query: ' for the
    question so query and document vectors live in the same embedding space.
    """
    try:
        # ── Step 1: read the blob from the DB ────────────────────────────────
        with _db_lock:
            data = _db_mod.get_document_blob(_db_conn, rel_path)
        if data is None:
            raise RuntimeError(f'Blob not found in DB for path={rel_path!r}')

        # ── Step 2: extract text and chunk ───────────────────────────────────
        text = _ingest.extract_text_from_bytes(rel_path, data)
        chunks = _ingest.chunk_text(text)
        for chunk in chunks:
            chunk['source_path'] = rel_path

        # Update n_chunks in the documents row now that we know the count
        with _db_lock:
            _db_conn.execute(
                "UPDATE documents SET n_chunks=? WHERE id=?",
                (len(chunks), doc_id),
            )
            _db_conn.commit()

        # ── Step 3: embed in batches, yielding the embed gate between batches ─
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
                    vector = _embed_resilient(prefixed_text)
                    embedded_batch.append((chunk, vector))
            # embed_gate is released here -- next /embed or next batch can proceed

            # ── Step 4: store each chunk + vector ────────────────────────────
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

        # ── Step 5: mark document ready ──────────────────────────────────────
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
        log_event('ingest_error', '-', 'WORKER', '/ingest',
                  request_id=request_id, error=str(exc))
        # A document that did not finish ingesting should not show in the tree.
        # Remove it (and any partial chunks) entirely -- the full error is in the
        # logs/ audit trail, so nothing is lost, but the user never sees a broken row.
        try:
            with _db_lock:
                _db_mod.delete_document(_db_conn, rel_path)
        except Exception:
            pass


##############################################################################
# P6 -- Chat worker thread
#
# A single daemon thread drains pending 'chat' requests from the requests
# table one at a time.  Serializing on a single thread (rather than one thread
# per request) means the gen gate is never contended by two concurrent chat
# workers, which would make one of them wait anyway -- single-threading is both
# simpler and equally fast.
#
# Flow for each chat request:
#   1. Embed the question (embed gate, 'search_query: ' prefix).
#   2. k-NN retrieval (no gate -- pure SQLite).
#   3. Build grounded prompt pieces (numbered [1]..[N] chunks, most relevant first).
#   4. Insert streaming assistant message row (content='', status='streaming').
#   5a. PACED PROMPT STREAMING: write each prompt piece into content with a
#       small time.sleep(~0.12) so the browser sees the prompt build up visually.
#   5b. GENERATION STREAMING: stream Phi's token deltas into content via
#       generate_grounded_streaming(); flush DB at most every ~80ms or ~20 tokens.
#   6. On done: set status='done', store clean answer in `answer` column, store
#      numbered references JSON in `references` column.
#   7. Log embed_request/embed_response + gen_request/gen_response (COMPLETE,
#      untruncated) with both the request id and session id.
##############################################################################

# Signal the worker to stop when the server is shutting down.
_worker_stop = threading.Event()

# How long to sleep between paced prompt pieces (seconds).
_PROMPT_PIECE_SLEEP = 0.12

# How often to flush streaming token accumulation to the DB (seconds).
_TOKEN_FLUSH_INTERVAL = 0.08

# How many tokens to accumulate before a forced DB flush (fallback bound).
_TOKEN_FLUSH_COUNT = 20


def _chat_worker():
    """Daemon thread: drain pending chat requests, one at a time."""
    while not _worker_stop.is_set():
        # Pause briefly when the DB or backends are not yet ready.
        if _db_conn is None or _embed_backend is None or _gen_backend is None:
            time.sleep(1.0)
            continue

        # Peek at the next pending chat request under the lock.
        with _db_lock:
            req = _db_mod.next_pending_request(_db_conn)

        if req is None:
            # Nothing to do; sleep and try again.
            time.sleep(0.5)
            continue

        # Mark the request running so it won't be picked up twice.
        with _db_lock:
            _db_mod.mark_request(_db_conn, req['id'], 'running')

        request_id = req['request_id']
        session_id = req['session_id']
        ip         = '127.0.0.1'   # worker-internal; no real client IP

        try:
            content_obj = json.loads(req['content'])
        except (json.JSONDecodeError, TypeError):
            content_obj = {}
        question = content_obj.get('question', '')

        print(f'[worker] processing request_id={request_id} '
              f'session_id={session_id} question={question[:80]!r}', flush=True)

        msg_id = None  # set in Step 4; used in error handler

        try:
            # ── Step 1: embed the question (embed gate) ───────────────────────
            prefixed = 'search_query: ' + question
            log_event('embed_request', ip, 'WORKER', '/enqueue',
                      request_id=request_id, chat_session_id=session_id,
                      request_body=prefixed)
            embed_t0 = time.time()
            with embed_gate:
                query_vec = _embed_resilient(prefixed)
            embed_ms = int((time.time() - embed_t0) * 1000)
            log_event('embed_response', ip, 'WORKER', '/enqueue',
                      request_id=request_id, chat_session_id=session_id,
                      status=200, dims=len(query_vec), latency_ms=embed_ms)

            # ── Step 2: k-NN retrieval (no gate -- pure SQLite) ───────────────
            with _db_lock:
                hits_raw = _db_mod.knn_chunks_with_score(
                    _db_conn, query_vec, k=CHAT_TOP_K
                )
                hits = []
                for chunk_id, distance in hits_raw:
                    meta = _db_mod.get_chunk_by_id(_db_conn, chunk_id)
                    if meta is not None:
                        meta['distance'] = distance
                        hits.append(meta)

            # ── Step 3: build grounded prompt ─────────────────────────────────
            with _db_lock:
                history = _db_mod.get_messages(_db_conn, session_id)
            # Exclude the current user turn (last row) from history so the model
            # does not see a duplicate of the question in the history block.
            prior_history = [m for m in history if m['role'] != 'user' or
                             m['content'] != question]
            user_content = build_prompt(question, hits, prior_history)
            prompt_pieces = build_prompt_pieces(question, hits, prior_history)

            # ── Step 4: insert streaming assistant message row ────────────────
            with _db_lock:
                msg_id = _db_mod.insert_message(
                    _db_conn,
                    session_id=session_id,
                    role='assistant',
                    content='',
                    request_id=request_id,
                    status='streaming',
                )

            # ── Step 5a: paced prompt streaming ───────────────────────────────
            # Write each prompt piece into the message content with a small
            # sleep between pieces so the browser sees the numbered chunks
            # appear one by one.  The browser polls /history at ~150-200ms so
            # this pace matches its rendering cadence.
            accumulated = ''
            for piece in prompt_pieces:
                accumulated += piece
                with _db_lock:
                    _db_mod.update_message(
                        _db_conn, msg_id, accumulated, status='streaming'
                    )
                time.sleep(_PROMPT_PIECE_SLEEP)

            # Separator between prompt and generation output
            accumulated += '\n\n--- Generating answer ---\n\n'
            with _db_lock:
                _db_mod.update_message(
                    _db_conn, msg_id, accumulated, status='streaming'
                )

            # ── Step 5b: streaming generation ─────────────────────────────────
            # Log the COMPLETE prompt sent to Phi (full system + full user content)
            # -- accurate, untruncated records of every model prompt are required.
            log_event('gen_request', ip, 'WORKER', '/enqueue',
                      request_id=request_id, chat_session_id=session_id,
                      request_body=json.dumps({
                          'system':  SYSTEM_PROMPT,
                          'user':    user_content,
                          'n_hits':  len(hits),
                      }, ensure_ascii=False))

            gen_t0 = time.time()
            infer_enter()

            # For streaming: accumulate tokens and flush to DB every ~80ms or
            # every ~20 tokens to bound the number of DB writes without making
            # the browser wait too long for each token update.
            token_buf     = ''
            last_flush    = time.time()
            tok_since_flush = 0

            def _on_token(token):
                nonlocal accumulated, token_buf, last_flush, tok_since_flush
                token_buf += token
                tok_since_flush += 1
                now = time.time()
                if (now - last_flush >= _TOKEN_FLUSH_INTERVAL or
                        tok_since_flush >= _TOKEN_FLUSH_COUNT):
                    accumulated += token_buf
                    token_buf = ''
                    tok_since_flush = 0
                    last_flush = now
                    with _db_lock:
                        _db_mod.update_message(
                            _db_conn, msg_id, accumulated, status='streaming'
                        )

            try:
                # Use streaming if the gen backend supports it (ProxyBackend on Windows).
                if hasattr(_gen_backend, 'generate_grounded_streaming'):
                    with gen_gate:
                        reply, n_tokens = _gen_backend.generate_grounded_streaming(
                            SYSTEM_PROMPT, user_content,
                            token_callback=_on_token,
                            max_tokens=512,
                        )
                else:
                    # InProc fallback (Linux): non-streaming, update once at the end.
                    with gen_gate:
                        reply, n_tokens = _gen_backend.generate_grounded(
                            SYSTEM_PROMPT, user_content, max_tokens=512
                        )
                    # Simulate token callback for the flush path below
                    token_buf = reply
            finally:
                infer_exit()

            gen_ms = int((time.time() - gen_t0) * 1000)

            # Flush any remaining token buffer
            if token_buf:
                accumulated += token_buf
                token_buf = ''

            log_event('gen_response', ip, 'WORKER', '/enqueue',
                      request_id=request_id, chat_session_id=session_id,
                      status=200, n_tokens=n_tokens, latency_ms=gen_ms,
                      response_body=reply or '')

            # ── Step 6: mark done, store clean answer + references ────────────
            # Build numbered references list matching the chunks used in the prompt.
            refs = [
                {
                    'n':           i,
                    'path':        hit['path'],
                    'chunk_index': hit['chunk_index'],
                    'text':        hit['text'],
                }
                for i, hit in enumerate(hits, start=1)
            ]
            refs_json = json.dumps(refs, ensure_ascii=False)

            with _db_lock:
                _db_mod.update_message(
                    _db_conn, msg_id, accumulated,
                    status='done', n_tokens=n_tokens, gen_ms=gen_ms,
                    answer=reply, references=refs_json,
                )
                _db_mod.mark_request(_db_conn, req['id'], 'done')

            print(f'[worker] done request_id={request_id} '
                  f'n_tokens={n_tokens} gen_ms={gen_ms}', flush=True)

        except Exception as exc:
            err = str(exc)[:1000]
            print(f'[worker] ERROR request_id={request_id}: {err}', flush=True)
            try:
                with _db_lock:
                    if msg_id is not None:
                        _db_mod.update_message(
                            _db_conn, msg_id,
                            f'[Error: {err}]',
                            status='error',
                        )
                    _db_mod.mark_request(_db_conn, req['id'], 'error', error=err)
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

    def _warmup(self):
        """Run one real inference so the first user request isn't a cold start.

        A healthy /health can precede the model being able to actually serve: the
        embedder's first /v1/embeddings call is the cold one, and under load it can
        otherwise blow past the request timeout and fail a user's first question.
        We warm only the embedder here -- the generator warms on its first answer
        and its generate timeout is generous. Retries patiently over a deadline."""
        if self.name != 'embedder':
            return
        deadline = time.time() + 180
        last_err = None
        while time.time() < deadline:
            try:
                vec = self.embed('warmup', timeout=120)
                if vec:
                    print(f'[serve] {self.name} warmup ok ({len(vec)} dims)', flush=True)
                    return
                last_err = 'empty vector'
            except Exception as e:
                last_err = e
            time.sleep(2)
        raise TimeoutError(f'{self.name} warmup failed: {last_err}')

    def _boot_once(self):
        ngl   = self._gpu_layers()
        where = f'GPU (ngl={ngl})' if ngl != '0' else 'CPU (ngl=0)'
        print(f'[serve] loading {self.name} on port {self.port} [{where}] ...', flush=True)
        self.proc = subprocess.Popen(
            self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            self._wait_ready()
            self._warmup()   # prove real inference works (and warm the model) before "ready"
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

    def embed(self, text, timeout=60):
        """POST /v1/embeddings to llama-server; return list of floats.
        llama-server --embedding exposes the OpenAI-compatible embeddings endpoint."""
        payload = json.dumps({'input': text, 'model': 'nomic-embed'}).encode()
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=timeout)
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

    def generate_grounded(self, system_prompt, user_content,
                          max_tokens=512, temperature=0.1, top_p=0.9):
        """POST /v1/chat/completions with an explicit system + user message pair.

        Uses a lower temperature (0.1) than the debug /generate endpoint because
        grounded RAG answers benefit from determinism: the model must cite exactly
        the source labels it was given, and temperature=0.1 keeps it on-script
        while still allowing natural phrasing variation.

        Returns (reply_text, n_tokens) where n_tokens is the completion token
        count reported by the backend (or 0 if the field is absent).
        """
        payload = json.dumps({
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user',   'content': user_content},
            ],
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
            n_tokens = obj.get('usage', {}).get('completion_tokens', 0) or 0
            return text, n_tokens
        finally:
            conn.close()

    def generate_grounded_streaming(self, system_prompt, user_content,
                                    token_callback,
                                    max_tokens=512, temperature=0.1, top_p=0.9):
        """POST /v1/chat/completions with stream=true; call token_callback(token)
        for each token delta as it arrives via SSE.

        Returns (full_text, n_tokens) on completion.  n_tokens is the count of
        tokens reported by the SSE 'usage' field (or approximated by counting
        non-empty deltas if the field is absent).

        The SSE stream format from llama-server is:
          data: {"choices":[{"delta":{"content":"..."}}], ...}
          data: [DONE]
        """
        payload = json.dumps({
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user',   'content': user_content},
            ],
            'max_tokens':   max_tokens,
            'temperature':  temperature,
            'top_p':        top_p,
            'stream':       True,
            'stream_options': {'include_usage': True},
        }).encode()
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=300)
        try:
            conn.request('POST', '/v1/chat/completions', payload,
                         headers={'Content-Type': 'application/json'})
            resp = conn.getresponse()
            if resp.status >= 400:
                err = resp.read().decode('utf-8', 'replace')
                raise RuntimeError(f'gen HTTP {resp.status}: {err}')

            full_text = ''
            n_tokens  = 0
            token_count = 0
            stream_done = False
            # Read SSE line by line; llama-server sends 'data: ...\n\n' per chunk.
            buf = b''
            while not stream_done:
                chunk = resp.read(256)   # read in small blocks (not byte-by-byte)
                if not chunk:
                    break
                buf += chunk
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    line = line.rstrip(b'\r').decode('utf-8', 'replace')
                    if not line.startswith('data:'):
                        continue
                    data_str = line[5:].strip()
                    if data_str == '[DONE]':
                        stream_done = True
                        break
                    try:
                        obj = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    # Token delta -- choices may be empty on the final usage chunk
                    choices = obj.get('choices') or []
                    if choices:
                        delta = choices[0].get('delta', {})
                        token = (delta.get('content') or
                                 delta.get('reasoning') or
                                 delta.get('reasoning_content') or '')
                        if token:
                            full_text += token
                            token_count += 1
                            token_callback(token)
                    # Usage field (may appear on final chunk or a separate chunk)
                    usage = obj.get('usage') or {}
                    if usage.get('completion_tokens'):
                        n_tokens = usage['completion_tokens']

            if n_tokens == 0:
                n_tokens = token_count
            return full_text, n_tokens
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
        return True

    def embed(self, text, timeout=None):
        # timeout is accepted for signature parity with ProxyBackend.embed (which
        # uses it for the HTTP call); in-process embedding has no socket to time out.
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
        print(f'[serve] loading gen model in-process from {self.path} [{where}] ...', flush=True)
        InProcGenBackend._llm = Llama(
            model_path=self.path,
            n_ctx=8192,
            n_threads=THREADS,
            n_threads_batch=THREADS,
            n_gpu_layers=ngl,
            verbose=False,
        )
        print(f'[serve] gen model ready (in-process, {where})', flush=True)
        return True

    def generate(self, prompt, max_tokens=256, temperature=0.7, top_p=0.9):
        result = InProcGenBackend._llm.create_chat_completion(
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
        )
        return result['choices'][0]['message'].get('content', '')

    def generate_grounded(self, system_prompt, user_content,
                          max_tokens=512, temperature=0.1, top_p=0.9):
        """In-process equivalent of ProxyBackend.generate_grounded."""
        result = InProcGenBackend._llm.create_chat_completion(
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user',   'content': user_content},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
        )
        text     = result['choices'][0]['message'].get('content', '')
        n_tokens = result.get('usage', {}).get('completion_tokens', 0) or 0
        return text, n_tokens

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

        # Embedder: CPU-only, --embedding flag.
        # ctx/batch sized to 2048: an embedding input must fit in ONE physical batch,
        # and a ~512-token chunk + the 'search_document: ' prefix can exceed 512 (a
        # 514-token chunk got rejected with HTTP 500 at batch 512). nomic-embed v1.5
        # supports 2048, giving ~4x headroom over our chunk target.
        embed_cmd = [
            llama_srv,
            '--model',         embed_gguf,
            '--port',          str(EMBED_PORT),
            '--host',          '127.0.0.1',
            '--ctx-size',      '2048',
            '--batch-size',    '2048',
            '--ubatch-size',   '2048',
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
            '--ctx-size',      '8192',
            '--n-gpu-layers',  phi_ngl,
            '--threads',       str(THREADS),
            '--no-webui',
        ]
        _gen_backend = ProxyBackend('phi', phi_cmd, PHI_PORT, 'llama')


def boot_backends():
    """Boot both backends SEQUENTIALLY (embedder first, then Phi).

    They must NOT be constructed concurrently.  On the Linux in-process path
    (llama-cpp-python) two Llama() constructors racing in the same process
    corrupt llama.cpp's global backend/CUDA init and the process dies with no
    Python traceback -- we verified that the same load succeeds fine when run
    alone, and that both models coexist happily once loaded; only simultaneous
    construction crashes.  The Windows path (subprocess llama-server) doesn't
    share that in-process state, but the weights are already on disk by now
    (ensure_weights() ran in main()), so serial boot costs only a couple
    seconds of construction time and is correct on both platforms.
    """
    errors = []

    def _boot(b):
        try:
            ok = b.boot()
            if not ok:
                errors.append(f'{b.name if hasattr(b,"name") else type(b).__name__} failed to boot')
        except Exception as e:
            errors.append(str(e))

    _boot(_embed_backend)
    _boot(_gen_backend)

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
# Orphan cleanup -- a crashed or force-killed serve.py must not leave its
# llama-server children holding the model ports and GPU VRAM. Two guards:
#   1. a Windows Job Object so children die with us even on a hard kill, and
#   2. a startup reap that frees the model ports before we boot.
# Together they make every launch start from a clean slate (last launch wins).
##############################################################################

_kill_on_exit_job = None   # keep the Windows job handle alive for our lifetime


def _setup_kill_on_exit_job():
    """Windows: place this process in a Job Object configured to kill all member
    processes when the job closes (when serve.py exits -- including a crash or
    taskkill). Child llama-server processes inherit the job, so they can never be
    orphaned holding a model port + VRAM. Best-effort; no-op on other OSes or on
    failure (the startup reap + SIGTERM teardown still apply)."""
    global _kill_on_exit_job
    if os.name != 'nt':
        return
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)

        class BASIC(ctypes.Structure):
            _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64),
                        ('PerJobUserTimeLimit', ctypes.c_int64),
                        ('LimitFlags', wintypes.DWORD),
                        ('MinimumWorkingSetSize', ctypes.c_size_t),
                        ('MaximumWorkingSetSize', ctypes.c_size_t),
                        ('ActiveProcessLimit', wintypes.DWORD),
                        ('Affinity', ctypes.c_size_t),
                        ('PriorityClass', wintypes.DWORD),
                        ('SchedulingClass', wintypes.DWORD)]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in
                        ('ReadOperationCount', 'WriteOperationCount',
                         'OtherOperationCount', 'ReadTransferCount',
                         'WriteTransferCount', 'OtherTransferCount')]

        class EXTENDED(ctypes.Structure):
            _fields_ = [('BasicLimitInformation', BASIC),
                        ('IoInfo', IO_COUNTERS),
                        ('ProcessMemoryLimit', ctypes.c_size_t),
                        ('JobMemoryLimit', ctypes.c_size_t),
                        ('PeakProcessMemoryUsed', ctypes.c_size_t),
                        ('PeakJobMemoryUsed', ctypes.c_size_t)]

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JobObjectExtendedLimitInformation = 9
        # Declare argtypes/restype so ctypes passes HANDLEs as pointer-sized values.
        # Without this, handle args default to 32-bit int and GetCurrentProcess()'s
        # -1 pseudo-handle overflows ("int too long to convert").
        HANDLE, BOOL, DWORD, LPVOID = (wintypes.HANDLE, wintypes.BOOL,
                                       wintypes.DWORD, wintypes.LPVOID)
        k32.CreateJobObjectW.restype = HANDLE
        k32.CreateJobObjectW.argtypes = [LPVOID, wintypes.LPCWSTR]
        k32.GetCurrentProcess.restype = HANDLE
        k32.GetCurrentProcess.argtypes = []
        k32.SetInformationJobObject.restype = BOOL
        k32.SetInformationJobObject.argtypes = [HANDLE, ctypes.c_int,
                                                ctypes.c_void_p, DWORD]
        k32.AssignProcessToJobObject.restype = BOOL
        k32.AssignProcessToJobObject.argtypes = [HANDLE, HANDLE]
        k32.CloseHandle.argtypes = [HANDLE]

        job = k32.CreateJobObjectW(None, None)
        if not job:
            return
        info = EXTENDED()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                                           ctypes.byref(info), ctypes.sizeof(info)):
            k32.CloseHandle(job)
            return
        if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
            k32.CloseHandle(job)
            return
        _kill_on_exit_job = job   # closes (and kills children) when the process ends
        print('[serve] kill-on-exit guard active (llama-server children die with serve.py)',
              flush=True)
    except Exception as e:
        print(f'[serve] kill-on-exit guard unavailable ({e}); '
              f'relying on startup reap + shutdown teardown', flush=True)


def _pids_listening_on(port):
    """PIDs holding a LISTEN socket on `port` (best-effort, stdlib netstat/lsof)."""
    pids = set()
    try:
        if os.name == 'nt':
            out = subprocess.run(['netstat', '-ano', '-p', 'tcp'],
                                 capture_output=True, text=True, timeout=10).stdout
            needle = f':{port}'
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[3].upper() == 'LISTENING' \
                        and parts[1].endswith(needle):
                    try:
                        pids.add(int(parts[4]))
                    except ValueError:
                        pass
        else:
            out = subprocess.run(['lsof', '-ti', f'tcp:{port}', '-sTCP:LISTEN'],
                                 capture_output=True, text=True, timeout=10).stdout
            for line in out.splitlines():
                if line.strip().isdigit():
                    pids.add(int(line.strip()))
    except Exception:
        pass
    return pids


def _kill_pid(pid):
    try:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)],
                           capture_output=True, timeout=10)
        else:
            os.kill(pid, signal.SIGKILL)
    except Exception:
        pass


def _kill_llama_servers_by_name():
    """Kill every llama-server process. We are the only llama.cpp process on this box,
    so at startup (before we boot ours) any llama-server is an orphan from a prior
    crashed run. Such an orphan holds GPU VRAM even if it is no longer bound to our
    ports, which skews the free-VRAM reading. Browser/desktop GPU use is left alone."""
    try:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/F', '/IM', 'llama-server.exe'],
                           capture_output=True, timeout=15)
        else:
            subprocess.run(['pkill', '-9', '-f', 'llama-server'],
                           capture_output=True, timeout=15)
    except Exception:
        pass


def refresh_free_vram():
    """Re-read free VRAM after reaping orphans so the GPU-vs-CPU decision sees the true
    number. The driver reclaims VRAM from killed processes asynchronously, so settle
    briefly before reading. Updates the GPU_FREE_GB global used by gpu_layers_for_phi."""
    global GPU_FREE_GB
    if GPU_TOTAL_GB is None:
        return
    time.sleep(1.5)
    _, free = nvidia_vram_gb()
    if free is not None:
        GPU_FREE_GB = free
        print(f'[serve] free VRAM after reap: {GPU_FREE_GB:.1f} GB '
              f'of {GPU_TOTAL_GB:.1f} GB total', flush=True)


def reap_orphan_backends():
    """Make the GPU and ports ours before booting. We are the only llama.cpp process,
    so kill any leftover llama-server (an orphan from a crashed run still holding VRAM),
    then anything still listening on EMBED_PORT / PHI_PORT, then re-read free VRAM so the
    GPU-vs-CPU decision isn't skewed by what the orphans were holding. Real browser/
    desktop VRAM use is respected. Last launch wins -- what a single-user app wants."""
    _kill_llama_servers_by_name()
    for port in (EMBED_PORT, PHI_PORT):
        for pid in _pids_listening_on(port):
            print(f'[serve] reaping orphan backend on port {port} (pid {pid})', flush=True)
            _kill_pid(pid)
    refresh_free_vram()


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
        print(f'[serve]   gen       Qwen2.5-7B Q4_K_M              in-process  {phi_where}')
    else:
        print(f'[serve]   embedder  nomic-embed-text-v1.5 Q4_K_M  port {EMBED_PORT}  CPU (ngl=0)')
        phi_where = f'GPU (ngl={phi_ngl})' if phi_ngl != '0' else 'CPU (ngl=0)'
        print(f'[serve]   gen       Qwen2.5-7B Q4_K_M              port {PHI_PORT}  {phi_where}')
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
        print('[serve] shutdown: stopping worker thread ...', flush=True)
        _worker_stop.set()
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

        log_event('http_request', ip, 'GET', path, request_id=request_id,
                  request_body=query or None)

        if path == '/health':
            self._handle_health(request_id)
        elif path == '/shutdown':
            self._handle_shutdown('GET', query, request_id)
        elif path == '/tree':
            self._handle_tree(request_id)
        elif path == '/doc':
            self._handle_doc(query, request_id)
        elif path == '/request':
            self._handle_request_status(query, request_id)
        elif path == '/history':
            self._handle_history(query, request_id)
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
                  request_body=body.decode('utf-8', 'replace') if body else None)

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
        elif path == '/enqueue':
            self._handle_enqueue(body, request_id)
        elif path == '/clear':
            self._handle_clear(body, request_id)
        elif path == '/delete':
            self._handle_delete(body, request_id)
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
                  request_body=text)
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
                  request_body=prompt)
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
                  response_body=text or '')
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

        # ── Derive file extension ────────────────────────────────────────────
        ext = os.path.splitext(rel_path)[1].lower()

        # ── Insert documents row (BLOB stored, status='vectorizing') ─────────
        # insert_document first removes any existing doc at this path (including
        # all its chunks/vectors) then inserts a fresh row -- clean re-ingest.
        try:
            with _db_lock:
                doc_id = _db_mod.insert_document(_db_conn, rel_path, file_bytes, ext)
        except Exception as e:
            self._json_response({'error': f'DB insert failed: {e}'}, 500,
                                request_id=request_id)
            return

        log_event('ingest_queued', ip, 'POST', '/ingest', request_id=request_id,
                  doc_id=doc_id, rel_path=rel_path)

        # ── Kick off background ingestion ────────────────────────────────────
        t = threading.Thread(
            target=_ingest_background,
            args=(doc_id, rel_path, request_id),
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
                    "SELECT path, status, n_chunks, byte_size, ext "
                    "FROM documents ORDER BY path"
                ).fetchall()
        except Exception as e:
            self._json_response({'error': str(e)}, 500, request_id=request_id)
            return

        files = []
        for path, status, n_chunks, byte_size, ext in rows:
            files.append({
                'path':        path,
                'status':      status,
                'chunk_count': n_chunks or 0,
                'byte_size':   byte_size,
                'ext':         ext,
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
                      'question': question,
                      'scope':    scope,
                      'k':        k,
                  }, ensure_ascii=False))

        t0 = time.time()

        # ── Step 1: embed the question (search_query: prefix for nomic v1.5) ──
        # nomic-embed-text v1.5 uses task prefixes for asymmetric retrieval.
        # Documents were stored with 'search_document: '; queries MUST use
        # 'search_query: ' so they project into the same embedding space.
        prefixed_question = 'search_query: ' + question
        log_event('embed_request', ip, 'POST', '/retrieve',
                  request_id=request_id,
                  request_body=prefixed_question)
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

    # ── P6 endpoints ──────────────────────────────────────────────────────────

    def _handle_enqueue(self, body: bytes, request_id: str):
        """POST /enqueue {"kind":"chat","content":"<question>","session_id":"<opt>"}

        Creates a requests row (status=pending) and a user messages row, mints
        a session_id if none was provided, and returns {request_id, session_id}
        immediately -- the actual answer is produced asynchronously by the
        chat worker thread.
        """
        ip = self.client_address[0]
        try:
            req = json.loads(body)
        except (json.JSONDecodeError, AttributeError):
            self._json_response({'error': 'Invalid JSON body'}, 400,
                                request_id=request_id)
            return

        kind    = req.get('kind', 'chat')
        content = req.get('content', '').strip()
        sid     = req.get('session_id') or str(uuid.uuid4())

        if kind != 'chat':
            self._json_response({'error': '"kind" must be "chat"'}, 400,
                                request_id=request_id)
            return
        if not content:
            self._json_response({'error': '"content" must be a non-empty string'},
                                400, request_id=request_id)
            return

        if _db_conn is None:
            self._json_response({'error': 'DB not initialised'}, 503,
                                request_id=request_id)
            return

        # Persist the user turn to the transcript and queue the request.
        payload_json = json.dumps({'question': content})
        try:
            with _db_lock:
                # User message first so history is correct when the worker runs.
                _db_mod.insert_message(
                    _db_conn,
                    session_id=sid,
                    role='user',
                    content=content,
                    request_id=request_id,
                    status='done',
                )
                _db_mod.insert_request(
                    _db_conn,
                    kind=kind,
                    content=payload_json,
                    request_id=request_id,
                    session_id=sid,
                )
        except Exception as e:
            self._json_response({'error': f'DB insert failed: {e}'}, 500,
                                request_id=request_id)
            return

        log_event('enqueue', ip, 'POST', '/enqueue',
                  request_id=request_id, chat_session_id=sid,
                  request_body=content)

        self._json_response(
            {'request_id': request_id, 'session_id': sid},
            200,
            request_id=request_id,
        )

    def _handle_request_status(self, query: str, request_id: str):
        """GET /request?id=<rid> -- return {status, error} for a request UUID."""
        ip = self.client_address[0]

        params = parse_qs(query or '')
        rid_list = params.get('id', [])
        if not rid_list:
            self._json_response({'error': 'id parameter required'}, 400,
                                request_id=request_id)
            return
        rid = rid_list[0]

        if _db_conn is None:
            self._json_response({'error': 'DB not initialised'}, 503,
                                request_id=request_id)
            return

        try:
            with _db_lock:
                row = _db_mod.get_request(_db_conn, rid)
        except Exception as e:
            self._json_response({'error': str(e)}, 500, request_id=request_id)
            return

        if row is None:
            self._json_response({'error': f'request not found: {rid!r}'}, 404,
                                request_id=request_id)
            return

        self._json_response(
            {'status': row['status'], 'error': row['error']},
            200,
            request_id=request_id,
        )

    def _handle_history(self, query: str, request_id: str):
        """GET /history?session_id=<sid> -- return ordered conversation messages."""
        ip = self.client_address[0]

        params = parse_qs(query or '')
        sid_list = params.get('session_id', [])
        if not sid_list:
            self._json_response({'error': 'session_id parameter required'}, 400,
                                request_id=request_id)
            return
        sid = sid_list[0]

        if _db_conn is None:
            self._json_response({'error': 'DB not initialised'}, 503,
                                request_id=request_id)
            return

        try:
            with _db_lock:
                msgs = _db_mod.get_messages(_db_conn, sid)
        except Exception as e:
            self._json_response({'error': str(e)}, 500, request_id=request_id)
            return

        self._json_response({'messages': msgs}, 200, request_id=request_id)

    def _handle_clear(self, body: bytes, request_id: str):
        """POST /clear {"session_id": "..."}

        Deletes all messages for the given session so that the next question
        starts from a blank context window.  Used by the CLI /clear command
        and (later) by the web UI clear button.

        Returns {ok: true, deleted: <n>} on success.
        """
        ip = self.client_address[0]
        try:
            req = json.loads(body) if body else {}
        except (json.JSONDecodeError, AttributeError):
            self._json_response({'error': 'Invalid JSON body'}, 400,
                                request_id=request_id)
            return

        sid = req.get('session_id', '').strip()
        if not sid:
            self._json_response({'error': '"session_id" required'}, 400,
                                request_id=request_id)
            return

        if _db_conn is None:
            self._json_response({'error': 'DB not initialised'}, 503,
                                request_id=request_id)
            return

        try:
            with _db_lock:
                deleted = _db_mod.clear_session(_db_conn, sid)
        except Exception as e:
            self._json_response({'error': str(e)}, 500, request_id=request_id)
            return

        log_event('clear_session', ip, 'POST', '/clear',
                  request_id=request_id, chat_session_id=sid,
                  deleted=deleted)
        self._json_response({'ok': True, 'deleted': deleted}, 200,
                            request_id=request_id)

    def _handle_delete(self, body: bytes, request_id: str):
        """POST /delete {"path": "..."} -- remove a document and all its data.

        Calls db.delete_document which explicitly removes chunk_vecs,
        chunk_vec_meta, chunks, and the documents row in that order (FK cascade
        is not relied upon -- chunk_vecs is a vec0 virtual table and connect()
        does not enable PRAGMA foreign_keys).

        Returns {"ok": true, "deleted": <bool>, "path": <path>}.
        Returns HTTP 404 with deleted=false when the path is not found.
        """
        ip = self.client_address[0]
        try:
            req = json.loads(body) if body else {}
        except (json.JSONDecodeError, AttributeError):
            self._json_response({'error': 'Invalid JSON body'}, 400,
                                request_id=request_id)
            return

        path = (req.get('path') or '').strip()
        if not path:
            self._json_response({'error': '"path" required'}, 400,
                                request_id=request_id)
            return

        if _db_conn is None:
            self._json_response({'error': 'DB not initialised'}, 503,
                                request_id=request_id)
            return

        try:
            with _db_lock:
                deleted = _db_mod.delete_document(_db_conn, path)
        except Exception as e:
            self._json_response({'error': str(e)}, 500, request_id=request_id)
            return

        log_event('delete_document', ip, 'POST', '/delete',
                  request_id=request_id, doc_path=path, deleted=deleted)

        status_code = 200 if deleted else 404
        self._json_response({'ok': True, 'deleted': deleted, 'path': path},
                            status_code, request_id=request_id)

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
        # Log the FULL response body for every API response -- accurate, untruncated
        # records of everything sent to the browser are required.
        log_event('http_response', ip, self.command, self.path.partition('?')[0],
                  request_id=request_id, status=status,
                  response_body=data.decode('utf-8', 'replace'))


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
# CLI REPL -- P7
#
# A thin HTTP client of the same running server (http://127.0.0.1:<port>).
# The terminal and the browser share one transcript and go through the same
# two gates (embed + gen), so no second inference path exists.
#
# Commands
# --------
#   /clear   -- delete this session's transcript (POST /clear) and start fresh
#   /help    -- print the command list
#   /quit    -- exit the REPL (and shut the server down)
#   /exit    -- alias for /quit
#   <text>   -- ask a question; polls until done; prints the answer inline
#
# There is no /model switch -- slm-rag serves a single model.
#
# Generation on this hardware can take up to ~50 s; the REPL polls patiently
# with a "thinking..." heartbeat every 15 s and a generous timeout (10 min).
##############################################################################

CLI_HELP = """\
[cli] Commands:
  /help   -- show this message
  /clear  -- erase the current session transcript (web UI sees it too)
  /quit   -- exit (also /exit)
  <text>  -- ask a question; citations are printed inline with the answer

There is no /model switch -- slm-rag serves one model."""

CLI_POLL_INTERVAL_S = 3.0
CLI_ANSWER_TIMEOUT_S = 10 * 60   # 10 min; generation can take ~50 s on GPU
CLI_HEARTBEAT_EVERY  = 15.0


def _cli_http_get(base_url, path_qs, timeout=10):
    """GET base_url+path_qs; return (status_code, parsed_json_or_None)."""
    try:
        with urlopen(base_url + path_qs, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except Exception as e:
        return None, {'error': str(e)}


def _cli_http_post(base_url, path, payload, timeout=30):
    """POST JSON payload to base_url+path; return (status_code, parsed_json_or_None)."""
    from urllib.error import HTTPError
    data = json.dumps(payload).encode()
    req  = Request(
        base_url + path,
        data=data,
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
    except Exception as e:
        return None, {'error': str(e)}


def cli_repl(base_url):
    """Interactive terminal chat loop.

    Runs on the main thread after the web server thread is up.  Exits when the
    user types /quit or /exit (or sends EOF).  Triggers a clean server shutdown
    when it exits so Ctrl-C / /quit both clean up properly.
    """
    session_id = str(uuid.uuid4())

    print('[cli] Terminal chat ready.  Type /help for commands.', flush=True)
    print(f'[cli] Session: {session_id[:8]}...', flush=True)
    print('[cli] (Web UI available at ' + base_url + ')', flush=True)

    while True:
        # Read one line from stdin.  EOF (piped input exhausted, or Ctrl-D)
        # is treated the same as /quit.
        try:
            line = sys.stdin.readline()
        except (EOFError, KeyboardInterrupt):
            line = ''

        if not line:
            # EOF -- exit cleanly
            print('\n[cli] EOF -- exiting.', flush=True)
            break

        text = line.rstrip('\n').rstrip('\r')

        # ── Built-in commands ─────────────────────────────────────────────────
        if text.strip() == '/help':
            print(CLI_HELP, flush=True)
            continue

        if text.strip() in ('/quit', '/exit'):
            print('[cli] Goodbye.', flush=True)
            break

        if text.strip() == '/clear':
            code, resp = _cli_http_post(
                base_url, '/clear', {'session_id': session_id}
            )
            if code == 200 and resp.get('ok'):
                session_id = str(uuid.uuid4())
                print(f'[cli] Transcript cleared.  New session: {session_id[:8]}...',
                      flush=True)
            else:
                print(f'[cli] /clear failed: {resp}', flush=True)
            continue

        # ── Skip blank lines ──────────────────────────────────────────────────
        if not text.strip():
            continue

        # ── Ask a question ────────────────────────────────────────────────────
        question = text.strip()

        # 1. Enqueue
        enq_code, enq_resp = _cli_http_post(
            base_url, '/enqueue',
            {'kind': 'chat', 'content': question, 'session_id': session_id},
        )
        if enq_code != 200:
            print(f'[cli] Error queuing question: {enq_resp}', flush=True)
            continue

        rid        = enq_resp.get('request_id')
        session_id = enq_resp.get('session_id', session_id)   # server may have minted one

        if not rid:
            print('[cli] Error: server did not return a request_id', flush=True)
            continue

        # 2. Poll until done
        print('[cli] thinking...', flush=True)
        deadline  = time.time() + CLI_ANSWER_TIMEOUT_S
        last_hb   = time.time()
        done_resp = None

        while time.time() < deadline:
            time.sleep(CLI_POLL_INTERVAL_S)
            poll_code, poll_resp = _cli_http_get(
                base_url, f'/request?id={rid}'
            )
            if poll_code == 200:
                status = poll_resp.get('status', '')
                if status in ('done', 'error'):
                    done_resp = poll_resp
                    break
            # Heartbeat so the user knows the CLI is still alive
            now = time.time()
            if now - last_hb >= CLI_HEARTBEAT_EVERY:
                remaining = int(deadline - now)
                print(f'[cli] still thinking... ({remaining}s left)', flush=True)
                last_hb = now

        if done_resp is None:
            print(f'[cli] Timed out waiting for answer (>{CLI_ANSWER_TIMEOUT_S}s)',
                  flush=True)
            continue

        if done_resp.get('status') == 'error':
            print(f'[cli] Generation error: {done_resp.get("error")}', flush=True)
            continue

        # 3. Fetch history and print the latest assistant answer
        h_code, h_resp = _cli_http_get(
            base_url, f'/history?session_id={session_id}'
        )
        if h_code != 200:
            print(f'[cli] Could not retrieve history: {h_resp}', flush=True)
            continue

        messages   = h_resp.get('messages', [])
        asst_msgs  = [m for m in messages
                      if m.get('role') == 'assistant' and m.get('status') == 'done']
        if not asst_msgs:
            print('[cli] No answer available yet.', flush=True)
            continue

        # Use the clean `answer` field (no prompt prefix); fall back to content.
        last_asst = asst_msgs[-1]
        answer = (last_asst.get('answer') or last_asst.get('content', '')).strip()
        # Print the answer with a blank line above and below for readability.
        # Numbered citations [1][2] are embedded in the answer text by the prompt.
        print(f'\n{answer}\n', flush=True)

        # Expand numbered citations into [Source: ...] lines so the terminal
        # output is human-readable and backward-compatible with tooling that
        # expects [Source: <file>, chunk <n>] markers in the CLI transcript.
        refs_raw = last_asst.get('references')
        if refs_raw:
            try:
                refs = json.loads(refs_raw)
                if refs:
                    print('[cli] Sources:', flush=True)
                    for ref in refs:
                        print(
                            f'  [{ref["n"]}] [Source: {ref["path"]}, chunk {ref["chunk_index"]}]',
                            flush=True,
                        )
                    print('', flush=True)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass


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
  Gen       Qwen2.5-7B Q4_K_M              port 52851  GPU if VRAM fits, else CPU
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
    _db_conn = _db_mod.init_db(db_path)
    print('[serve] rag.db ready', flush=True)

    # ── ensure llama-server binary is present (Windows: download if missing) ─
    ensure_llama_server()

    # ── download weights (embedder first, then Phi) ───────────────────────────
    ensure_weights()

    # ── orphan guards: kill leftover backends from a prior crashed run, and
    #    arrange for our own child llama-servers to die with us (Windows job) ───
    _setup_kill_on_exit_job()
    reap_orphan_backends()

    # ── build and boot both backends ──────────────────────────────────────────
    build_backends()
    boot_backends()

    # ── start the chat worker thread (P6) ─────────────────────────────────────
    # The worker drains pending 'chat' requests one at a time, serializing
    # generation through the gen gate without blocking the HTTP server.
    _worker_stop.clear()
    worker_thread = threading.Thread(target=_chat_worker, daemon=True,
                                     name='chat-worker')
    worker_thread.start()
    print('[serve] chat worker started', flush=True)

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
              port=port, version='p6-answer', db_path=db_path)

    if args['mode'] == 'cli':
        cli_repl(f'http://{HOST}:{port}')
        begin_shutdown(server)

    # Block until server_thread finishes (i.e. server.shutdown() was called).
    server_thread.join()
    print('[serve] server stopped.', flush=True)


if __name__ == '__main__':
    main()
