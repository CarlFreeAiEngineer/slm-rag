# slm-rag -- a portable, self-improving RAG over your own files

Drop files into a tree on the left, ask questions on the right. A **small language
model** (Phi-4-mini) answers using **only what it retrieves from your documents**, and
cites where each answer came from. When an answer is wrong, you correct it -- those
corrections become training data, and the model is fine-tuned to do better next time.

The whole thing is **self-contained and portable**: a bundled `uv`, a bundled
`llama.cpp`, and a single SQLite file hold the app, the models, the documents, and the
vector index. No system installs, no database server, no cloud at inference time.

It is a sibling of [`merv`](../merv) and reuses its proven pattern -- one `serve.py`
running a static UI plus local `llama-server` backends, with weights auto-downloaded
from Hugging Face on first run.

---

## The idea

Start with an **uncustomized** Phi-4-mini and good retrieval. That alone answers a lot
of questions well. Where it falls short -- ignores the context, rambles, fails to say
"I don't know," or formats poorly -- you flag the answer and write the version you
wanted. Those `(question, retrieved-context, good-answer)` examples accumulate into a
training set, and you periodically fine-tune Phi on them (on Colab, with the same
Hugging Face and Colab accounts used for `merv`). The fine-tuned model is uploaded to
Hugging Face and `serve.py` picks it up on the next restart. **Retrieval stays the
same; the model gets better at using it.**

---

## Two panels

| Left -- file tree | Right -- chat |
|-------------------|---------------|
| Your ingested documents, in folders. **Drag a file in** to add it. Each file shows its status: *vectorizing* -> *ready*. Click a file to preview it. | Ask questions; answers stream back grounded in your files, each with **citations** (which file + chunk it used, click to jump to the source). A *"this is wrong -> fix it"* control on any answer captures a training example. |

---

## How it works

### Ingestion (drop -> searchable)
1. **Drop / upload** a file into the tree. Supported in v1: **`.txt`, `.md`, `.pdf`**
   (PDF text via `pypdf`). More types later.
2. **Extract** plain text, **chunk** it (~512 tokens with ~64-token overlap), and
   **embed** each chunk with **nomic-embed-text v1.5** (768-dim) running as a small
   GGUF on the bundled `llama.cpp`.
3. **Store** the document, its chunks, and their vectors in **one SQLite file** using
   the [`sqlite-vec`](https://github.com/asg017/sqlite-vec) extension (a `vec0` virtual
   table). Ingestion and retrieval live in the same file -- nothing else to run.

### Retrieval + answer
1. The question is embedded with the same model.
2. **`sqlite-vec` k-NN** returns the top matching chunks (brute-force, exact -- plenty
   fast for a personal corpus), optionally scoped to a folder/file.
3. Those chunks are stuffed into a grounded prompt for **Phi-4-mini** (served by
   `llama.cpp`), which answers using only the supplied context and **cites its
   sources**, or says it doesn't know when the answer isn't there.

### Serving: two gates (vector + language)

merv funnels all model access through **one** gate (a single worker), because it swaps
one model in and out of a single slot. slm-rag keeps **both** models resident, so it
uses **two independent gates**, one per model -- the same "single-file the model access"
idea, doubled:

- **Embed gate** -- serializes the **vector model** (CPU-pinned). Two callers share it:
  **ingestion** (embedding a dropped file's chunks) and **retrieval** (embedding the
  incoming question). Ingestion embeds in **small batches and yields the gate**, so an
  interactive query-embed slips in *ahead* of a long ingest instead of waiting behind
  the whole file.
- **Gen gate** -- serializes the **language model** (Phi, on the GPU when present). One
  caller: answering. Two questions never generate at once.

A chat touches both gates **sequentially, never nested**: take the embed gate -> embed
the question -> release -> `sqlite-vec` k-NN (no gate) -> take the gen gate -> generate
-> release. So while one answer is still generating (gen gate held), a freshly dropped
file -- or the next question's query-embed -- can use the embed gate **at the same
time**. That overlap is the whole point of two gates instead of one.

**CPU/GPU split.** The vector model runs **CPU-only** (`--n-gpu-layers 0`): it is ~140 MB
and embedding is a single forward pass (no autoregressive decode), so CPU is plenty fast,
and it leaves **all** VRAM to Phi (weights + KV cache). The language model takes the GPU
when it fits and falls back to CPU otherwise (merv's per-model rule, applied to the one
model that benefits from it).

### Improvement loop (same shape as merv)
1. Flag a bad answer and supply the correct one. The app saves
   `(question, retrieved context, corrected answer)` to **SQLite** (exported to the
   `training/rag_finetune.csv` file when you fine-tune).
2. When you have enough, run the Colab notebook: Unsloth LoRA fine-tune of Phi-4-mini
   on those examples -> export Q4_K_M GGUF -> upload to your Hugging Face account.
3. Restart `serve.py`; its staleness check pulls the new weights and serves them.

---

## Why these choices

- **SQLite + sqlite-vec, not Postgres.** Verified working in the bundled Python:
  `sqlite-vec` loads as an extension and does correct k-NN, so the entire vector store
  is one portable file with no server to manage. Postgres + `pgvector` is documented as
  a fallback **only if** the corpus ever outgrows brute-force search (well past typical
  personal-document scale); it is intentionally **not** built.
- **Phi-4-mini for generation.** Small, runs CPU-only via `llama.cpp`, and is the model
  we already know how to fine-tune. "Uncustomized" to start -- the base instruct GGUF.
  It needs **very little background knowledge baked in**: it is instructed to ignore what
  it "knows" and answer **only from the retrieved document context**, so a small model is
  enough -- comprehension and grounding matter far more than world knowledge here.
- **nomic-embed-text v1.5 for embeddings.** Strong retrieval quality, small (~140 MB
  GGUF), and runs through the same `llama.cpp` we already bundle -- no separate Python
  embedding stack.
- **Reuse merv's portable runtime.** Bundled `uv` + `llama.cpp`, `run.bat`/`run.sh`,
  HF auto-download, static UI served by `serve.py`. Two tiny `llama-server` instances
  run side by side: one for Phi (chat), one for the embedder.

---

## Running it

```bat
run.bat            ::  Windows
```
```bash
./run.sh           #   macOS / Linux
```

`run.*` picks the right bundled `uv` binary, creates an isolated venv from the inline
script metadata in `serve.py`, ensures the bundled `llama.cpp` is present, downloads
the Phi and embedder GGUFs from Hugging Face on first run (smallest first, so you can
start as soon as the embedder is ready), then serves the UI. Open
<http://localhost:51548>.

| Port | Process |
|------|---------|
| 51548 | `serve.py` -- web UI + API + SQLite/sqlite-vec |
| 52851 | `llama-server` -- Phi-4-mini (generation) |
| 52852 | `llama-server` -- nomic-embed-text (embeddings) |

**The llama.cpp backend differs per OS** (same approach as merv). Windows has no
clean package path, so the one binary we bundle is the prebuilt `llama-server.exe`
(CUDA build) under `bin/llama.cpp/`; **macOS** uses the system `llama-server` from
`brew install llama.cpp` (Metal GPU); **Linux** ships no server binary -- the
`llama-cpp-python` wheel (a Linux-only inline dep) has llama.cpp compiled in and runs
in-process. Unlike merv (one model at a time), slm-rag keeps **two** models loaded at
once (generation + embeddings): on Windows/macOS that's two `llama-server` instances
on the two ports above; on Linux both load in-process via `llama-cpp-python`.

### Command-line chat (same pattern as merv)

By default `serve.py` runs **web-only**. Pass `--cli` to also drop into a **terminal
chat** alongside the web UI -- ask a question, get the grounded answer with its
**citations printed inline**. The CLI is a thin HTTP client of the *same* server, so the
terminal and the browser go through the **same two gates** (see *Serving*): two questions
never generate at once, the shared transcript stays consistent across both, and there is
no second inference path to drift out of sync. Pass the flag through the launcher:
`run.bat --cli` / `./run.sh --cli`.

| Flag | Effect |
|------|--------|
| `--web` | run the web server only (**default**) |
| `--cli` | run the web server **plus** the terminal chat |
| `--port <n>` | listen port (default 51548; this flag is the only override) |
| `--check` | print the detected backend plan and exit (no downloads, no models start) |
| `--help` | print command-line help and exit |

In the terminal: type a question to ask it; `/clear` erases the shared transcript (for
the web UI too), `/help` lists commands, `/quit` exits. There is no `/model` switch --
slm-rag serves a single model.

**Configuration is command-line only.** Every setting has a sensible default baked in
and a `--flag` to override it -- there are **no environment variables** (this is a
deliberate departure from merv, which reads `MERV_PORT`, `MERV_HOST`, `MERV_THREADS`,
etc.). If a new knob is needed, add a flag with a default, not an env var.

### Everything is testable from the command line

We **maximize what can be exercised from the shell** -- the `--cli` chat plus plain
`curl` against the HTTP API -- so the browser holds **as few surprises as possible**.
The browser is just another HTTP client of `serve.py`; it has **no private path**.
Every user-visible action -- ingest a file, list the tree, embed + retrieve, ask a
question and read the cited answer, flag a correction, clear history, shut down -- is a
documented endpoint you can hit with `curl` and assert on in a script. So the rule is:
**no behavior reachable only through the UI.** If the browser can do it, `curl` and the
CLI can do it too, and that path is what we test. This keeps the API the real contract,
makes failures reproducible without a browser, and means anything green from the shell
behaves the same when clicked.

### Logging & observability

Two tiers, on purpose:

- **SQLite is the system of record.** As much state as possible lives in the one
  portable `rag.db` -- documents, chunks and their vectors, the chat transcript, the
  request queue, the resident-/loading-model state, and the training corrections. The
  browser and the `--cli` repaint themselves purely by **reading** SQLite, so there is
  no per-client state to keep in sync and the whole app's state travels in one file.

- **`./logs/` is the append-only audit trail (same shape as merv).** One JSON object per
  line (JSONL), in hourly-rotated UTC files (`YYYY-MM-DD-HHZ.log`), written under a lock.
  We record the **request and the response as separate lines**, for **both** the HTTP
  API and each inference call. Every line carries **two correlation IDs**:

  - a **request id** -- lives from the moment an HTTP request arrives until its response
    is sent, and binds that one exchange's `http_request`/`http_response` to the
    inference calls it triggered; and
  - a **chat session id** -- groups the several messages of one conversation, and lives
    until the user hits **Clear**.

  ```
  http_request    <- the browser / CLI / curl call arrives     [request id starts]
    embed_request    /  embed_response    -- query embedded on the vector model
    gen_request      /  gen_response      -- answer generated on the language model
  http_response   -> what we sent back                          [request id ends]
  ```

  Because one chat **fans out into two inference calls** (embed the query, then
  generate), the **request id** is what binds that exchange's HTTP pair and *both*
  inference pairs together -- grep it and you see the retrieved context, the exact prompt
  sent to Phi, the reply, token counts / latency, and any error. Grep the **chat session
  id** instead and you get the whole conversation, every exchange in order, up to the
  last Clear. Both ids are also stored on the SQLite rows they belong to (requests carry
  the request id; transcript messages carry the chat session id), so the structured
  record and the raw `./logs/` blow-by-blow **cross-reference each other by the same
  ids**. Queryable truth in SQLite; full request/response bodies in `./logs/`.

### Shutdown (same strategy as merv)

`serve.py` installs `SIGINT`/`SIGTERM` handlers that **stop both `llama-server`
subprocesses** (terminate, then kill on timeout) before exiting, so Ctrl-C never leaves
orphaned model processes holding VRAM/RAM. For a clean remote stop it also exposes a
**localhost-only** `/shutdown` endpoint: `POST /shutdown`, or `GET /shutdown?<UTC
timestamp>` where the timestamp must be within 5 minutes (a guard so a stray or cached
GET can't kill the server). Shutdown is **idempotent** -- the first request wins, drains
the HTTP server, and frees the backends.

---

## Project layout (planned)

```
slm-rag/
  serve.py              # HTTP server, RAG pipeline, SQLite/sqlite-vec, llama backends
  index.html            # file-tree + chat UI (static, served by serve.py)
  run.bat / run.sh      # portable launchers (bundled uv)
  bin/                  # bundled uv.* and llama.cpp (shared pattern with merv)
  ragdocs/              # ingested source files (the file tree) -- your corpus  (git-ignored)
  rag.db                # SQLite system of record: docs, chunks, vec0 embeddings,
                        #   chat transcript, request queue, model state, corrections  (git-ignored)
  logs/                 # hourly JSONL audit trail: http + inference req/resp, by session id  (git-ignored)
  training/
    rag_finetune.csv    # corrections exported from rag.db for the Colab fine-tune
    finetune_phi_rag.ipynb   # Colab: Unsloth LoRA -> GGUF -> Hugging Face
  model/                # model manifests + auto-downloaded GGUFs (git-ignored)
```

---

## Status

Design stage -- this README is the spec. Next: scaffold `serve.py` (ingestion +
retrieval + chat, plus the `--cli` terminal chat and the signal/`/shutdown` teardown,
both reusing merv's pattern), the two-panel `index.html`, the SQLite schema, and the
Colab fine-tune notebook.

## Notes

- **Citations are first-class.** Every answer shows its sources; that is how you spot
  retrieval vs generation failures and decide what to add to the training set.
- **Embeddings are versioned.** If the embedding model changes, stored vectors must be
  rebuilt -- the schema records which embedder produced each vector.
- **Portable by construction.** Like merv, everything needed is bundled or
  auto-fetched; there is no global install step.
