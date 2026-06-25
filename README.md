# slm-rag -- a portable, self-improving RAG over your own files

Drop files into a tree on the left, ask questions on the right. A **small language
model** (Phi-4-mini) answers using **only what it retrieves from your documents**, and
cites where each answer came from. When an answer is wrong, you correct it -- those
corrections become training data, and the model is fine-tuned to do better next time.

The whole thing is **self-contained and portable**: a bundled `uv`, a bundled
`llama.cpp`, and a single SQLite file hold the app, the models, the documents, and the
vector index. No system installs, no database server, no cloud at inference time.

The architecture is deliberately small: one `serve.py` runs a static UI plus local
`llama-server` backends, with weights auto-downloaded from Hugging Face on first run.
One file you can read top to bottom; nothing hidden behind a framework.

---

## The idea

Start with an **uncustomized** Phi-4-mini and good retrieval. That alone answers a lot
of questions well. Where it falls short -- ignores the context, rambles, fails to say
"I don't know," or formats poorly -- you flag the answer and write the version you
wanted. Those `(question, retrieved-context, good-answer)` examples accumulate into a
training set, and you periodically fine-tune Phi on them (on Colab, with your own
Hugging Face and Colab accounts). The fine-tuned model is uploaded to Hugging Face and
`serve.py` picks it up on the next restart. **Retrieval stays the same; the model gets
better at using it.**

---

## Two panels

| Left -- file tree | Right -- chat |
|-------------------|---------------|
| Your ingested documents, in folders. **Drag a file in** to add it (it's stored as a blob in `rag.db`, not a loose file). Each file shows its status: *vectorizing* -> *ready*. Click a file to preview it; **trash it to delete** the document and its chunks. | Ask questions; answers stream back grounded in your files, each with **citations** (which file + chunk it used, click to jump to the source). A *"this is wrong -> fix it"* control on any answer captures a training example. |

---

## How it works

### Ingestion (drop -> searchable)
1. **Drop / upload** a file into the tree. The raw bytes are stored as a **blob in
   `rag.db`** -- there is no `ragdocs/` directory, so status, content, and deletion all
   live in one row (no OS file locking). Supported in v1: **`.txt`, `.md`, `.pdf`**
   (PDF text via `pypdf`). More types later.
2. **Extract** plain text from the stored blob, **chunk** it (~512 tokens with ~64-token
   overlap), and **embed** each chunk with **nomic-embed-text v1.5** (768-dim) running as
   a small GGUF on the bundled `llama.cpp`.
3. **Store** the document blob, its chunks, and their vectors in **one SQLite file** using
   the [`sqlite-vec`](https://github.com/asg017/sqlite-vec) extension (a `vec0` virtual
   table). The whole corpus is one portable file -- nothing else to run, and deleting a
   document is a single cascading delete.

### Retrieval + answer
1. The question is embedded with the same model.
2. **`sqlite-vec` k-NN** returns the top matching chunks (brute-force, exact -- plenty
   fast for a personal corpus), optionally scoped to a folder/file.
3. Those chunks are stuffed into a grounded prompt for **Phi-4-mini** (served by
   `llama.cpp`), which answers using only the supplied context and **cites its
   sources**, or says it doesn't know when the answer isn't there.

### Serving: two gates (vector + language)

Both models stay resident at once, so slm-rag serializes each one **independently** --
**two gates**, one per model. Funneling everything through a single gate would make a
dropped file block chat (and vice versa); two gates let the embedder and the generator
work at the same time:

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
when it fits and falls back to CPU otherwise -- the only model that benefits from the GPU
gets it.

### Improvement loop
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
- **Phi-4-mini for generation** *(MIT)*. Small, runs CPU-only via `llama.cpp`, and is the
  model we already know how to fine-tune. "Uncustomized" to start -- the base instruct
  GGUF. It needs **very little background knowledge baked in**: it is instructed to ignore
  what it "knows" and answer **only from the retrieved document context**, so a small model
  is enough -- comprehension and grounding matter far more than world knowledge here.
- **nomic-embed-text v1.5 for embeddings** *(Apache 2.0)*. Strong retrieval quality, small
  (~140 MB GGUF), and runs through the same `llama.cpp` we already bundle -- no separate
  Python embedding stack.
- **A self-contained portable runtime.** Bundled `uv` + `llama.cpp`, `run.bat`/`run.sh`,
  HF auto-download, static UI served by `serve.py`. Two tiny `llama-server` instances
  run side by side: one for Phi (chat), one for the embedder. No global install step.

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

**The llama.cpp backend differs per OS.** Windows has no clean package path, so the one
binary we bundle is the prebuilt `llama-server.exe` (CUDA build) under `bin/llama.cpp/`;
**macOS** uses the system `llama-server` from `brew install llama.cpp` (Metal GPU);
**Linux** ships no server binary -- the `llama-cpp-python` wheel (a Linux-only inline
dep) has llama.cpp compiled in and runs in-process. slm-rag keeps **two** models loaded
at once (generation + embeddings): on Windows/macOS that's two `llama-server` instances
on the two ports above; on Linux both load in-process via `llama-cpp-python`.

### Command-line chat

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
and a `--flag` to override it -- there are **no environment variables**. If a new knob is
needed, add a flag with a default, not an env var.

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

- **`./logs/` is the append-only audit trail.** One JSON object per
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

### Shutdown

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
  bin/                  # bundled uv.* and llama.cpp
  rag.db                # SQLite system of record: document blobs, chunks, vec0 embeddings,
                        #   chat transcript, request queue, model state, corrections  (git-ignored)
  logs/                 # hourly JSONL audit trail: http + inference req/resp, by session id  (git-ignored)
  training/
    rag_finetune.csv    # corrections exported from rag.db for the Colab fine-tune
    finetune_phi_rag.ipynb   # Colab: Unsloth LoRA -> GGUF -> Hugging Face
  model/                # model manifests + auto-downloaded GGUFs (git-ignored)
```

---

## Status

The build follows the independently testable phases in **[PLAN.md](PLAN.md)** (each
ends with a `curl`/CLI acceptance test). **The core app works end to end today:**

- **Done & green:** P0 lifecycle, P1 SQLite/sqlite-vec store, P2 extraction+chunking,
  P3 two backends + two gates, P4 ingestion, P5 retrieval, P6 grounded cited answers,
  P7 `--cli` chat, P8 two-panel web UI. The fine-tune notebook (P10) is drafted and
  smoke-tested. Launch with `run.bat` / `./run.sh`, then open <http://localhost:51548>.
- **Remaining:** P9 (capture corrections from the *"this is wrong -> fix it"* control
  into SQLite + CSV) and P11 (staleness check that auto-pulls a re-fine-tuned model).
- **Training is gated on real use.** There is no training data until you ingest your own
  documents, ask real questions, and flag corrections -- so the actual fine-tune (P10)
  is meant to be run *after* corrections accumulate, not before. The app is fully usable
  for grounded Q&A in the meantime; the self-improvement loop is the payoff of using it.

## Notes

- **Citations are first-class.** Every answer shows its sources; that is how you spot
  retrieval vs generation failures and decide what to add to the training set.
- **Embeddings are versioned.** If the embedding model changes, stored vectors must be
  rebuilt -- the schema records which embedder produced each vector.
- **Portable by construction.** Everything needed is bundled or auto-fetched; there is
  no global install step.
- **Content moderation -- nice-to-have, not built.** A third small "moderator" model
  (CPU-only, slotting in as a **moderation gate** beside the embed and gen gates -- one
  gate, two callers, just as the embed gate already serves both ingestion and retrieval)
  could screen two surfaces:
  - **Dropped files, at ingest time** -- reject inappropriate content (e.g. instructions
    for harming people) *before* extraction/embedding, so the corpus never holds it.
  - **Incoming chat messages** -- moderate the user's question. Run it in a **separate
    thread, concurrently with the normal RAG answer** (two threads: one moderating, one
    answering as usual), so moderation adds **no latency** to the common case. If the
    moderator flags the message, **retroactively delete the message and its answer**:
    cheap here because the transcript is just rows in `rag.db` and every client repaints
    by reading it -- delete the rows, bump the revision, and also **drop the exchange from
    the prompt-history window** so it can't poison later answers. The verdict is logged
    under the same request / chat-session id.

  This is **optimistic** by design (answer first, retract if flagged -- a flagged exchange
  may be briefly visible before it's pulled); a stricter variant would **block** the
  answer until the moderator clears, trading latency for never showing flagged content.
  Deferred -- noted here so the idea isn't lost; revisit after the core RAG loop works.
