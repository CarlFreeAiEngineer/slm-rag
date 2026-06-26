# PLAN.md -- building slm-rag in testable slices

The spec lives in [README.md](README.md). This file breaks the build into **phases that
each ship something you can test from the shell** -- `curl` against the HTTP API, the
`--cli` chat, or a standalone script. Nothing is "done" until its **acceptance test**
passes from a terminal, which is exactly the property the README promises ("no behavior
reachable only through the UI").

## How to read this

- Each phase lists **Build** (what to write), **Test** (the shell command that proves
  it), and **Depends** (which phases must be green first).
- **Independent phases can be built in parallel** -- they touch no models and no server:
  - **P1** SQLite + sqlite-vec store
  - **P2** text extraction + chunking
  - **P10** Colab fine-tune notebook (against a sample CSV)
- Everything else is a stack: each layer assumes the ones below it pass their tests.

```
P0 server skeleton ─┬─ P3 backends (2 gates) ─┬─ P4 ingest ─ P5 retrieve ─ P6 answer ─┬─ P7 CLI
                    │                          │                                       └─ P8 web UI
P1 store ───────────┘                          │            P9 corrections ─ P10 Colab ─ P11 staleness
P2 chunker ────────────────────────────────────┘
```

- **Tests are plain Python scripts -- no test framework, no `pytest`.** Each lives in
  `tests/` (e.g. `tests/p4_ingest.py`), runs with `uv run tests/p4_ingest.py`, prints
  what it checked, and **exits `0` on pass / non-zero on fail** (raise or
  `sys.exit(1)`). A `tests/run_all.py` runs each phase script in turn and exits non-zero
  if any did -- that is the whole harness. CI, if any, is just `uv run tests/run_all.py`.

---

## P0 -- Server skeleton & lifecycle

**Build.** `serve.py` that boots with no models: arg parsing (`--web` / `--cli` /
`--port` / `--check` / `--help`, defaults baked in, **no env vars**); binds `51548`;
serves a placeholder `index.html`; `GET /health` returns JSON; `SIGINT`/`SIGTERM`
handlers + localhost-only `/shutdown` (`POST`, and `GET` with the within-5-min timestamp
guard); the `./logs/` JSONL writer stamping every line with a **request id** and a **chat
session id**. Backends are stubbed; `--check` prints the planned backend layout and exits.

**Test.**
- `curl -s localhost:51548/health` -> `{"status": ...}`.
- `python serve.py --check` -> prints the plan, downloads nothing, exits 0.
- `curl -X POST localhost:51548/shutdown` -> process exits cleanly, no orphan procs.
- a `logs/<...>.log` file appears, one JSON object per line, each carrying a request id.

**Depends.** none.

---

## P1 -- SQLite schema + sqlite-vec store *(independent)*

**Build.** The `rag.db` schema: `documents`, `chunks`, a `vec0` virtual table (768-dim),
`messages`, `requests`, `state`, `corrections`. Confirm `sqlite-vec` loads as an
extension in the bundled Python. A self-test entry point (`serve.py --selftest-vec` or a
standalone script) that inserts known vectors and runs a k-NN.

**Test.** Insert 3 chunks with hand-built 768-dim vectors; query the nearest to a 4th ->
returns the expected ids in the expected order. Proves the extension loads and k-NN is
correct in the portable runtime.

**Depends.** none (parallel with P0, P2).

---

## P2 -- Text extraction + chunking *(independent, pure)*

**Build.** A module: file -> plain text (`.txt`, `.md`, `.pdf` via `pypdf`) -> chunks
(paragraph-aware, ~250-char target / chunks ~390 chars with 1-paragraph overlap; small
focused chunks so each embedding is about one idea), each with metadata for citations
(source path, chunk index, char offsets).

**Test.** `python ingest_lib.py samples/notes.pdf` prints N chunks; assert chunk
size/overlap are within tolerance and that a known sentence lands in the expected chunk
with correct offsets. No server, no models.

**Depends.** none (parallel with P0, P1).

---

## P3 -- Two backends, two gates (embed = CPU, gen = GPU)

**Build.** HF auto-download (smallest first) of nomic-embed + Phi GGUFs; ensure
`llama.cpp` present; launch the embedder (`--n-gpu-layers 0`, CPU) and Phi (GPU if it
fits, else CPU). The **embed gate** and **gen gate** (each a lock/worker); ingestion
batches yield the embed gate. Expose debug endpoints on `serve.py`: `POST /embed
{text}` -> vector, and a minimal generate path.

**Test.**
- `curl -s localhost:51548/embed -d '{"text":"hello"}'` -> 768 floats.
- a minimal generate call -> a non-empty completion.
- the embedder is CPU-only (startup log says `ngl=0`; `nvidia-smi` shows only Phi when a
  GPU is present); Phi reports GPU offload when it fits.

**Depends.** P0.

---

## P4 -- Ingestion pipeline (drop -> searchable)

**Build.** `POST /ingest` (file upload): store the raw bytes as a **blob in `rag.db`**
(no `ragdocs/` directory) -> extract (P2) -> chunk (P2) -> embed via the embed gate (P3),
batched + yielding -> store chunks/vectors (P1). Status moves *vectorizing* -> *ready*.
`GET /tree` lists the corpus; `GET /doc?path=` previews text; `POST /delete {path}`
removes a document and cascades its chunks/vectors.

**Test.** `curl -F file=@samples/notes.md localhost:51548/ingest` -> 200; poll
`GET /tree` until the file shows *ready* with N chunks; `GET /doc?path=notes.md` returns
the text; row counts in `rag.db` match N.

**Depends.** P1, P2, P3.

---

## P5 -- Retrieval (question -> chunks)

**Build.** `POST /retrieve {question, scope?}`: embed the question (embed gate) ->
`sqlite-vec` k-NN -> return top chunks with citation metadata; optional folder/file
scope.

**Test.** Ingest a known doc, then `curl /retrieve -d '{"question":"<answer is in
chunk 7>"}'` -> chunk 7 is the top hit; a `scope` filter restricts hits to that
folder/file. Assert top-1 on a deterministic corpus.

**Depends.** P4.

---

## P6 -- Grounded answer + citations

**Build.** The chat flow through the `requests` queue + worker + gen gate: embed ->
retrieve -> build the grounded prompt -> Phi streams an answer that **cites its sources**
or says **"I don't know"** when the context lacks the answer. Persist to `messages` with
the request id + chat session id; log the whole chain.

**Test.**
- `curl /enqueue -d '{"kind":"chat","content":"<in-corpus question>"}'`, poll
  `/request?id=`, then `GET /history` -> answer grounded in the ingested doc, citation
  points to the right file + chunk.
- an out-of-corpus question -> "I don't know."
- `grep <request id> logs/*.log` -> the `embed_*` and `gen_*` request/response lines for
  that exchange; `grep <chat session id>` -> the whole conversation.

**Depends.** P5.

---

## P7 -- CLI chat

**Build.** `--cli` REPL as a thin HTTP client of the running server: ask a question,
`/clear`, `/help`, `/quit`; citations printed inline. Same gates as the web UI.

**Test.** Pipe questions on stdin to `python serve.py --cli`; assert grounded answers +
citations in stdout; `/clear` empties the transcript (verify via `GET /history`).

**Depends.** P6.

---

## P8 -- Web UI (two panels)

**Build.** `index.html`: left = file tree (drag-drop ingest, *vectorizing* -> *ready*
status, click to preview); right = chat (streamed answers, click-through citations, a
*"this is wrong -> fix it"* control). Pure client of the documented API -- **no private
endpoints**.

**Test.** Because the UI is only the API, the P4-P6 phase scripts already cover the
behavior. For the one non-curl surface, a `tests/p8_ui.py` script drives a headless
browser (Playwright driven from plain Python -- still just exit 0 / non-zero, no test
framework): drag a file in, watch it go *ready*, ask a question, assert a clickable
citation is present. This is the only phase whose acceptance is partly visual -- which is
why we kept the UI thin.

**Depends.** P4, P5, P6 (stable API).

---

## P9 -- Correction capture & training export

**Build.** `POST /correct {question, context, answer}` -> store the correction in
SQLite. `GET /training.csv` (or `serve.py --export-training`) -> emit
`training/rag_finetune.csv`.

**Test.** Post a correction, export, assert the CSV row matches and has the columns
`(question, context, answer)`; the corrections table grows by one.

**Depends.** P6.

---

## P10 -- Colab fine-tune notebook *(independent)*

**Build.** `training/finetune_phi_rag.ipynb`: load the CSV -> Unsloth LoRA fine-tune of
Phi-4-mini -> export Q4_K_M GGUF -> upload to Hugging Face.

**Test.** Run on Colab against a tiny sample CSV -> produces a GGUF and an HF upload.
Developable in parallel from a hand-made sample CSV (only the CSV format from P9 is
needed, not a running server).

**Depends.** P9 (CSV columns) -- but parallelizable against a sample.

---

## P11 -- Staleness refresh & model pickup

**Build.** The HF staleness check (size + sha256 sidecar): on restart, re-fetch any
weights whose HF copy changed, so a freshly fine-tuned Phi is served after a restart.

**Test.** Point Phi at an updated HF file (or simulate a changed sha), restart, assert
the new weights download and serve (the served model's hash/id changes). With the network
unreachable, the cached copy is kept (offline-graceful).

**Depends.** P3, P10.

---

## Milestones

- **M1 "retrieval works"** = P0 + P1 + P2 + P3 + P4 + P5. You can ingest and retrieve
  from the shell; no generation yet.
- **M2 "it answers"** = + P6 + P7. Grounded, cited answers via curl and the CLI.
- **M3 "usable"** = + P8 + P9. The browser two-panel app and correction capture.
- **M4 "self-improving"** = + P10 + P11. The fine-tune loop closes; restart serves the
  improved model.
