# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pypdf>=4.0",
# ]
# ///
"""
ingest_lib.py -- Text extraction and chunking for slm-rag (Phase P2).

Pure module: no server, no models, no DB.  Import and call directly, or run
as a CLI for manual inspection:

    uv run ingest_lib.py <file>

Functions
---------
extract_text(path) -> str
    Read supported files (.txt, .md, .pdf) and return plain UTF-8 text.
    Raises ValueError for unsupported extensions.

chunk_text(text, target_chars=2200, overlap_paras=1) -> list[dict]
    Paragraph-aware chunker (~2200 chars + 1-para overlap).
    Chosen from the Colab A100 chunking sweep: keeps dialogue/Q&A exchanges
    intact so the small model can extract an answer stated indirectly,
    while keeping retrieval discrimination sharp.

    Each dict has:
        chunk_index : int   -- 0-based position in the chunk sequence
        text        : str   -- the chunk's text content
        char_start  : int   -- byte offset of the first character in the
                               original text (inclusive)
        char_end    : int   -- byte offset just past the last character
                               (exclusive), i.e. text[char_start:char_end]
                               recovers the chunk text exactly.

ingest_file(path) -> list[dict]
    extract_text + chunk_text, with 'source_path' added to every chunk.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_bytes(filename_or_ext: str, data: bytes) -> str:
    """
    Extract plain text from raw *data* bytes.

    Parameters
    ----------
    filename_or_ext : the filename or just the extension (e.g. 'doc.md' or '.md')
    data            : raw file bytes

    Supported extensions:
        .txt, .md   -- decoded as UTF-8 (errors='replace')
        .pdf        -- text extracted via pypdf from an in-memory BytesIO

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    suffix = Path(filename_or_ext).suffix.lower()

    if suffix in (".txt", ".md"):
        return data.decode("utf-8", errors="replace")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "pypdf is required for PDF extraction. "
                "Run this script with `uv run ingest_lib.py` so the inline "
                "dependency is auto-installed."
            ) from exc

        reader = PdfReader(io.BytesIO(data))
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)

    raise ValueError(
        f"Unsupported file type: '{suffix}'. "
        "Supported extensions: .txt, .md, .pdf"
    )


def extract_text(path: str | Path) -> str:
    """
    Extract plain text from *path*.

    Supported extensions:
        .txt, .md   -- read as UTF-8 text (fallback to latin-1 on decode error)
        .pdf        -- extract text via pypdf (page separator: newline)

    Raises
    ------
    ValueError
        If the file extension is not supported.
    FileNotFoundError
        If the file does not exist.

    Delegates to extract_text_from_bytes so both paths share one implementation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data = path.read_bytes()
    return extract_text_from_bytes(path.name, data)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    target_chars: int = 2200,
    overlap_paras: int = 1,
) -> list[dict[str, Any]]:
    """
    Paragraph-aware chunker: ~2200 chars + 1-para overlap.

    Chosen from the Colab A100 chunking sweep (2025-06): keeping whole
    paragraphs together preserves dialogue / Q&A exchanges so the small
    model can extract an answer stated indirectly, while keeping chunks
    short enough (< 2048 embedder tokens) for sharp retrieval.

    Strategy
    --------
    1. Split the full text on blank-line paragraph boundaries
       (re.split(r'\\n\\s*\\n', text)).
    2. Greedily accumulate consecutive paragraphs until the accumulated
       length >= target_chars, then emit a chunk.  Carry the last
       *overlap_paras* paragraphs forward into the next chunk.
    3. If a single paragraph alone exceeds ~1.5 * target_chars (3300 chars
       by default), sentence-split THAT paragraph (regex ``(?<=[.!?])\\s+``)
       and pack its sentences to the target so no chunk is enormous.
    4. Track exact char_start / char_end offsets into the original *text*
       so that ``text[chunk["char_start"] : chunk["char_end"]] == chunk["text"]``.

    Parameters
    ----------
    text          : the full document text
    target_chars  : target chunk length in characters (default 2200,
                    ~500-600 tokens on typical English prose)
    overlap_paras : number of tail paragraphs carried into the next chunk
                    as overlap (default 1)

    Returns
    -------
    List of dicts with keys: chunk_index, text, char_start, char_end.
    Returns an empty list if *text* is empty or whitespace-only.
    """
    if not text.strip():
        return []

    SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
    PARA_SPLIT     = re.compile(r'\n\s*\n')
    MAX_PARA_CHARS = int(target_chars * 1.5)   # ~3300 chars -- split if exceeded

    # ------------------------------------------------------------------ #
    # Step 1: locate paragraph spans in the original text.               #
    # We keep (start, end) offsets so we can reconstruct char positions. #
    # ------------------------------------------------------------------ #
    para_spans: list[tuple[int, int]] = []   # (char_start, char_end) per para
    prev_end = 0
    for m in PARA_SPLIT.finditer(text):
        seg_start = prev_end
        seg_end   = m.start()
        para_text = text[seg_start:seg_end]
        if para_text.strip():
            # trim leading/trailing whitespace within the span
            lstrip = len(para_text) - len(para_text.lstrip())
            rstrip = len(para_text) - len(para_text.rstrip())
            real_start = seg_start + lstrip
            real_end   = seg_end   - rstrip
            if real_start < real_end:
                para_spans.append((real_start, real_end))
        prev_end = m.end()
    # Last paragraph (after final blank line, or the whole text if no blank lines)
    tail = text[prev_end:]
    if tail.strip():
        lstrip = len(tail) - len(tail.lstrip())
        rstrip = len(tail) - len(tail.rstrip())
        real_start = prev_end + lstrip
        real_end   = len(text) - rstrip
        if real_start < real_end:
            para_spans.append((real_start, real_end))

    if not para_spans:
        return []

    # ------------------------------------------------------------------ #
    # Step 2: if any paragraph exceeds MAX_PARA_CHARS, sub-split it       #
    # into sentence-packed spans so no chunk is enormous.                 #
    # ------------------------------------------------------------------ #
    def _sentence_spans(p_start: int, p_end: int) -> list[tuple[int, int]]:
        """Break a large paragraph into sentence-packed sub-spans."""
        para_text = text[p_start:p_end]
        sentences = SENTENCE_SPLIT.split(para_text)
        spans: list[tuple[int, int]] = []
        bucket_start_in_para = 0
        bucket_len = 0
        bucket_first = 0   # index of first sentence in current bucket
        # re-locate each sentence's start within para_text
        sent_starts: list[int] = []
        pos = 0
        for s in sentences:
            idx = para_text.find(s, pos)
            sent_starts.append(idx)
            pos = idx + len(s)

        for si, s in enumerate(sentences):
            s_len = len(s)
            if bucket_len + s_len > target_chars and bucket_len > 0:
                # emit current bucket
                bs = p_start + sent_starts[bucket_first]
                be = p_start + sent_starts[si - 1] + len(sentences[si - 1])
                spans.append((bs, be))
                bucket_first = si
                bucket_len = 0
            bucket_len += s_len + 1  # +1 for separator

        # emit remaining
        if bucket_first < len(sentences):
            bs = p_start + sent_starts[bucket_first]
            be = p_end
            spans.append((bs, be))
        return spans if spans else [(p_start, p_end)]

    expanded: list[tuple[int, int]] = []
    for ps, pe in para_spans:
        if (pe - ps) > MAX_PARA_CHARS:
            expanded.extend(_sentence_spans(ps, pe))
        else:
            expanded.append((ps, pe))

    para_spans = expanded

    # ------------------------------------------------------------------ #
    # Step 3: greedily accumulate spans into chunks.                      #
    # ------------------------------------------------------------------ #
    chunks: list[dict[str, Any]] = []
    chunk_index = 0
    i = 0
    n_spans = len(para_spans)

    while i < n_spans:
        acc_start = para_spans[i][0]
        acc_end   = para_spans[i][1]
        j = i + 1
        while j < n_spans:
            candidate_end = para_spans[j][1]
            # include the gap between spans (whitespace) in the length count
            accumulated = candidate_end - acc_start
            if accumulated >= target_chars:
                # include this span to hit/exceed the target, then stop
                acc_end = candidate_end
                j += 1
                break
            acc_end = candidate_end
            j += 1

        # Recover exact text (including any whitespace between paragraphs
        # that fell inside the span range) -- just slice the original text.
        chunk_str = text[acc_start:acc_end]

        # Re-trim leading/trailing whitespace for a clean chunk text, but
        # keep the offsets pointing at the non-whitespace content.
        lstrip = len(chunk_str) - len(chunk_str.lstrip())
        rstrip = len(chunk_str) - len(chunk_str.rstrip())
        real_cs = acc_start + lstrip
        real_ce = acc_end   - rstrip
        chunk_str = text[real_cs:real_ce]

        if chunk_str:
            chunks.append({
                "chunk_index": chunk_index,
                "text":        chunk_str,
                "char_start":  real_cs,
                "char_end":    real_ce,
            })
            chunk_index += 1

        # Overlap: next chunk starts overlap_paras before j.
        next_start = max(i + 1, j - overlap_paras)
        if next_start <= i:
            next_start = i + 1   # safety: always advance
        i = next_start

    return chunks


# ---------------------------------------------------------------------------
# ingest_file
# ---------------------------------------------------------------------------

def ingest_file(path: str | Path) -> list[dict[str, Any]]:
    """
    Extract text from *path* and return its chunks, each with 'source_path'.

    Combines extract_text() + chunk_text(), tagging every chunk dict with:
        source_path : str  -- str(path) as passed (normalised to forward slashes)
    """
    path = Path(path)
    text = extract_text(path)
    chunks = chunk_text(text)
    source = str(path)
    for chunk in chunks:
        chunk["source_path"] = source
    return chunks


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run ingest_lib.py <file>", file=sys.stderr)
        sys.exit(1)

    target = Path(sys.argv[1])
    print(f"Ingesting: {target}")

    try:
        chunks = ingest_file(target)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Produced {len(chunks)} chunk(s).\n")

    for c in chunks:
        preview = c["text"][:120].replace("\n", " ")
        print(
            f"  [{c['chunk_index']:3d}] chars {c['char_start']:6d}-{c['char_end']:6d} "
            f"| {preview!r}"
        )


if __name__ == "__main__":
    _main()
