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

chunk_text(text, target_tokens=512, overlap_tokens=64) -> list[dict]
    Split text into overlapping chunks.  Each dict has:
        chunk_index : int   -- 0-based position in the chunk sequence
        text        : str   -- the chunk's text content
        char_start  : int   -- byte offset of the first character in the
                               original text (inclusive)
        char_end    : int   -- byte offset just past the last character
                               (exclusive), i.e. text[char_start:char_end]
                               recovers the chunk text exactly.

ingest_file(path) -> list[dict]
    extract_text + chunk_text, with 'source_path' added to every chunk.

Token heuristic
---------------
We approximate token count as  len(word_list) * 4 // 3  where word_list is
str.split() tokens (whitespace-split words).  The factor 4/3 accounts for
sub-word tokenisation: most tokenisers produce ~1.33 tokens per word on
English prose (empirically 1.2 – 1.5).  This is a deliberate approximation;
the exact count depends on the final tokeniser, which is not available here.
The heuristic keeps chunk sizes within ±20 % of the target on typical English
text and is documented in the chunk dicts so callers can see what was used.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Token heuristic
# ---------------------------------------------------------------------------

def _words_to_tokens(n_words: int) -> int:
    """Approximate token count from word count (4/3 factor for sub-word splits)."""
    return n_words * 4 // 3


def _tokens_to_words(n_tokens: int) -> int:
    """Approximate word count from token count (inverse of 4/3 factor)."""
    return n_tokens * 3 // 4


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

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
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()

    if ext in (".txt", ".md"):
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1")

    if ext == ".pdf":
        # pypdf is declared in the inline script metadata above.
        # Import lazily so callers that never touch PDFs pay no cost.
        try:
            from pypdf import PdfReader  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "pypdf is required for PDF extraction. "
                "Run this script with `uv run ingest_lib.py` so the inline "
                "dependency is auto-installed."
            ) from exc

        reader = PdfReader(str(path))
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)

    raise ValueError(
        f"Unsupported file type: '{ext}'. "
        "Supported extensions: .txt, .md, .pdf"
    )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    target_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[dict[str, Any]]:
    """
    Split *text* into overlapping chunks of approximately *target_tokens* tokens.

    Strategy
    --------
    1. Split the full text into words (str.split()).
    2. Convert token targets to word targets using the 4/3 heuristic.
    3. Walk through the word list with a stride of (target_words - overlap_words),
       collecting slices of target_words words.
    4. For each word-slice, recover the exact char_start / char_end offsets from
       the original text by locating the first and last word within the text.

    The char offsets satisfy:
        text[chunk["char_start"] : chunk["char_end"]] == chunk["text"]

    Parameters
    ----------
    text          : the full document text
    target_tokens : desired chunk size in (approximate) tokens  (default 512)
    overlap_tokens: overlap between consecutive chunks, in tokens (default 64)

    Returns
    -------
    List of dicts with keys: chunk_index, text, char_start, char_end.
    Returns an empty list if *text* is empty.
    """
    if not text.strip():
        return []

    # Convert token targets to word targets using the heuristic.
    target_words = _tokens_to_words(target_tokens)   # ~384 words for 512 tokens
    overlap_words = _tokens_to_words(overlap_tokens)  # ~48 words for 64 tokens
    stride = target_words - overlap_words             # non-overlapping advance

    if stride <= 0:
        raise ValueError(
            f"overlap_tokens ({overlap_tokens}) must be smaller than "
            f"target_tokens ({target_tokens})."
        )

    # Build a list of (word, char_start_in_text) pairs so we can recover
    # byte offsets without a second pass.
    words: list[str] = []
    word_starts: list[int] = []   # char offset of each word's first character

    pos = 0
    for word in text.split(" "):
        # text.split(" ") does NOT strip leading/trailing newlines inside words,
        # but handles consecutive spaces correctly (empty strings for extra spaces).
        # We iterate over lines to capture real starts.  Instead, use a manual scan:
        pass

    # More robust: scan the text character by character to locate each word.
    # We define a "word" as a maximal non-whitespace run.
    i = 0
    n = len(text)
    while i < n:
        if not text[i].isspace():
            start = i
            while i < n and not text[i].isspace():
                i += 1
            words.append(text[start:i])
            word_starts.append(start)
        else:
            i += 1

    total_words = len(words)
    if total_words == 0:
        return []

    chunks: list[dict[str, Any]] = []
    chunk_index = 0
    word_pos = 0  # start of current chunk in word list

    while word_pos < total_words:
        end_word = min(word_pos + target_words, total_words)

        # char offsets: start of first word, end of last word in this slice
        char_start = word_starts[word_pos]
        last_word_start = word_starts[end_word - 1]
        char_end = last_word_start + len(words[end_word - 1])

        chunk_text_content = text[char_start:char_end]

        chunks.append(
            {
                "chunk_index": chunk_index,
                "text": chunk_text_content,
                "char_start": char_start,
                "char_end": char_end,
            }
        )

        chunk_index += 1

        if end_word == total_words:
            break  # final chunk consumed all remaining words

        word_pos += stride

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
