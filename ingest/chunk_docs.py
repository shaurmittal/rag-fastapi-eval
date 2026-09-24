"""Strategy 2, docs half - header-aware markdown chunking.

Split each document at its headings, so a chunk is one section: a unit the
author already decided belongs together. Each chunk's text is prefixed with its
heading path ("Request Body > Create your data model"), because a section
body often never repeats the topic its heading names - "Use standard Python
types for all the attributes" says nothing about request bodies on its own.

Sections longer than MAX_CHARS are split further at paragraph boundaries.
The embedding model only reads the first ~512 tokens of any input; anything
beyond that is silently ignored, so oversized chunks lose content invisibly.

A "#" inside a fenced code block is a Python comment, not a heading. Since
ingest resolves include directives, code examples are inline and full of
them, so fence state has to be tracked.
"""

import re

from ingest.chunk import Chunk, make_chunk_id

MAX_CHARS = 1500  # ~375 tokens, comfortably inside the embedder's 512
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


def split_sections(text):
    """Return [(heading_path, body_text)], one entry per heading-delimited section."""
    sections = []
    path = []  # stack of (level, title)
    body = []
    in_fence = False

    def flush():
        content = "\n".join(body).strip()
        if content:
            sections.append((" > ".join(title for _, title in path), content))

    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
        heading = None if in_fence else HEADING_RE.match(line)
        if heading:
            flush()
            body = []
            level = len(heading[1])
            # Pop headings at the same or deeper level; this one replaces them.
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, heading[2]))
        else:
            body.append(line)
    flush()
    return sections


def split_paragraphs(text):
    """Split at blank lines, but never inside a code fence."""
    blocks, current, in_fence = [], [], False
    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def pack(blocks, max_chars):
    """Greedily pack consecutive blocks into pieces no longer than max_chars.

    A single block longer than max_chars (a long code example) is split by
    lines as a last resort.
    """
    pieces, current = [], ""
    for block in blocks:
        if len(block) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            lines, part = block.splitlines(), ""
            for line in lines:
                if part and len(part) + len(line) + 1 > max_chars:
                    pieces.append(part)
                    part = ""
                part = f"{part}\n{line}" if part else line
            if part:
                pieces.append(part)
            continue
        if current and len(current) + len(block) + 2 > max_chars:
            pieces.append(current)
            current = ""
        current = f"{current}\n\n{block}" if current else block
    if current:
        pieces.append(current)
    return pieces


def chunk_markdown(text, source_file):
    chunks = []
    for heading_path, body in split_sections(text):
        prefix = f"{heading_path}\n\n" if heading_path else ""
        for piece in pack(split_paragraphs(body), MAX_CHARS - len(prefix)):
            chunks.append(
                Chunk(
                    id=make_chunk_id("structured", source_file, len(chunks)),
                    text=prefix + piece,
                    source_file=source_file,
                    section=heading_path,
                    chunk_type="docs",
                    strategy="structured",
                )
            )
    return chunks
