"""Chunk the pinned corpus with both strategies and write them to disk.

OUTPUT
    data/chunks/naive.jsonl
    data/chunks/structured.jsonl

These are derivable from the pinned corpus, so they're gitignored.

RUN
    uv run python -m ingest.build_chunks
"""

import statistics
from pathlib import Path

from ingest.chunk_code import chunk_python
from ingest.chunk_docs import chunk_markdown
from ingest.chunk_naive import chunk_text
from ingest.corpus import load_docs, load_source

OUT_DIR = Path("data/chunks")


def build(strategy):
    chunks = []
    for path, text in load_docs():
        chunks += chunk_text(text, path, "docs") if strategy == "naive" else chunk_markdown(text, path)
    for path, text in load_source():
        chunks += chunk_text(text, path, "code") if strategy == "naive" else chunk_python(text, path)
    return chunks


def write(chunks, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for chunk in chunks:
            f.write(chunk.to_json() + "\n")


def load_chunks(strategy):
    with open(OUT_DIR / f"{strategy}.jsonl") as f:
        from ingest.chunk import Chunk

        return [Chunk.from_json(line) for line in f if line.strip()]


def main():
    for strategy in ("naive", "structured"):
        chunks = build(strategy)
        write(chunks, OUT_DIR / f"{strategy}.jsonl")
        sizes = [len(c.text) for c in chunks]
        by_type = {t: sum(1 for c in chunks if c.chunk_type == t) for t in ("docs", "code")}
        print(
            f"{strategy:<11} {len(chunks):>5} chunks  "
            f"(docs {by_type['docs']}, code {by_type['code']})  "
            f"chars: median {statistics.median(sizes):.0f}, p95 {sorted(sizes)[int(len(sizes) * 0.95)]}, max {max(sizes)}"
        )


if __name__ == "__main__":
    main()
