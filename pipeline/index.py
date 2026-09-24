"""Build the vector index, and export/import it for deployment.

    uv run python -m pipeline.index           # embed chunks and load them into the store
    uv run python -m pipeline.index export    # write data/index/, a portable copy for deploy

The export is plain NumPy vectors plus gzipped chunk JSON. A deployed app
can't reach the Qdrant container on a laptop, and Qdrant's embedded on-disk
format is tied to the qdrant-client version that wrote it - so the app loads
the export into an in-memory store at startup instead (about a second for
3,000 vectors). Vectors stay float32, so the deployed app retrieves exactly
what the evaluation measured.
"""

import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

from ingest.build_chunks import load_chunks
from ingest.chunk import Chunk
from pipeline.embed import DIMENSIONS, embed_passages
from pipeline.vector_store import VectorStore, collection_name

STRATEGIES = ("naive", "structured")
EXPORT_DIR = Path("data/index")


def build():
    store = VectorStore()
    for strategy in STRATEGIES:
        chunks = load_chunks(strategy)
        name = collection_name(strategy)

        started = time.perf_counter()
        vectors = embed_passages([c.text for c in chunks])
        embed_seconds = time.perf_counter() - started

        store.recreate(name, DIMENSIONS)
        store.upsert(name, chunks, vectors)
        print(f"{name:<22} {store.count(name):>5} vectors  (embedded in {embed_seconds:.0f}s)")


def export():
    """Copy every collection out of the current store into data/index/."""
    store = VectorStore()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    for strategy in STRATEGIES:
        chunks, vectors = store.dump(collection_name(strategy))
        np.savez_compressed(EXPORT_DIR / f"{strategy}.npz", vectors=np.asarray(vectors, dtype=np.float32))
        with gzip.open(EXPORT_DIR / f"{strategy}.jsonl.gz", "wt") as f:
            for chunk in chunks:
                f.write(chunk.to_json() + "\n")
        print(f"exported {len(chunks)} {strategy} chunks")


def load_export(store):
    """Load data/index/ into `store` (used when no Qdrant server is configured)."""
    for strategy in STRATEGIES:
        vectors = np.load(EXPORT_DIR / f"{strategy}.npz")["vectors"]
        with gzip.open(EXPORT_DIR / f"{strategy}.jsonl.gz", "rt") as f:
            chunks = [Chunk.from_json(line) for line in f if line.strip()]
        name = collection_name(strategy)
        store.recreate(name, DIMENSIONS)
        store.upsert(name, chunks, vectors)


if __name__ == "__main__":
    export() if sys.argv[1:] == ["export"] else build()
