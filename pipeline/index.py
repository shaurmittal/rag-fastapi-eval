"""Embed both chunk sets and load them into the vector store.

RUN (after ingest.build_chunks)
    QDRANT_URL=http://localhost:6333 uv run python -m pipeline.index    # Docker server
    uv run python -m pipeline.index                                     # embedded, for deploy
"""

import time

from ingest.build_chunks import load_chunks
from pipeline.embed import DIMENSIONS, embed_passages
from pipeline.vector_store import VectorStore, collection_name


def main():
    store = VectorStore()
    for strategy in ("naive", "structured"):
        chunks = load_chunks(strategy)
        name = collection_name(strategy)

        started = time.perf_counter()
        vectors = embed_passages([c.text for c in chunks])
        embed_seconds = time.perf_counter() - started

        store.recreate(name, DIMENSIONS)
        store.upsert(name, chunks, vectors)
        print(f"{name:<22} {store.count(name):>5} vectors  (embedded in {embed_seconds:.0f}s)")


if __name__ == "__main__":
    main()
