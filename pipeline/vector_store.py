"""A thin interface over Qdrant.

Nothing outside this file imports qdrant_client. Retrieval, evaluation and
the app all talk to VectorStore's four methods, so swapping Qdrant for
another store means rewriting this file and nothing else.

Three ways to connect, chosen by environment:
  QDRANT_URL set          -> a Qdrant server (the Docker container, locally)
  data/qdrant_local/ here -> Qdrant's embedded on-disk mode, no server
  neither                 -> an in-memory store loaded from the portable
                             export in data/index/. This is what the deployed
                             app uses: Streamlit Cloud can't reach a container
                             on a laptop.

Each chunking strategy gets its own collection (fastapi_naive,
fastapi_structured) so the two indexes never mix.
"""

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from ingest.chunk import Chunk

load_dotenv()

LOCAL_PATH = "data/qdrant_local"


def collection_name(strategy):
    return f"fastapi_{strategy}"


class VectorStore:
    def __init__(self, url=None, path=None):
        url = url or os.getenv("QDRANT_URL")
        path = path or LOCAL_PATH
        if url:
            self.client = QdrantClient(url=url)
        elif os.path.isdir(path):
            self.client = QdrantClient(path=path)
        else:
            from pipeline.index import load_export

            self.client = QdrantClient(":memory:")
            load_export(self)

    def recreate(self, collection, dimensions):
        """Drop and recreate a collection, so re-indexing starts clean."""
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection,
            vectors_config=models.VectorParams(size=dimensions, distance=models.Distance.COSINE),
        )

    def upsert(self, collection, chunks, vectors, batch_size=256):
        """Store chunks with their vectors. The whole Chunk rides along as payload."""
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self.client.upsert(
                collection,
                points=[
                    models.PointStruct(
                        # Qdrant ids must be ints or UUIDs; our 16-hex id fits in a 64-bit int.
                        id=int(chunk.id, 16),
                        vector=vector.tolist(),
                        payload=chunk.__dict__,
                    )
                    for chunk, vector in zip(batch, vectors[start : start + batch_size])
                ],
            )

    def search(self, collection, vector, k):
        """Return the k nearest chunks as [(Chunk, cosine_similarity)], best first."""
        hits = self.client.query_points(collection, query=vector.tolist(), limit=k, with_payload=True).points
        return [(Chunk(**hit.payload), hit.score) for hit in hits]

    def count(self, collection):
        return self.client.count(collection).count

    def dump(self, collection):
        """Every chunk and its vector, in chunk-id order (for export)."""
        chunks, vectors, offset = [], [], None
        while True:
            points, offset = self.client.scroll(collection, limit=512, offset=offset,
                                                with_payload=True, with_vectors=True)
            for p in points:
                chunks.append(Chunk(**p.payload))
                vectors.append(p.vector)
            if offset is None:
                break
        order = sorted(range(len(chunks)), key=lambda i: chunks[i].id)
        return [chunks[i] for i in order], [vectors[i] for i in order]
